"""Embodied agent: unified loop integrating LRSG + locomotion + physics + reward."""
import numpy as np
import torch
from typing import Dict, Tuple
from src.config import LRSGConfig, LocomotionConfig, EmbodiedConfig
from src.lrsg.graph import LRSGraph
from src.lrsg.forward import forward_pass
from src.lrsg.plasticity import hebbian_update
from src.simulation.locomotion import LocomotionController
from src.simulation.battleground_env import BattlegroundEnvironment
from src.simulation.semantic_encoder import SemanticEncoder
from src.lrsg.embodied_reward import compute_embodied_reward_enhanced
from src.simulation.combat_system import resolve_combat_tick, compute_agent_strength


class EmbodiedAgent:
    """
    Single embodied agent in physics simulation.

    Runs LRSG forward pass to select motor commands,
    which drive differential-drive robot in procedural grid environment.
    Learns online via Hebbian plasticity on shaped distance-to-goal reward.
    """

    def __init__(
        self,
        lrsg_config: LRSGConfig,
        locomotion_config: LocomotionConfig,
        embodied_config: EmbodiedConfig,
        device: str = 'cpu'
    ):
        """
        Args:
            lrsg_config: LRSG neural architecture config
            locomotion_config: Differential-drive parameters
            embodied_config: Navigation environment config
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.emb_config = embodied_config
        self.loc_config = locomotion_config
        self.lrsg_config = lrsg_config

        # Create LRSG brain
        self.lrsg = LRSGraph(lrsg_config, self.device)

        # Create locomotion controller
        self.locomotion = LocomotionController(
            wheelbase=locomotion_config.wheelbase,
            wheel_radius=locomotion_config.wheel_radius,
            max_motor_speed=locomotion_config.max_motor_speed,
            friction_coeff=locomotion_config.friction_coeff
        )

        # Create battleground environment
        self.env = BattlegroundEnvironment(
            grid_size=embodied_config.grid_size,
            num_obstacles=embodied_config.num_obstacles,
            num_enemies=3,
            seed=None  # Randomized each episode
        )

        # Semantic encoder
        self.semantic_encoder = SemanticEncoder()

        # Episode state
        self.agent_pos = None
        self.agent_yaw = 0.0
        self.goal_pos = None
        self.prev_distance = None
        self.tick_count = 0
        self.episode_rewards = []
        self._last_action: int = None

        # Combat tracking
        self._agent_accumulated_reward = 0.0
        self._prev_accumulated_reward = 0.0

        # Reset for first episode
        self._reset_episode()

    def _reset_episode(self):
        """Start new episode."""
        start_2d, goal_2d = self.env.reset()
        # Convert 2D positions to 3D (z = 0)
        self.agent_pos = (start_2d[0], start_2d[1], 0.0)
        self.goal_pos = (goal_2d[0], goal_2d[1], 0.0)
        self.agent_yaw = 0.0
        self.prev_distance = np.linalg.norm(
            np.array(self.agent_pos) - np.array(self.goal_pos)
        )
        self.tick_count = 0
        self.episode_rewards = []
        self._last_action = None
        # Reset combat reward tracking
        self._agent_accumulated_reward = 0.0
        self._prev_accumulated_reward = 0.0

    def _build_observation(self) -> np.ndarray:
        """
        Build 64-dim semantic state vector.

        First builds raw 18D observation, then encodes to 64D semantic state:
        - Raw observation (18D): Position, velocity, LiDAR, goal delta, distance to goal, safe zone
        - Enemies: List of alive EnemyNPC objects
        - Accumulated reward: Episode reward sum
        - Previous accumulated reward: Reward sum from previous tick

        Returns 64D semantic state vector.
        """
        # Build raw 18D observation
        lidar = self.env.get_lidar(
            agent_pos=self.agent_pos,  # 3D position (x, y, z)
            num_rays=self.emb_config.lidar_rays
        )

        # Use 2D positions for goal delta (ignore z)
        goal_delta_x = self.goal_pos[0] - self.agent_pos[0]
        goal_delta_y = self.goal_pos[1] - self.agent_pos[1]
        dist_to_goal = np.linalg.norm(
            np.array(self.agent_pos[:2]) - np.array(self.goal_pos[:2])
        )

        raw_obs_18d = np.concatenate([
            self.agent_pos,  # 3
            [0.0, 0.0, 0.0],  # velocity (3) — todo: integrate from motors
            lidar,  # 8
            [goal_delta_x, goal_delta_y],  # 2
            [dist_to_goal],  # 1
            [0.0],  # in_safe_zone (1)
        ]).astype(np.float32)

        # Get alive enemies
        enemies = self.env.get_enemies()

        # Encode to 64D semantic state
        semantic_state = self.semantic_encoder.encode(
            raw_obs=raw_obs_18d,
            enemies=enemies,
            accumulated_reward=self._agent_accumulated_reward,
            prev_accumulated_reward=self._prev_accumulated_reward
        )

        return semantic_state

    def step(self) -> Dict[str, float]:
        """
        Single simulation step:
        1. Build 64D semantic observation from environment
        2. LRSG forward pass → action logits (5 actions)
        3. Sample discrete action (MOVE_FWD, TURN_LEFT, TURN_RIGHT, IDLE, ATTACK)
        4. Map action to motor commands
        5. Update agent position via kinematics
        6. Tick enemies (update behaviors based on agent proximity)
        7. Resolve combat if action is ATTACK
        8. Compute enhanced reward (navigation + combat)
        9. Hebbian plasticity update (if learning enabled)
        10. Reset episode if terminal

        Returns:
            Dict with keys: 'reward', 'distance_to_goal', 'is_collision', 'is_goal_reached', 'action', 'num_enemies'
        """
        # Get 64D semantic observation
        obs = self._build_observation()

        # LRSG action selection (5 actions: 0=FWD, 1=LEFT, 2=RIGHT, 3=IDLE, 4=ATTACK)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self.lrsg.reset_activations()
        self.lrsg.inject_input(obs_tensor)        # writes obs[0:64] into neurons 0–63
        forward_pass(self.lrsg, self.lrsg_config, last_action=self._last_action, energy=200.0)
        output_acts = self.lrsg.read_output()     # shape [num_output_neurons]
        action = int(torch.argmax(output_acts[:5]).item())  # 5 actions now
        self._last_action = action

        # Map discrete action to motor commands
        if action == 0:  # MOVE_FWD
            left_motor, right_motor = 0.8, 0.8
        elif action == 1:  # TURN_LEFT
            left_motor, right_motor = 0.3, 0.8
        elif action == 2:  # TURN_RIGHT
            left_motor, right_motor = 0.8, 0.3
        elif action == 3:  # IDLE
            left_motor, right_motor = 0.0, 0.0
        else:  # ATTACK (action == 4)
            left_motor, right_motor = 0.0, 0.0

        # Locomotion: motors → velocities
        linear_vel, angular_vel = self.locomotion.motors_to_velocity(left_motor, right_motor)

        # Integrate kinematics over one time step
        dt = 1.0 / self.emb_config.physics_freq
        dx, dy, dyaw = self.locomotion.step(linear_vel, angular_vel, dt)

        # Update agent pose (in world frame)
        old_pos = self.agent_pos
        cos_yaw = np.cos(self.agent_yaw)
        sin_yaw = np.sin(self.agent_yaw)
        x_world = cos_yaw * dx - sin_yaw * dy
        y_world = sin_yaw * dx + cos_yaw * dy

        new_pos = (
            self.agent_pos[0] + x_world,
            self.agent_pos[1] + y_world,
            self.agent_pos[2]  # z unchanged (no vertical movement)
        )
        self.agent_yaw += dyaw

        # Collision check (check_collision expects 2D position)
        is_collision = self.env.check_collision((new_pos[0], new_pos[1]), radius=0.1)

        # Only update position if no collision
        if not is_collision:
            self.agent_pos = new_pos

        # Tick enemies: update behaviors based on agent proximity
        # Pass 2D position (x, y)
        self.env.tick_enemies(agent_pos=self.agent_pos[:2])

        # Resolve combat if action is ATTACK
        combat_result = None
        if action == 4:  # ATTACK
            nearby_enemies = self.env.get_enemy_near(self.agent_pos[:2], radius=2.0)
            if nearby_enemies:
                nearest = min(nearby_enemies, key=lambda e: np.linalg.norm(
                    np.array(e.position) - np.array(self.agent_pos[:2])
                ))
                agent_strength = compute_agent_strength(self._agent_accumulated_reward)
                combat_result = resolve_combat_tick(agent_strength, nearest)
                if combat_result["victory"]:
                    self.env.remove_enemy(nearest)

        # Check if goal reached (use 2D distance only)
        curr_distance = np.linalg.norm(
            np.array(self.agent_pos[:2]) - np.array(self.goal_pos[:2])
        )
        is_goal_reached = curr_distance < self.emb_config.goal_reach_threshold

        # Compute enhanced reward (navigation + combat)
        # Pass 2D positions to reward function
        reward = compute_embodied_reward_enhanced(
            agent_pos=self.agent_pos[:2],
            goal_pos=self.goal_pos[:2],
            prev_distance=self.prev_distance,
            is_collision=is_collision,
            is_goal_reached=is_goal_reached,
            combat_result=combat_result
        )

        # Track reward for combat strength calculation
        self._prev_accumulated_reward = self._agent_accumulated_reward
        self._agent_accumulated_reward += reward

        # Track for learning
        self.episode_rewards.append(reward)
        self.prev_distance = curr_distance
        self.tick_count += 1

        # Episode termination
        episode_done = self.tick_count >= self.emb_config.episode_length or is_goal_reached

        if episode_done:
            accumulated_reward = float(sum(self.episode_rewards))
            if accumulated_reward > 0:
                hebbian_update(self.lrsg, self.lrsg_config,
                               reward=accumulated_reward, learning_rate=0.01)
            self.lrsg.eligibility_trace.zero_()   # clear trace between episodes
            self._reset_episode()

        return {
            'reward': float(reward),
            'distance_to_goal': float(curr_distance),
            'is_collision': bool(is_collision),
            'is_goal_reached': bool(is_goal_reached),
            'action': int(action),
            'num_enemies': len(self.env.get_enemies()),
        }
