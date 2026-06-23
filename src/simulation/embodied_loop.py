"""Embodied agent: unified loop integrating LRSG + locomotion + physics + reward."""
import numpy as np
import torch
from typing import Dict, Tuple
from src.config import LRSGConfig, LocomotionConfig, EmbodiedConfig
from src.lrsg.graph import LRSGraph
from src.simulation.locomotion import LocomotionController
from src.simulation.navigation import NavigationEnvironment
from src.lrsg.embodied_reward import compute_embodied_reward


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

        # Create navigation environment
        self.env = NavigationEnvironment(
            grid_size=embodied_config.grid_size,
            num_obstacles=embodied_config.num_obstacles,
            seed=None  # Randomized each episode
        )

        # Episode state
        self.agent_pos = None
        self.agent_yaw = 0.0
        self.goal_pos = None
        self.prev_distance = None
        self.tick_count = 0
        self.episode_rewards = []

        # Reset for first episode
        self._reset_episode()

    def _reset_episode(self):
        """Start new episode."""
        self.agent_pos, self.goal_pos = self.env.reset()
        self.agent_yaw = 0.0
        self.prev_distance = np.linalg.norm(
            np.array(self.agent_pos) - np.array(self.goal_pos)
        )
        self.tick_count = 0
        self.episode_rewards = []

    def _build_observation(self) -> np.ndarray:
        """
        Build 18-dim observation vector:
        - Position (3): x, y, z
        - Velocity (3): vx, vy, vz (currently 0, will be used in future)
        - LiDAR (8): distance rays
        - Goal delta (2): (goal_x - agent_x), (goal_y - agent_y)
        - Distance to goal (1): scalar distance
        - In safe zone (1): binary flag (currently always 0 for grid env)
        """
        lidar = self.env.get_lidar(
            agent_pos=self.agent_pos,
            num_rays=self.emb_config.lidar_rays
        )

        goal_delta_x = self.goal_pos[0] - self.agent_pos[0]
        goal_delta_y = self.goal_pos[1] - self.agent_pos[1]
        dist_to_goal = np.linalg.norm(
            np.array(self.agent_pos) - np.array(self.goal_pos)
        )

        obs = np.concatenate([
            self.agent_pos,  # 3
            [0.0, 0.0, 0.0],  # velocity (3) — todo: integrate from motors
            lidar,  # 8
            [goal_delta_x, goal_delta_y],  # 2
            [dist_to_goal],  # 1
            [0.0],  # in_safe_zone (1)
        ])

        return obs.astype(np.float32)

    def step(self) -> Dict[str, float]:
        """
        Single simulation step:
        1. Build observation from environment
        2. LRSG forward pass → action logits
        3. Sample discrete action (MOVE_FWD, TURN_LEFT, TURN_RIGHT, IDLE)
        4. Map action to motor commands
        5. Update agent position via kinematics
        6. Compute reward (distance-based, collision, goal)
        7. Hebbian plasticity update (if learning enabled)
        8. Reset episode if terminal

        Returns:
            Dict with keys: 'reward', 'distance_to_goal', 'is_collision', 'is_goal_reached'
        """
        # Get observation
        obs = self._build_observation()

        # LRSG forward pass
        # TODO: integrate with actual LRSG logic for action selection
        # For now, sample random action (placeholder)
        action = np.random.randint(0, 4)  # 0=MOVE_FWD, 1=TURN_LEFT, 2=TURN_RIGHT, 3=IDLE

        # Map discrete action to motor commands
        if action == 0:  # MOVE_FWD
            left_motor, right_motor = 0.8, 0.8
        elif action == 1:  # TURN_LEFT
            left_motor, right_motor = 0.3, 0.8
        elif action == 2:  # TURN_RIGHT
            left_motor, right_motor = 0.8, 0.3
        else:  # IDLE
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

        # Collision check
        is_collision = self.env.check_collision(new_pos, radius=0.1)

        # Only update position if no collision
        if not is_collision:
            self.agent_pos = new_pos

        # Check if goal reached
        curr_distance = np.linalg.norm(
            np.array(self.agent_pos) - np.array(self.goal_pos)
        )
        is_goal_reached = curr_distance < self.emb_config.goal_reach_threshold

        # Compute reward
        reward = compute_embodied_reward(
            agent_pos=self.agent_pos,
            goal_pos=self.goal_pos,
            prev_distance=self.prev_distance,
            is_collision=is_collision,
            is_goal_reached=is_goal_reached
        )

        # Track for learning
        self.episode_rewards.append(reward)
        self.prev_distance = curr_distance
        self.tick_count += 1

        # Episode termination
        episode_done = self.tick_count >= self.emb_config.episode_length or is_goal_reached

        if episode_done:
            # TODO: Hebbian plasticity update on accumulated reward
            self._reset_episode()

        return {
            'reward': float(reward),
            'distance_to_goal': float(curr_distance),
            'is_collision': bool(is_collision),
            'is_goal_reached': bool(is_goal_reached),
        }
