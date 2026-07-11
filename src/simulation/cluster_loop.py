"""
Cluster simulation loop.

Replaces LRSG-based loop.py for the new cluster brain architecture.

The simulation matches robot interface exactly:
  - Same encoder (SimEncoder → CAMERA_FEATURE_DIM float16 vector)
  - Same SensorPacket format (packet.py)
  - Same ActionPacket format (packet.py)
  - 30–60 Hz tick rate (default 30 Hz)

sim-to-real transfer is guaranteed because the cluster brain sees the same
packet format in simulation that AuroreMkVII sends from the real robot.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from src.cluster.packet import SensorPacket, ActionPacket, CAMERA_FEATURE_DIM, SERVO_CHANNELS, MOTOR_CHANNELS
from src.cluster.scheduler import ClusterScheduler
from src.simulation.navigation import NavigationEnvironment


# -------------------------------------------------------------------
# Frozen sensor encoder
# -------------------------------------------------------------------

class SimEncoder:
    """
    Fixed pretrained-style encoder: maps raw sim observations → CAMERA_FEATURE_DIM float16.

    Replaces the real Camera Module 3 libcamera pipeline in sim.
    Weights are random but frozen at creation (no gradient, no update).
    Must be identical between sim and hardware to ensure sim-to-real transfer.
    """

    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        # Input: 2 (position x,y) + 1 (goal dir) + 1 (goal dist) + 8 (lidar) = 12 dims
        self._input_dim = 12
        self._W = rng.standard_normal((self._input_dim, CAMERA_FEATURE_DIM)).astype(np.float32)
        self._W /= np.linalg.norm(self._W, axis=0, keepdims=True) + 1e-8

    def encode(
        self,
        agent_pos: tuple,
        goal_pos: tuple,
        lidar: np.ndarray,
        grid_size: float,
    ) -> np.ndarray:
        """Return float16 feature vector of shape (CAMERA_FEATURE_DIM,)."""
        ax, ay = agent_pos[0] / grid_size, agent_pos[1] / grid_size
        gx, gy = goal_pos[0] / grid_size, goal_pos[1] / grid_size
        dx, dy = gx - ax, gy - ay
        goal_dist = float(np.sqrt(dx**2 + dy**2))
        goal_dir = float(np.arctan2(dy, dx)) / np.pi  # normalized to [-1, 1]
        lidar_norm = np.clip(lidar / grid_size, 0.0, 1.0)

        raw = np.array([ax, ay, goal_dir, goal_dist] + lidar_norm[:8].tolist(), dtype=np.float32)
        # Pad if lidar has fewer than 8 rays
        if len(raw) < self._input_dim:
            raw = np.pad(raw, (0, self._input_dim - len(raw)))

        features = np.tanh(raw[:self._input_dim] @ self._W).astype(np.float16)
        return features


# -------------------------------------------------------------------
# Episode result
# -------------------------------------------------------------------

@dataclass
class EpisodeResult:
    episode: int
    steps: int
    total_reward: float
    reached_goal: bool
    final_distance: float


# -------------------------------------------------------------------
# Cluster simulation
# -------------------------------------------------------------------

class ClusterSimulation:
    """
    Single-agent navigation simulation using the cluster brain.

    The agent navigates a 10×10 grid to reach a goal while avoiding obstacles.
    Reward is shaped: +1 on reaching goal, step penalty, dense distance bonus.
    """

    GOAL_RADIUS = 0.5       # meters
    MAX_STEPS = 500         # per episode
    STEP_PENALTY = -0.005
    GOAL_REWARD = 1.0
    DIST_SCALE = 0.1        # scale for dense distance reward

    def __init__(
        self,
        scheduler: ClusterScheduler,
        grid_size: float = 10.0,
        n_obstacles: int = 3,
        tick_rate_hz: float = 30.0,
        seed: Optional[int] = None,
    ) -> None:
        self.scheduler = scheduler
        self.grid_size = grid_size
        self.tick_rate_hz = tick_rate_hz
        self._rng = np.random.default_rng(seed)

        # Encoder is frozen (same weights in sim and hardware)
        self._encoder = SimEncoder(seed=42)

        self._nav = NavigationEnvironment(
            grid_size=grid_size,
            num_obstacles=n_obstacles,
            seed=seed,
        )

        self._agent_pos = np.zeros(3, dtype=np.float64)
        self._goal_pos = np.zeros(3, dtype=np.float64)
        self._tick = 0
        self._episode = 0

    def _make_sensor_packet(self) -> SensorPacket:
        lidar = self._nav.get_lidar(tuple(self._agent_pos))
        cam_features = self._encoder.encode(
            self._agent_pos, self._goal_pos, lidar, self.grid_size
        )
        # Rangefinder: forward-facing ray (lidar index 0 = 0°)
        rangefinder = np.float32(lidar[0]) if len(lidar) > 0 else np.float32(self.grid_size)
        # Servo positions: zero (no servos in nav task, but packet format must be full)
        servo_pos = np.zeros(SERVO_CHANNELS, dtype=np.float16)
        return SensorPacket(
            camera_features=cam_features,
            rangefinder=rangefinder,
            servo_positions=servo_pos,
            tick=np.int64(self._tick),
        )

    def _apply_action(self, action: ActionPacket) -> None:
        """Decode motor_vector → 2D velocity and move agent."""
        mv = action.motor_vector.astype(np.float32)
        # Motor channels: 0=FL, 1=FR, 2=RL, 3=RR
        # Differential drive: vx = mean(all), vy = (right - left) / 2
        left = (float(mv[0]) + float(mv[2])) / 2.0
        right = (float(mv[1]) + float(mv[3])) / 2.0
        vx = (left + right) / 2.0
        vy = (right - left) / 2.0

        dt = 1.0 / self.tick_rate_hz
        speed = 1.5  # m/s max

        new_x = self._agent_pos[0] + vx * speed * dt
        new_y = self._agent_pos[1] + vy * speed * dt

        # Clamp to grid and check obstacle collision
        new_x = float(np.clip(new_x, 0.1, self.grid_size - 0.1))
        new_y = float(np.clip(new_y, 0.1, self.grid_size - 0.1))

        # Simple obstacle avoidance: revert axis that penetrates obstacle
        collided = False
        for ox, oy, orad in self._nav.obstacles:
            d = np.sqrt((new_x - ox) ** 2 + (new_y - oy) ** 2)
            if d < orad + 0.2:
                collided = True
                break

        if not collided:
            self._agent_pos[0] = new_x
            self._agent_pos[1] = new_y

    def _compute_reward(self, prev_dist: float, curr_dist: float, reached: bool) -> float:
        if reached:
            return self.GOAL_REWARD
        return self.STEP_PENALTY + self.DIST_SCALE * (prev_dist - curr_dist)

    def run_episode(self) -> EpisodeResult:
        """Run one complete episode. Returns metrics."""
        self._episode += 1
        start, goal = self._nav.reset()
        self._agent_pos = np.array(start, dtype=np.float64)
        self._goal_pos = np.array(goal, dtype=np.float64)

        total_reward = 0.0
        prev_dist = float(np.linalg.norm(self._agent_pos[:2] - self._goal_pos[:2]))

        for step in range(self.MAX_STEPS):
            self._tick += 1
            curr_dist = float(np.linalg.norm(self._agent_pos[:2] - self._goal_pos[:2]))
            reached = curr_dist < self.GOAL_RADIUS

            packet = self._make_sensor_packet()
            reward = self._compute_reward(prev_dist, curr_dist, reached)
            total_reward += reward

            action = self.scheduler.step(packet, reward=reward)
            self._apply_action(action)

            prev_dist = curr_dist

            if reached:
                return EpisodeResult(
                    episode=self._episode,
                    steps=step + 1,
                    total_reward=total_reward,
                    reached_goal=True,
                    final_distance=curr_dist,
                )

        final_dist = float(np.linalg.norm(self._agent_pos[:2] - self._goal_pos[:2]))
        return EpisodeResult(
            episode=self._episode,
            steps=self.MAX_STEPS,
            total_reward=total_reward,
            reached_goal=False,
            final_distance=final_dist,
        )

    def run(self, n_episodes: int) -> list[EpisodeResult]:
        """Run multiple episodes and return all results."""
        return [self.run_episode() for _ in range(n_episodes)]
