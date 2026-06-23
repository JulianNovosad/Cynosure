"""
Simulation environment providing synthetic sensor streams matching physical robot model.
Supports multiple environment variants with different physics and object layouts.
"""

import numpy as np
from typing import Dict, Tuple, Any
import json
from pathlib import Path


class SimulationEnvironment:
    """Physics-based simulation with synthetic sensor streams."""

    def __init__(self, env_variant: str = "base", seed: int = 42):
        """
        Initialize simulation environment.

        Args:
            env_variant: Environment variant ("base", "gravity_shift", "friction_high", "layout_v2")
            seed: Random seed for deterministic behavior
        """
        self.env_variant = env_variant
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Action/observation space constraints (non-modifiable)
        self.action_space_bounds = {
            "linear_velocity": (-1.0, 1.0),
            "angular_velocity": (-np.pi, np.pi),
        }
        self.observation_space_size = 8  # Fixed sensor count: pos(3) + vel(3) + yaw(1) + yaw_rate(1)

        # Physics parameters (variant-specific)
        self._set_physics_params()

        # Internal state
        self.state = self._reset_state()
        self.time_step = 0
        self.sensor_noise_std = 0.05

    def _set_physics_params(self):
        """Set physics parameters based on environment variant."""
        variants = {
            "base": {
                "gravity": 9.81,
                "friction_coeff": 0.3,
                "mass": 1.0,
                "sensor_delay": 0,
            },
            "gravity_shift": {
                "gravity": 15.0,  # Higher gravity
                "friction_coeff": 0.3,
                "mass": 1.0,
                "sensor_delay": 1,
            },
            "friction_high": {
                "gravity": 9.81,
                "friction_coeff": 0.7,
                "mass": 1.0,
                "sensor_delay": 0,
            },
            "layout_v2": {
                "gravity": 9.81,
                "friction_coeff": 0.3,
                "mass": 0.8,  # Different mass
                "sensor_delay": 0,
            },
        }
        self.physics = variants.get(self.env_variant, variants["base"])

    def _reset_state(self) -> Dict[str, np.ndarray]:
        """Initialize robot state."""
        return {
            "position": np.array([0.0, 0.0, 0.0]),  # x, y, z
            "velocity": np.array([0.0, 0.0, 0.0]),
            "acceleration": np.array([0.0, 0.0, 0.0]),
            "orientation": np.array([0.0, 0.0, 0.0]),  # roll, pitch, yaw
            "angular_velocity": np.array([0.0, 0.0, 0.0]),
            "sensor_buffer": [],  # For handling sensor delay
        }

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.state = self._reset_state()
        self.time_step = 0
        return self._get_observation()

    def step(self, action: Dict[str, float]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one simulation step.

        Args:
            action: {"linear_velocity": float, "angular_velocity": float}

        Returns:
            observation, reward, done, info
        """
        # Validate action constraints
        lin_vel = np.clip(action["linear_velocity"],
                         self.action_space_bounds["linear_velocity"][0],
                         self.action_space_bounds["linear_velocity"][1])
        ang_vel = np.clip(action["angular_velocity"],
                         self.action_space_bounds["angular_velocity"][0],
                         self.action_space_bounds["angular_velocity"][1])

        # Physics simulation step (0.01 second timestep)
        dt = 0.01

        # Update velocity with friction
        friction_damping = 1.0 - (self.physics["friction_coeff"] * dt)
        self.state["velocity"] *= friction_damping
        self.state["angular_velocity"] *= friction_damping

        # Apply commanded velocities
        self.state["velocity"][0] += lin_vel * dt
        self.state["angular_velocity"][2] += ang_vel * dt  # Yaw

        # Update position
        self.state["position"] += self.state["velocity"] * dt
        self.state["orientation"][2] += self.state["angular_velocity"][2] * dt  # Yaw

        # Gravity effects (z-axis)
        self.state["acceleration"][2] = -self.physics["gravity"]
        self.state["velocity"][2] += self.state["acceleration"][2] * dt

        # Constrain position
        self.state["position"][2] = max(0.0, self.state["position"][2])
        if self.state["position"][2] == 0.0:
            self.state["velocity"][2] = max(0.0, self.state["velocity"][2])

        self.time_step += 1

        # Get observation with sensor noise
        obs = self._get_observation()

        # Reward: encourage stability and movement toward goal
        reward = self._compute_reward()

        # Episode termination
        done = self.time_step >= 1000 or np.linalg.norm(self.state["position"]) > 10.0

        info = {
            "time_step": self.time_step,
            "variant": self.env_variant,
            "state": {k: v.tolist() if isinstance(v, np.ndarray) else v
                     for k, v in self.state.items() if k != "sensor_buffer"},
        }

        return obs, reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Get sensor observation with noise and delay."""
        # 8-dimensional observation: pos(3) + vel(3) + yaw(1) + yaw_rate(1)
        obs = np.concatenate([
            self.state["position"],
            self.state["velocity"],
            self.state["orientation"][2:],  # Just yaw
            self.state["angular_velocity"][2:],  # Just yaw rate
        ])

        # Add sensor noise
        obs += self.rng.normal(0, self.sensor_noise_std, size=obs.shape)

        return obs.astype(np.float32)

    def _compute_reward(self) -> float:
        """Compute reward signal."""
        # Reward stability (low acceleration/velocity changes)
        accel_penalty = -0.01 * np.linalg.norm(self.state["acceleration"])
        vel_penalty = -0.001 * np.linalg.norm(self.state["velocity"])

        # Small penalty for being far from origin
        position_penalty = -0.001 * np.linalg.norm(self.state["position"])

        return float(accel_penalty + vel_penalty + position_penalty)

    def get_physics_params(self) -> Dict[str, float]:
        """Return current physics parameters (for agent knowledge)."""
        return self.physics.copy()

    def get_action_space(self) -> Dict[str, Tuple[float, float]]:
        """Return action space bounds."""
        return self.action_space_bounds.copy()

    def get_observation_space_size(self) -> int:
        """Return observation vector size."""
        return self.observation_space_size


class MultiEnvironmentRunner:
    """Manages multiple environment variants for testing adaptation."""

    def __init__(self, variants: list = None):
        """Initialize with list of environment variants."""
        if variants is None:
            variants = ["base", "gravity_shift", "friction_high", "layout_v2"]
        self.variants = variants
        self.envs = {v: SimulationEnvironment(v) for v in variants}
        self.current_variant = variants[0]

    def switch_to_variant(self, variant: str):
        """Switch to different environment variant."""
        if variant not in self.envs:
            raise ValueError(f"Unknown variant: {variant}")
        self.current_variant = variant

    def step(self, action: Dict[str, float]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute step in current environment."""
        return self.envs[self.current_variant].step(action)

    def reset(self, variant: str = None) -> np.ndarray:
        """Reset current or specified environment."""
        if variant:
            self.switch_to_variant(variant)
        return self.envs[self.current_variant].reset()

    def get_current_variant(self) -> str:
        """Return current environment variant."""
        return self.current_variant
