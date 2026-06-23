"""
Learning agent for robotic control.
Integrates simulation, online learning, state persistence, dynamics modeling, and stability monitoring.
Transfers from simulation to physical deployment without architectural change.
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional
import json

from .sim_environment import SimulationEnvironment, MultiEnvironmentRunner
from .dynamics_model import MultiEnvironmentDynamicsModel
from .stability_monitor import StabilityMonitor
from .state_persistence import StatePersistence, ContinuousMemory


class RoboticsLearningAgent:
    """
    Agent that learns to control a robot in simulation.

    Key properties:
    - Online learning: updates policy and dynamics from interaction signals
    - Multi-environment adaptation: learns across variants with different physics
    - State continuity: maintains memory across sessions without reset
    - Stability monitoring: tracks prediction error and adjusts exploration
    - Transfer-ready: architecture works in simulation and physical deployment
    """

    def __init__(self, env_variants: list = None, storage_dir: str = ".agent_state"):
        """Initialize agent with multi-environment support."""
        if env_variants is None:
            env_variants = ["base", "gravity_shift", "friction_high", "layout_v2"]

        # Core components
        self.env = MultiEnvironmentRunner(variants=env_variants)
        self.env_variants = env_variants

        # Learned components
        obs_dim = self.env.envs[env_variants[0]].observation_space_size
        action_dim = 2  # [linear_velocity, angular_velocity]

        self.dynamics_model = MultiEnvironmentDynamicsModel(obs_dim, action_dim, env_variants)
        self.memory = ContinuousMemory()
        self.monitor = StabilityMonitor()

        # State persistence
        self.persistence = StatePersistence(storage_dir)

        # Policy parameters (simple linear policy for now)
        self.policy_weights = np.random.randn(action_dim, obs_dim) * 0.01
        self.policy_bias = np.zeros(action_dim)

        # Episode tracking
        self.episode = 0
        self.global_step = 0
        self.current_variant = env_variants[0]

        # Try to restore continuous state
        self._restore_or_initialize()

    def _restore_or_initialize(self):
        """Restore previous state or initialize fresh."""
        if not self.persistence.is_first_session():
            try:
                restored = self.persistence.restore_continuous_state()

                # Restore agent state
                if restored["agent_state"]:
                    self.policy_weights = restored["agent_state"].get("weights", self.policy_weights)
                    self.policy_bias = restored["agent_state"].get("bias", self.policy_bias)

                # Restore dynamics model
                if restored["dynamics_model"]:
                    self.dynamics_model.set_all_model_states(restored["dynamics_model"])

                # Restore memory
                if restored["memory"]:
                    self.memory = ContinuousMemory.from_dict(restored["memory"])

                # Restore monitor
                if restored["checkpoint"]:
                    checkpoint = restored["checkpoint"]
                    self.episode = checkpoint.get("episode", 0)
                    self.global_step = checkpoint.get("global_step", 0)

                print(f"[OK] Restored agent state from previous session")
                print(f"  Episodes: {self.episode}, Steps: {self.global_step}")

            except Exception as e:
                print(f"[WARN] Failed to restore state: {e}")
                print(f"  Starting fresh session")
                self.episode = 0
                self.global_step = 0

    def act(self, observation: np.ndarray, deterministic: bool = False) -> Dict[str, float]:
        """
        Select action from observation.

        Args:
            observation: Current state observation (from environment)
            deterministic: If True, use greedy policy; if False, add exploration noise

        Returns:
            {"linear_velocity": float, "angular_velocity": float}
        """
        # Linear policy: a = W @ s + b
        action_vec = self.policy_weights @ observation + self.policy_bias

        # Add exploration noise if not deterministic
        if not deterministic and self.monitor.should_increase_exploration():
            noise = self.monitor.exploration_intensity * np.random.randn(2) * 0.1
            action_vec += noise

        # Clip to action space bounds
        action_vec[0] = np.clip(action_vec[0], -1.0, 1.0)  # linear velocity
        action_vec[1] = np.clip(action_vec[1], -np.pi, np.pi)  # angular velocity

        return {
            "linear_velocity": float(action_vec[0]),
            "angular_velocity": float(action_vec[1]),
        }

    def learn_step(self, observation: np.ndarray, action: Dict[str, float],
                   next_observation: np.ndarray, reward: float, done: bool) -> Dict[str, float]:
        """
        One learning step from interaction.
        Updates both policy and dynamics model online.

        Returns:
            {"prediction_error": float, "policy_loss": float, "stability": str}
        """
        # 1. Update dynamics model
        pred_error = self.dynamics_model.update(
            observation, action, next_observation, self.current_variant
        )

        # 2. Record in memory
        self.memory.record_step(observation, action, reward)
        self.monitor.record_reward(reward)
        self.monitor.record_prediction_error(
            self.dynamics_model.predict(observation, action),
            next_observation
        )

        # 3. Update policy online (simple gradient-based improvement)
        # Loss: encourage actions that lead to high reward
        policy_loss = self._update_policy(observation, reward)

        # 4. Check for distribution shifts
        shift_detected, suggested_variant = self.dynamics_model.detect_distribution_shift()
        if shift_detected:
            print(f"  Distribution shift detected: {self.current_variant} -> {suggested_variant}")
            self.env.switch_to_variant(suggested_variant)
            self.current_variant = suggested_variant

        # 5. Record episode if done
        if done:
            self.memory.record_episode_summary({
                "episode": self.episode,
                "variant": self.current_variant,
                "steps": self.global_step % 1000,
                "final_reward": float(reward),
            })

        self.global_step += 1

        return {
            "prediction_error": float(pred_error),
            "policy_loss": float(policy_loss),
            "stability": self.monitor.stability_level.value,
        }

    def _update_policy(self, observation: np.ndarray, reward: float) -> float:
        """Update policy weights based on reward signal."""
        # Simple: move policy toward actions that achieved high reward
        lr = 0.001 * self.monitor.learning_rate_multiplier

        # Policy gradient: dL/dW ∝ reward * observation
        dW = lr * reward * np.outer(np.ones(2), observation)
        self.policy_weights += dW

        return float(np.linalg.norm(dW))

    def run_episode(self, variant: Optional[str] = None, max_steps: int = 1000,
                   deterministic: bool = False) -> Dict[str, Any]:
        """
        Run one complete episode in specified environment variant.

        Returns:
            Episode statistics
        """
        if variant:
            self.env.reset(variant)
            self.current_variant = variant
        else:
            self.env.reset()

        obs = self.env.envs[self.current_variant].reset()
        episode_reward = 0.0
        episode_steps = 0
        variant_start = self.current_variant

        for step in range(max_steps):
            # Agent decision
            action = self.act(obs, deterministic=deterministic)

            # Environment step
            next_obs, reward, done, info = self.env.step(action)

            # Learning
            learn_info = self.learn_step(obs, action, next_obs, reward, done)

            episode_reward += reward
            episode_steps += 1

            if done:
                break

            obs = next_obs

        # Record completion
        self.memory.record_episode_summary({
            "variant": variant_start,
            "steps": episode_steps,
            "reward": float(episode_reward),
            "episode": self.episode,
        })
        self.monitor.record_task_completion(variant_start, episode_steps > 100, episode_steps)

        self.episode += 1

        return {
            "episode": self.episode - 1,
            "variant": variant_start,
            "total_steps": episode_steps,
            "total_reward": float(episode_reward),
            "avg_reward": float(episode_reward / episode_steps) if episode_steps > 0 else 0.0,
            "stability": self.monitor.stability_level.value,
        }

    def test_multi_environment_adaptation(self, episodes_per_variant: int = 3) -> Dict[str, Any]:
        """
        Test adaptation across all environment variants.
        Critical for validating transfer-ready architecture.

        Returns:
            Adaptation metrics across variants
        """
        results = {}

        for variant in self.env_variants:
            variant_results = []
            for ep in range(episodes_per_variant):
                result = self.run_episode(variant=variant, max_steps=500)
                variant_results.append(result)

            results[variant] = {
                "mean_reward": float(np.mean([r["total_reward"] for r in variant_results])),
                "std_reward": float(np.std([r["total_reward"] for r in variant_results])),
                "mean_steps": float(np.mean([r["total_steps"] for r in variant_results])),
                "episodes": variant_results,
            }

        return results

    def save_state(self) -> None:
        """Persist agent state for continuity across sessions."""
        # Save agent state
        self.persistence.save_agent_state({
            "weights": self.policy_weights.copy(),
            "bias": self.policy_bias.copy(),
        })

        # Save dynamics model
        self.persistence.save_dynamics_model(
            self.dynamics_model.get_all_model_states()
        )

        # Save memory
        self.persistence.save_memory(self.memory.to_dict())

        # Save checkpoint
        self.persistence.save_checkpoint({
            "episode": self.episode,
            "global_step": self.global_step,
            "variant": self.current_variant,
            "monitor_state": self.monitor.to_dict(),
        })

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status."""
        return {
            "episode": self.episode,
            "global_step": self.global_step,
            "current_variant": self.current_variant,
            "stability": self.monitor.get_stability_state(),
            "performance": self.monitor.get_performance_summary(),
            "dynamics_accuracy": self.dynamics_model.models[self.current_variant].get_prediction_accuracy(),
            "variant_history": self.dynamics_model.get_variant_history(),
        }
