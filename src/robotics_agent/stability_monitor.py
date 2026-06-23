"""
Stability monitoring system.
Tracks divergence between predicted and observed outcomes.
Adjusts exploration and learning intensity based on error signals.
"""

import numpy as np
from collections import deque
from typing import Dict, Any
from enum import Enum


class StabilityLevel(Enum):
    """Agent stability states."""
    STABLE = "stable"           # Low prediction error, confident decisions
    UNCERTAIN = "uncertain"     # Moderate error, need exploration
    DIVERGED = "diverged"       # High error, major distribution shift
    RECOVERING = "recovering"   # Actively correcting after divergence


class StabilityMonitor:
    """
    Monitors agent stability by tracking:
    - Prediction error (actual vs predicted outcomes)
    - Task completion rate across environment variants
    - Adaptation speed to new environments
    """

    def __init__(self, window_size: int = 100):
        """Initialize stability monitor with rolling window."""
        self.window_size = window_size

        # Error tracking
        self.prediction_errors = deque(maxlen=window_size)
        self.reward_history = deque(maxlen=window_size)

        # Performance tracking
        self.task_completion_rates = {}  # per variant
        self.adaptation_speed = {}  # steps to adapt per variant

        # Current stability state
        self.stability_level = StabilityLevel.STABLE
        self.divergence_threshold = 1.0
        self.error_trend = deque(maxlen=10)  # For detecting error trajectory

        # Learning adjustment parameters
        self.exploration_intensity = 1.0  # Start at 1.0 (normal exploration)
        self.learning_rate_multiplier = 1.0

        # History
        self.stability_transitions = []

    def record_prediction_error(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """
        Record prediction error from one step.
        Returns: prediction error magnitude
        """
        error = np.linalg.norm(actual - predicted)
        self.prediction_errors.append(error)

        # Track error trend
        self.error_trend.append(error)

        # Update stability level based on error
        self._update_stability_level()

        return error

    def record_reward(self, reward: float) -> None:
        """Record reward from environment step."""
        self.reward_history.append(reward)

    def _update_stability_level(self) -> None:
        """Update stability state based on prediction error."""
        if len(self.prediction_errors) < 10:
            return

        mean_error = np.mean(list(self.prediction_errors)[-20:])
        error_trend = np.mean(self.error_trend)

        old_level = self.stability_level

        if mean_error < 0.1:
            new_level = StabilityLevel.STABLE
        elif mean_error < 0.5:
            new_level = StabilityLevel.UNCERTAIN
        elif error_trend > 0.5:  # Errors increasing
            new_level = StabilityLevel.DIVERGED
        else:
            new_level = StabilityLevel.RECOVERING

        if new_level != old_level:
            self.stability_level = new_level
            self.stability_transitions.append({
                "from": old_level.value,
                "to": new_level.value,
                "error": mean_error,
                "step": len(self.prediction_errors),
            })

            # Adjust learning parameters on state change
            self._adjust_learning_params()

    def _adjust_learning_params(self) -> None:
        """Adjust exploration and learning rates based on stability."""
        if self.stability_level == StabilityLevel.STABLE:
            self.exploration_intensity = 0.5  # Less exploration when stable
            self.learning_rate_multiplier = 1.0  # Normal learning
        elif self.stability_level == StabilityLevel.UNCERTAIN:
            self.exploration_intensity = 1.0  # Normal exploration
            self.learning_rate_multiplier = 1.0
        elif self.stability_level == StabilityLevel.DIVERGED:
            self.exploration_intensity = 2.0  # Increase exploration to recover
            self.learning_rate_multiplier = 2.0  # Faster adaptation
        else:  # RECOVERING
            self.exploration_intensity = 1.5
            self.learning_rate_multiplier = 1.5

    def record_task_completion(self, variant: str, success: bool, steps: int) -> None:
        """Record task completion for variant-specific metrics."""
        if variant not in self.task_completion_rates:
            self.task_completion_rates[variant] = {"successes": 0, "attempts": 0, "total_steps": 0}

        self.task_completion_rates[variant]["attempts"] += 1
        if success:
            self.task_completion_rates[variant]["successes"] += 1
        self.task_completion_rates[variant]["total_steps"] += steps

    def record_adaptation_time(self, variant: str, steps_to_adapt: int) -> None:
        """Record how many steps needed to adapt to new environment."""
        if variant not in self.adaptation_speed:
            self.adaptation_speed[variant] = []
        self.adaptation_speed[variant].append(steps_to_adapt)

    def get_stability_state(self) -> Dict[str, Any]:
        """Get current stability metrics."""
        mean_error = np.mean(self.prediction_errors) if self.prediction_errors else 0.0
        mean_reward = np.mean(self.reward_history) if self.reward_history else 0.0

        return {
            "level": self.stability_level.value,
            "mean_prediction_error": float(mean_error),
            "std_prediction_error": float(np.std(self.prediction_errors)) if self.prediction_errors else 0.0,
            "mean_reward": float(mean_reward),
            "exploration_intensity": float(self.exploration_intensity),
            "learning_rate_multiplier": float(self.learning_rate_multiplier),
            "error_trend": float(np.mean(self.error_trend)) if self.error_trend else 0.0,
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance metrics."""
        summary = {
            "variants_tested": list(self.task_completion_rates.keys()),
            "task_completion_by_variant": {},
            "adaptation_speed_by_variant": {},
        }

        for variant, metrics in self.task_completion_rates.items():
            if metrics["attempts"] > 0:
                completion_rate = metrics["successes"] / metrics["attempts"]
                avg_steps = metrics["total_steps"] / metrics["attempts"]
                summary["task_completion_by_variant"][variant] = {
                    "rate": float(completion_rate),
                    "avg_steps_per_task": float(avg_steps),
                }

        for variant, times in self.adaptation_speed.items():
            if times:
                summary["adaptation_speed_by_variant"][variant] = {
                    "mean_steps": float(np.mean(times)),
                    "std_steps": float(np.std(times)) if len(times) > 1 else 0.0,
                }

        return summary

    def should_increase_exploration(self) -> bool:
        """Check if agent should explore more."""
        return self.stability_level in [StabilityLevel.UNCERTAIN, StabilityLevel.DIVERGED]

    def should_reset_learning(self) -> bool:
        """Check if learning parameters should be reset (major distribution shift)."""
        return self.stability_level == StabilityLevel.DIVERGED and len(self.stability_transitions) > 3

    def get_mean_prediction_error(self) -> float:
        """Get mean prediction error for recent steps."""
        return float(np.mean(self.prediction_errors)) if self.prediction_errors else 0.0

    def get_error_percentile(self, percentile: float = 95.0) -> float:
        """Get error at given percentile."""
        if not self.prediction_errors:
            return 0.0
        return float(np.percentile(self.prediction_errors, percentile))

    def to_dict(self) -> Dict[str, Any]:
        """Export state for persistence."""
        return {
            "prediction_errors": list(self.prediction_errors),
            "reward_history": list(self.reward_history),
            "task_completion_rates": self.task_completion_rates,
            "adaptation_speed": self.adaptation_speed,
            "stability_level": self.stability_level.value,
            "stability_transitions": self.stability_transitions,
            "exploration_intensity": float(self.exploration_intensity),
            "learning_rate_multiplier": float(self.learning_rate_multiplier),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "StabilityMonitor":
        """Restore from persistence."""
        monitor = StabilityMonitor()
        monitor.prediction_errors = deque(data.get("prediction_errors", []), maxlen=100)
        monitor.reward_history = deque(data.get("reward_history", []), maxlen=100)
        monitor.task_completion_rates = data.get("task_completion_rates", {})
        monitor.adaptation_speed = data.get("adaptation_speed", {})
        monitor.stability_transitions = data.get("stability_transitions", [])
        monitor.exploration_intensity = data.get("exploration_intensity", 1.0)
        monitor.learning_rate_multiplier = data.get("learning_rate_multiplier", 1.0)

        if data.get("stability_level"):
            monitor.stability_level = StabilityLevel(data["stability_level"])

        return monitor
