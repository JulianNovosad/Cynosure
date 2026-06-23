"""
Learned environment dynamics model.
Enables agent to predict consequences of actions across time steps.
Trained online from interaction signals without offline retraining.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import deque


class DynamicsModel:
    """
    Linear dynamics model: s_{t+1} = W @ [s_t, a_t] + b
    Updates online from prediction errors during interaction.
    """

    def __init__(self, obs_dim: int, action_dim: int, learning_rate: float = 0.001):
        """
        Initialize linear dynamics model.

        Args:
            obs_dim: Observation vector dimension (12 for robot)
            action_dim: Action vector dimension (2: linear + angular velocity)
            learning_rate: Online learning rate for weight updates
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr = learning_rate

        # Model: s_{t+1} = W @ z + b where z = [s_t, a_t]
        input_dim = obs_dim + action_dim
        self.W = np.random.randn(obs_dim, input_dim) * 0.01
        self.b = np.zeros(obs_dim)

        # Online learning tracking
        self.prediction_errors = deque(maxlen=100)
        self.update_count = 0

        # Prediction error thresholds for stability
        self.error_threshold = 0.5
        self.high_error_count = 0

    def predict(self, state: np.ndarray, action: Dict[str, float]) -> np.ndarray:
        """
        Predict next state given current state and action.
        Returns: predicted next observation
        """
        # Convert action to vector [linear_vel, angular_vel]
        action_vec = np.array([
            action["linear_velocity"],
            action["angular_velocity"],
        ])

        # Concatenate state and action
        z = np.concatenate([state, action_vec])

        # Linear prediction
        next_state = self.W @ z + self.b
        return next_state

    def predict_trajectory(self, state: np.ndarray, action: Dict[str, float],
                          horizon: int = 10) -> np.ndarray:
        """
        Predict trajectory over multiple timesteps.
        Returns: (horizon, obs_dim) array of predicted states
        """
        trajectory = np.zeros((horizon, self.obs_dim))
        current_state = state.copy()

        for t in range(horizon):
            next_state = self.predict(current_state, action)
            trajectory[t] = next_state
            current_state = next_state

        return trajectory

    def update(self, state: np.ndarray, action: Dict[str, float],
               next_state: np.ndarray) -> float:
        """
        Online update from one interaction step.
        Updates weights using prediction error signal.

        Returns: prediction error
        """
        # Get prediction
        predicted_next = self.predict(state, action)

        # Compute error
        error = next_state - predicted_next
        prediction_error = np.linalg.norm(error)

        # Track error for stability monitoring
        self.prediction_errors.append(prediction_error)

        # Adjust learning based on error magnitude
        if prediction_error > self.error_threshold:
            self.high_error_count += 1
            # Reduce learning rate if error is consistently high (sign of distribution shift)
            adjusted_lr = self.lr / (1.0 + 0.01 * self.high_error_count)
        else:
            self.high_error_count = max(0, self.high_error_count - 1)
            adjusted_lr = self.lr

        # Gradient descent update
        action_vec = np.array([
            action["linear_velocity"],
            action["angular_velocity"],
        ])
        z = np.concatenate([state, action_vec])

        # dL/dW = -error @ z^T
        dW = -np.outer(error, z)
        self.W += adjusted_lr * dW
        self.b += adjusted_lr * error

        self.update_count += 1
        return prediction_error

    def get_model_state(self) -> Dict[str, np.ndarray]:
        """Export model weights for persistence."""
        return {
            "W": self.W.copy(),
            "b": self.b.copy(),
            "update_count": self.update_count,
        }

    def set_model_state(self, state: Dict[str, np.ndarray]) -> None:
        """Restore model weights from persistence."""
        self.W = state["W"].copy()
        self.b = state["b"].copy()
        self.update_count = state["update_count"]

    def get_prediction_accuracy(self) -> float:
        """Return recent prediction accuracy (inverse of mean error)."""
        if not self.prediction_errors:
            return 1.0
        mean_error = np.mean(self.prediction_errors)
        return 1.0 / (1.0 + mean_error)  # Bounded in [0, 1]

    def is_high_error_state(self) -> bool:
        """Check if model is in high-error state (distribution shift)."""
        return self.high_error_count > 10

    def reset_error_tracking(self) -> None:
        """Reset error counters for new environment."""
        self.high_error_count = 0
        self.prediction_errors.clear()


class MultiEnvironmentDynamicsModel:
    """
    Manages dynamics models for multiple environment variants.
    Detects and adapts to distribution shifts between variants.
    """

    def __init__(self, obs_dim: int, action_dim: int, variants: List[str] = None):
        """Initialize with separate models for each variant."""
        if variants is None:
            variants = ["base", "gravity_shift", "friction_high", "layout_v2"]

        self.variants = variants
        self.models = {v: DynamicsModel(obs_dim, action_dim) for v in variants}
        self.current_variant = variants[0]
        self.variant_history = deque(maxlen=50)

    def predict(self, state: np.ndarray, action: Dict[str, float]) -> np.ndarray:
        """Predict using current variant model."""
        return self.models[self.current_variant].predict(state, action)

    def predict_trajectory(self, state: np.ndarray, action: Dict[str, float],
                          horizon: int = 10) -> np.ndarray:
        """Predict trajectory in current variant."""
        return self.models[self.current_variant].predict_trajectory(state, action, horizon)

    def update(self, state: np.ndarray, action: Dict[str, float],
               next_state: np.ndarray, actual_variant: str) -> float:
        """Update model for actual environment variant."""
        # Update current variant model
        error = self.models[actual_variant].update(state, action, next_state)

        # Record variant transition
        if not self.variant_history or self.variant_history[-1] != actual_variant:
            self.variant_history.append(actual_variant)

        # Switch model if variant changed
        if actual_variant != self.current_variant:
            self.current_variant = actual_variant

        return error

    def detect_distribution_shift(self) -> Tuple[bool, Optional[str]]:
        """
        Detect if we've shifted to a new environment variant.
        Returns: (shift_detected, suggested_variant)
        """
        current_model = self.models[self.current_variant]

        if current_model.is_high_error_state():
            # Check other variants
            best_variant = self.current_variant
            best_accuracy = current_model.get_prediction_accuracy()

            for variant, model in self.models.items():
                if variant != self.current_variant:
                    accuracy = model.get_prediction_accuracy()
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_variant = variant

            if best_variant != self.current_variant:
                return True, best_variant

        return False, None

    def get_all_model_states(self) -> Dict[str, Dict]:
        """Export all variant models for persistence."""
        return {v: self.models[v].get_model_state() for v in self.variants}

    def set_all_model_states(self, states: Dict[str, Dict]) -> None:
        """Restore all variant models from persistence."""
        for variant, state in states.items():
            if variant in self.models:
                self.models[variant].set_model_state(state)

    def get_variant_history(self) -> List[str]:
        """Return sequence of variants encountered."""
        return list(self.variant_history)
