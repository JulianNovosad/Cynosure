"""Pet-level behavior patterns for navigation and interaction responses."""

import numpy as np


# Action constants
ACTION_MOVE_FWD = 0
ACTION_TURN_LEFT = 1
ACTION_TURN_RIGHT = 2
ACTION_IDLE = 3


class PetBehaviors:
    """Behavior patterns for pet-level navigation and interaction."""

    def __init__(self, stagnation_turn_prob=0.1):
        """
        Initialize PetBehaviors.

        Args:
            stagnation_turn_prob: Probability of turning during idle to prevent stagnation.
                                  Default 0.1 (10% chance to break stagnation).
        """
        self.stagnation_turn_prob = stagnation_turn_prob

    def get_idle_safety_action(self) -> int:
        """
        Get an action for idle/safety state.

        Returns mostly IDLE (action 3), occasionally turns (actions 1 or 2)
        to prevent stagnation.

        Returns:
            int: Action code (0-3)
        """
        # Probability of stagnation prevention (turning)
        if np.random.random() < self.stagnation_turn_prob:
            # Randomly choose between left and right turn
            return np.random.choice([ACTION_TURN_LEFT, ACTION_TURN_RIGHT])
        else:
            # Default to idle
            return ACTION_IDLE

    def get_approach_action(self, goal_direction: np.ndarray) -> int:
        """
        Determine action to approach a goal.

        Args:
            goal_direction: 2D direction vector pointing toward goal (normalized or relative)

        Returns:
            int: Action code (0-2, never IDLE when approaching)
        """
        # goal_direction is [x, y] where y > 0 means forward
        # x < 0 means left, x > 0 means right

        if goal_direction[1] > 0.3:
            # Goal is mostly forward
            return ACTION_MOVE_FWD
        elif goal_direction[0] < -0.3:
            # Goal is to the left
            return ACTION_TURN_LEFT
        else:
            # Goal is to the right or slightly behind on right
            return ACTION_TURN_RIGHT

    def get_flee_action(self, threat_direction: np.ndarray) -> int:
        """
        Determine action to flee from a threat.

        Uses opposite logic from approach: if threat is ahead,
        turn away; if threat is to the side, move away or turn.

        Args:
            threat_direction: 2D direction vector pointing toward threat

        Returns:
            int: Action code (0-3)
        """
        # Opposite of approach: if threat is forward, turn away
        # If threat is to left, turn right; if threat is to right, turn left

        if threat_direction[1] > 0.3:
            # Threat is mostly forward - turn away (either left or right)
            return np.random.choice([ACTION_TURN_LEFT, ACTION_TURN_RIGHT])
        elif threat_direction[0] < -0.3:
            # Threat is to the left - turn right (away from threat)
            return ACTION_TURN_RIGHT
        else:
            # Threat is to the right - turn left (away from threat)
            return ACTION_TURN_LEFT

    def get_interaction_response(self, signal_type: str) -> int:
        """
        Determine action in response to a social/interaction signal.

        Args:
            signal_type: Type of interaction signal ('approach_human', 'call_away', 'pet_touch')

        Returns:
            int: Action code (0-3)
        """
        if signal_type == "approach_human":
            return ACTION_MOVE_FWD
        elif signal_type == "call_away":
            return ACTION_TURN_RIGHT
        elif signal_type == "pet_touch":
            return ACTION_IDLE
        else:
            # Unknown signal defaults to idle
            return ACTION_IDLE
