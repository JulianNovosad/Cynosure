import numpy as np
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lrsg.pet_behaviors import PetBehaviors


def test_idle_safety_returns_valid_action():
    """Test that idle safety returns one of the valid actions (0-3)."""
    behaviors = PetBehaviors()
    action = behaviors.get_idle_safety_action()
    assert action in [0, 1, 2, 3], f"Expected action in [0, 1, 2, 3], got {action}"


def test_approach_toward_goal():
    """Test that approach toward goal returns a non-idle action."""
    behaviors = PetBehaviors()
    goal_dir = np.array([1.0, 0.0])
    action = behaviors.get_approach_action(goal_dir)
    assert action in [0, 1, 2], f"Expected action in [0, 1, 2] (not IDLE), got {action}"


def test_flee_away_from_threat():
    """Test that flee returns a valid action."""
    behaviors = PetBehaviors()
    threat_dir = np.array([1.0, 0.0])
    action = behaviors.get_flee_action(threat_dir)
    assert action in [0, 1, 2, 3], f"Expected action in [0, 1, 2, 3], got {action}"


def test_behavior_consistency():
    """Test that same input produces same output (deterministic)."""
    behaviors = PetBehaviors()
    goal_dir = np.array([0.707, 0.707])
    action1 = behaviors.get_approach_action(goal_dir)
    action2 = behaviors.get_approach_action(goal_dir)
    assert action1 == action2, f"Expected deterministic behavior, got {action1} and {action2}"
