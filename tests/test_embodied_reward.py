"""Unit tests for embodied reward shaping function."""
import pytest
import numpy as np
from src.lrsg.embodied_reward import compute_embodied_reward


class TestEmbodiedReward:
    """Test suite for goal-seeking reward shaping."""

    def test_reward_for_approaching_goal(self):
        """Test that approaching the goal yields positive reward."""
        agent_pos = (6.0, 6.0, 0.1)
        goal_pos = (8.0, 8.0, 0.1)
        prev_distance = np.sqrt(18)  # ~4.24 (was further away)
        reward = compute_embodied_reward(
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            prev_distance=prev_distance,
            is_collision=False,
            is_goal_reached=False
        )
        assert reward > 0.0, f"Approaching goal should give positive reward, got {reward}"

    def test_reward_for_colliding(self):
        """Test that collision yields negative reward."""
        agent_pos = (5.0, 5.0, 0.1)
        goal_pos = (8.0, 8.0, 0.1)
        prev_distance = 3.0
        reward = compute_embodied_reward(
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            prev_distance=prev_distance,
            is_collision=True,
            is_goal_reached=False
        )
        assert reward < 0.0, f"Collision should produce negative reward, got {reward}"

    def test_reward_for_goal_reached(self):
        """Test that reaching goal yields large positive reward."""
        agent_pos = (8.0, 8.0, 0.1)
        goal_pos = (8.0, 8.0, 0.1)
        prev_distance = 0.5
        reward = compute_embodied_reward(
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            prev_distance=prev_distance,
            is_collision=False,
            is_goal_reached=True
        )
        assert reward > 5.0, f"Goal reached should give large reward, got {reward}"

    def test_reward_for_moving_away(self):
        """Test that moving away from goal yields negative reward."""
        agent_pos = (2.0, 2.0, 0.1)
        goal_pos = (8.0, 8.0, 0.1)
        prev_distance = 3.0
        reward = compute_embodied_reward(
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            prev_distance=prev_distance,
            is_collision=False,
            is_goal_reached=False
        )
        assert reward < 0.0, f"Moving away should produce negative reward, got {reward}"
