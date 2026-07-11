"""Unit tests for embodied reward shaping function."""
import pytest
import numpy as np
from src.lrsg.embodied_reward import compute_embodied_reward, compute_embodied_reward_enhanced


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


class TestEmbodiedRewardEnhanced:
    """Test suite for enhanced reward shaping with combat and survival."""

    def test_enhanced_reward_navigation(self):
        """Test that moving closer gives positive reward, moving away gives negative."""
        # Test moving closer: prev_distance=5.0, curr_distance=3.0 (reduction=2.0)
        reward_closer = compute_embodied_reward_enhanced(
            agent_pos=(3.0, 0.0),
            goal_pos=(0.0, 0.0),
            prev_distance=5.0,
            is_collision=False,
            is_goal_reached=False,
            combat_result=None
        )
        assert reward_closer > 0.0, f"Moving closer should give positive reward, got {reward_closer}"
        # Expected: 0.1 * 2.0 = 0.2
        assert np.isclose(reward_closer, 0.2), f"Expected 0.2, got {reward_closer}"

        # Test moving away: prev_distance=2.0, curr_distance=4.0 (reduction=-2.0)
        reward_away = compute_embodied_reward_enhanced(
            agent_pos=(4.0, 0.0),
            goal_pos=(0.0, 0.0),
            prev_distance=2.0,
            is_collision=False,
            is_goal_reached=False,
            combat_result=None
        )
        # Expected: 0.1 * -2.0 = -0.2 (symmetric: moving away is penalized, matching the docstring above)
        assert np.isclose(reward_away, -0.2), f"Expected -0.2, got {reward_away}"

    def test_enhanced_reward_collision(self):
        """Test that collision applies -5.0 penalty."""
        reward = compute_embodied_reward_enhanced(
            agent_pos=(5.0, 0.0),
            goal_pos=(10.0, 0.0),
            prev_distance=5.0,
            is_collision=True,
            is_goal_reached=False,
            combat_result=None
        )
        # Current distance is 5.0, prev_distance is 5.0, so distance_reward = 0
        # Collision penalty: -5.0
        assert reward == -5.0, f"Collision should apply -5.0 penalty, got {reward}"

    def test_enhanced_reward_goal(self):
        """Test that reaching goal gives +100.0 bonus."""
        reward = compute_embodied_reward_enhanced(
            agent_pos=(10.0, 0.0),
            goal_pos=(10.0, 0.0),
            prev_distance=1.0,
            is_collision=False,
            is_goal_reached=True,
            combat_result=None
        )
        # Goal reached: +100.0
        # Navigation: 0.1 * 1.0 = 0.1
        # Total: 100.0 + 0.1 = 100.1
        assert np.isclose(reward, 100.1), f"Goal reached should give +100.0 bonus, got {reward}"

    def test_enhanced_reward_combat_victory(self):
        """Test that combat victory reward propagates correctly."""
        combat_result = {
            "hit": True,
            "damage": 15.0,
            "victory": True,
            "reward": 50.0,
            "enemy_health_after": 0.0
        }
        reward = compute_embodied_reward_enhanced(
            agent_pos=(5.0, 0.0),
            goal_pos=(10.0, 0.0),
            prev_distance=6.0,
            is_collision=False,
            is_goal_reached=False,
            combat_result=combat_result
        )
        # Navigation: 0.1 * 1.0 = 0.1 (moved from 6.0 to 5.0 distance)
        # Combat: 50.0
        # Total: 0.1 + 50.0 = 50.1
        assert np.isclose(reward, 50.1), f"Combat victory should add +50.0, got {reward}"

    def test_enhanced_reward_combat_hit(self):
        """Test that combat hit reward propagates correctly."""
        combat_result = {
            "hit": True,
            "damage": 10.0,
            "victory": False,
            "reward": 5.0,
            "enemy_health_after": 20.0
        }
        reward = compute_embodied_reward_enhanced(
            agent_pos=(6.0, 0.0),
            goal_pos=(10.0, 0.0),
            prev_distance=4.0,
            is_collision=False,
            is_goal_reached=False,
            combat_result=combat_result
        )
        # Navigation: 0.1 * 0.0 = 0.0 (stayed at same distance: curr_distance = 4.0)
        # Combat: 5.0
        # Total: 0.0 + 5.0 = 5.0
        assert np.isclose(reward, 5.0), f"Combat hit should add +5.0, got {reward}"

    def test_enhanced_reward_combat_miss(self):
        """Test that combat miss applies -10.0 penalty through combat_result."""
        combat_result = {
            "hit": False,
            "damage": 0.0,
            "victory": False,
            "reward": -10.0,
            "enemy_health_after": 30.0
        }
        reward = compute_embodied_reward_enhanced(
            agent_pos=(5.0, 0.0),
            goal_pos=(10.0, 0.0),
            prev_distance=6.0,
            is_collision=False,
            is_goal_reached=False,
            combat_result=combat_result
        )
        # Navigation: 0.1 * 1.0 = 0.1 (moved closer)
        # Combat: -10.0
        # Total: 0.1 - 10.0 = -9.9
        assert np.isclose(reward, -9.9), f"Combat miss should add -10.0, got {reward}"

    def test_enhanced_reward_combined(self):
        """Test that multiple rewards stack correctly."""
        # Scenario: agent moves closer to goal AND hits enemy
        combat_result = {
            "hit": True,
            "damage": 8.0,
            "victory": False,
            "reward": 5.0,
            "enemy_health_after": 22.0
        }
        reward = compute_embodied_reward_enhanced(
            agent_pos=(3.0, 0.0),
            goal_pos=(10.0, 0.0),
            prev_distance=7.0,
            is_collision=False,
            is_goal_reached=False,
            combat_result=combat_result
        )
        # Current distance: 7.0
        # Navigation: 0.1 * (7.0 - 7.0) = 0.0 (no change in distance)
        # Combat: 5.0
        # Total: 0.0 + 5.0 = 5.0
        assert np.isclose(reward, 5.0), f"Combined rewards should sum correctly, got {reward}"

    def test_enhanced_reward_combined_with_collision(self):
        """Test that collision penalty applies even with other rewards."""
        combat_result = {
            "hit": True,
            "damage": 5.0,
            "victory": False,
            "reward": 5.0,
            "enemy_health_after": 25.0
        }
        reward = compute_embodied_reward_enhanced(
            agent_pos=(4.0, 0.0),
            goal_pos=(10.0, 0.0),
            prev_distance=5.0,
            is_collision=True,
            is_goal_reached=False,
            combat_result=combat_result
        )
        # Navigation: 0.1 * (5.0 - 6.0) = -0.1 (distance increased from 5.0 to 6.0)
        # Collision: -5.0
        # Combat: 5.0
        # Total: -0.1 - 5.0 + 5.0 = -0.1
        assert np.isclose(reward, -0.1), f"Collision + combat should combine, got {reward}"

    def test_enhanced_reward_no_combat(self):
        """Test that function handles None combat_result gracefully."""
        reward = compute_embodied_reward_enhanced(
            agent_pos=(5.0, 0.0),
            goal_pos=(10.0, 0.0),
            prev_distance=6.0,
            is_collision=False,
            is_goal_reached=False,
            combat_result=None
        )
        # Navigation: 0.1 * 1.0 = 0.1
        # Total: 0.1
        assert np.isclose(reward, 0.1), f"Without combat, should get only navigation reward, got {reward}"

    def test_enhanced_reward_goal_with_combat(self):
        """Test goal bonus combines with combat reward."""
        combat_result = {
            "hit": True,
            "damage": 20.0,
            "victory": True,
            "reward": 50.0,
            "enemy_health_after": 0.0
        }
        reward = compute_embodied_reward_enhanced(
            agent_pos=(10.0, 0.0),
            goal_pos=(10.0, 0.0),
            prev_distance=1.0,
            is_collision=False,
            is_goal_reached=True,
            combat_result=combat_result
        )
        # Navigation: 0.1 * 1.0 = 0.1
        # Goal: 100.0
        # Combat: 50.0
        # Total: 0.1 + 100.0 + 50.0 = 150.1
        assert np.isclose(reward, 150.1), f"Goal + combat should combine, got {reward}"
