"""Tests for 64D semantic state encoder."""
import pytest
import numpy as np
from src.simulation.semantic_encoder import SemanticEncoder
from src.simulation.enemy_npc import EnemyNPC


class TestSemanticEncoderCreation:
    """Test encoder instantiation."""

    def test_semantic_encoder_creation(self):
        """Test that SemanticEncoder can be instantiated."""
        encoder = SemanticEncoder()
        assert encoder is not None


class TestSemanticEncoderBasic:
    """Test basic encoding functionality."""

    def test_semantic_encode_basic(self):
        """Test basic encoding produces 64D vector."""
        encoder = SemanticEncoder()
        raw_obs = np.zeros(18, dtype=np.float32)
        enemies = []
        encoded = encoder.encode(raw_obs, enemies, accumulated_reward=0.0)

        assert isinstance(encoded, np.ndarray)
        assert encoded.shape == (64,)
        assert encoded.dtype == np.float32

    def test_semantic_encode_deterministic(self):
        """Test that same input produces same output (deterministic)."""
        encoder = SemanticEncoder()
        raw_obs = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0] + [1.0] * 8 + [5.0, 5.0, 7.0, 0.0], dtype=np.float32)
        enemies = []

        encoded1 = encoder.encode(raw_obs, enemies, accumulated_reward=100.0)
        encoded2 = encoder.encode(raw_obs, enemies, accumulated_reward=100.0)

        assert np.allclose(encoded1, encoded2, atol=1e-6)

    def test_semantic_encode_with_zero_observations(self):
        """Test encoding with all-zero observations."""
        encoder = SemanticEncoder()
        raw_obs = np.zeros(18, dtype=np.float32)
        enemies = []
        encoded = encoder.encode(raw_obs, enemies, accumulated_reward=0.0)

        # All dimensions should be finite
        assert np.all(np.isfinite(encoded))
        # Most dimensions should be in [0, 1]
        assert np.all(encoded >= -0.1) and np.all(encoded <= 1.1)


class TestSemanticEncoderWithEnemies:
    """Test encoding with enemies in view."""

    def test_semantic_encode_with_single_enemy(self):
        """Test encoding with one enemy."""
        encoder = SemanticEncoder()
        raw_obs = np.array([5.0, 5.0, 0.0,  # pos
                            0.0, 0.0, 0.0,  # vel
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # lidar (8 rays)
                            5.0, 5.0,  # goal delta
                            7.0,  # dist to goal
                            0.0], dtype=np.float32)  # safe zone

        enemies = [
            EnemyNPC(position=(3.0, 5.0), health=50.0, strength=40.0),
        ]

        encoded = encoder.encode(raw_obs, enemies, accumulated_reward=100.0)
        assert encoded.shape == (64,)

        # Threat should be non-zero if enemies are present
        threat_idx = 8  # threat_level
        assert encoded[threat_idx] > 0

        # Nearest enemy distance should be non-zero
        nearest_dist_idx = 7
        assert encoded[nearest_dist_idx] > 0

    def test_semantic_encode_with_multiple_enemies(self):
        """Test encoding with multiple enemies."""
        encoder = SemanticEncoder()
        raw_obs = np.array([5.0, 5.0, 0.0,  # pos
                            0.0, 0.0, 0.0,  # vel
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # lidar
                            5.0, 5.0,  # goal delta
                            7.0,  # dist to goal
                            0.0], dtype=np.float32)

        enemies = [
            EnemyNPC(position=(3.0, 5.0), health=50.0, strength=40.0),
            EnemyNPC(position=(7.0, 5.0), health=30.0, strength=20.0),
        ]

        encoded = encoder.encode(raw_obs, enemies, accumulated_reward=100.0)
        assert encoded.shape == (64,)

        # Enemy count should reflect multiple enemies
        enemy_count_idx = 9
        assert encoded[enemy_count_idx] > 0

    def test_semantic_encode_weak_vs_strong_enemies(self):
        """Test that weak and strong enemies are distinguished."""
        encoder = SemanticEncoder()
        raw_obs = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0] + [1.0] * 8 + [5.0, 5.0, 7.0, 0.0], dtype=np.float32)

        # Weak enemy nearby
        weak_enemies = [EnemyNPC(position=(5.5, 5.0), health=30.0, strength=30.0)]
        encoded_weak = encoder.encode(raw_obs, weak_enemies, accumulated_reward=100.0)

        # Strong enemy nearby
        strong_enemies = [EnemyNPC(position=(5.5, 5.0), health=80.0, strength=80.0)]
        encoded_strong = encoder.encode(raw_obs, strong_enemies, accumulated_reward=100.0)

        # Attack readiness should be higher for weak enemies
        attack_idx = 11
        assert encoded_weak[attack_idx] > encoded_strong[attack_idx]

        # Evasion priority should be higher for strong enemies
        evasion_idx = 12
        assert encoded_strong[evasion_idx] > encoded_weak[evasion_idx]


class TestSemanticEncoderRange:
    """Test that semantic dimensions stay in reasonable ranges."""

    def test_semantic_encode_range_bounded(self):
        """Test that all semantic dimensions are bounded."""
        encoder = SemanticEncoder()
        raw_obs = np.random.randn(18).astype(np.float32) * 5.0  # Larger variations
        enemies = [
            EnemyNPC(position=(np.random.uniform(-10, 10), np.random.uniform(-10, 10)),
                     health=np.random.uniform(20, 100),
                     strength=np.random.uniform(20, 80))
            for _ in range(3)
        ]

        encoded = encoder.encode(raw_obs, enemies, accumulated_reward=500.0)

        # All dimensions should be finite
        assert np.all(np.isfinite(encoded))
        # Most dimensions should be roughly in [-3.5, 3.5] (allowing some overflow for non-clipped dimensions)
        assert np.min(encoded) >= -4.0
        assert np.max(encoded) <= 4.0

    def test_semantic_encode_high_reward(self):
        """Test encoding with high accumulated reward."""
        encoder = SemanticEncoder()
        raw_obs = np.ones(18, dtype=np.float32) * 2.0
        enemies = []

        encoded = encoder.encode(raw_obs, enemies, accumulated_reward=1000.0)

        # Energy-related dimensions should be high
        energy_idx = 2
        assert encoded[energy_idx] > 0.5

        # Capability should be higher with high reward
        capability_idx = 35
        assert encoded[capability_idx] > 0.3

    def test_semantic_encode_low_reward(self):
        """Test encoding with low/negative accumulated reward."""
        encoder = SemanticEncoder()
        raw_obs = np.ones(18, dtype=np.float32) * 0.5
        enemies = []

        encoded = encoder.encode(raw_obs, enemies, accumulated_reward=-100.0)

        # Energy-related dimensions should be low
        energy_idx = 2
        assert encoded[energy_idx] <= 0.0


class TestSemanticEncoderDomains:
    """Test that individual semantic domains are computed correctly."""

    def test_navigation_domain(self):
        """Test navigation domain (indices 0-6)."""
        encoder = SemanticEncoder()
        # Close to goal
        raw_obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [5.0] * 8 + [2.0, 2.0, 3.0, 0.0], dtype=np.float32)
        encoded = encoder.encode(raw_obs, [], accumulated_reward=50.0)

        # Path confidence should be high (close to goal)
        assert encoded[0] > 0.8

        # Obstacle proximity should be moderate (lidar = 5.0)
        assert 0.4 < encoded[1] < 0.6

    def test_threat_assessment_domain(self):
        """Test threat assessment domain (indices 7-13)."""
        encoder = SemanticEncoder()
        raw_obs = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0] + [5.0] * 8 + [0.0, 0.0, 10.0, 0.0], dtype=np.float32)

        # No enemies
        encoded_no_enemies = encoder.encode(raw_obs, [], accumulated_reward=0.0)
        assert encoded_no_enemies[7] == 0.0  # nearest_enemy_distance
        assert encoded_no_enemies[8] == 0.0  # threat_level

        # With enemies
        enemies = [EnemyNPC(position=(6.0, 5.0), health=50.0, strength=60.0)]
        encoded_with_enemies = encoder.encode(raw_obs, enemies, accumulated_reward=0.0)
        assert encoded_with_enemies[7] > 0.0  # nearest_enemy_distance
        assert encoded_with_enemies[8] > 0.0  # threat_level

    def test_tactical_domain(self):
        """Test tactical domain (indices 14-20)."""
        encoder = SemanticEncoder()
        raw_obs = np.array([5.0, 5.0, 0.0, 0.5, 0.5, 0.0] + [5.0] * 8 + [5.0, 5.0, 10.0, 0.0], dtype=np.float32)
        encoded = encoder.encode(raw_obs, [], accumulated_reward=100.0)

        # Exploration drive should be moderate (dist to goal = 10.0)
        exploration_idx = 14
        assert 0.2 < encoded[exploration_idx] < 0.4

    def test_energy_survival_domain(self):
        """Test energy & survival domain (indices 21-27)."""
        encoder = SemanticEncoder()
        raw_obs = np.array([0.0] * 18, dtype=np.float32)

        encoded_low = encoder.encode(raw_obs, [], accumulated_reward=50.0)
        encoded_high = encoder.encode(raw_obs, [], accumulated_reward=500.0)

        energy_idx = 21
        fatigue_idx = 22

        # High reward → high energy, low fatigue
        assert encoded_high[energy_idx] > encoded_low[energy_idx]
        assert encoded_high[fatigue_idx] < encoded_low[fatigue_idx]

    def test_self_model_domain(self):
        """Test self model domain (indices 35-41)."""
        encoder = SemanticEncoder()
        raw_obs = np.array([0.0] * 18, dtype=np.float32)

        encoded = encoder.encode(raw_obs, [], accumulated_reward=300.0)

        # Capability confidence should scale with reward
        capability_idx = 35
        assert encoded[capability_idx] > 0.5

    def test_cognitive_domain(self):
        """Test cognitive domain (indices 56-62)."""
        encoder = SemanticEncoder()
        raw_obs = np.array([0.0] * 18, dtype=np.float32)

        encoded_low_reward = encoder.encode(raw_obs, [], accumulated_reward=0.0)
        encoded_high_reward = encoder.encode(raw_obs, [], accumulated_reward=500.0)

        comprehension_idx = 56

        # Higher reward → higher comprehension (model confidence)
        assert encoded_high_reward[comprehension_idx] > encoded_low_reward[comprehension_idx]


class TestSemanticEncoderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_encode_large_distances(self):
        """Test encoding with very large positions."""
        encoder = SemanticEncoder()
        raw_obs = np.array([100.0, 100.0, 0.0, 0.0, 0.0, 0.0] + [1.0] * 8 + [50.0, 50.0, 70.0, 0.0], dtype=np.float32)
        encoded = encoder.encode(raw_obs, [], accumulated_reward=0.0)

        # Should not produce NaN or Inf
        assert np.all(np.isfinite(encoded))

    def test_encode_zero_lidar(self):
        """Test encoding with zero lidar readings (collision imminent)."""
        encoder = SemanticEncoder()
        raw_obs = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 8 + [5.0, 5.0, 7.0, 0.0], dtype=np.float32)
        encoded = encoder.encode(raw_obs, [], accumulated_reward=0.0)

        # Escape feasibility should be 0
        assert encoded[10] == 0.0

        # Obstacle proximity should be 0
        assert encoded[1] == 0.0

    def test_encode_max_lidar(self):
        """Test encoding with max lidar readings (clear space)."""
        encoder = SemanticEncoder()
        raw_obs = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0] + [100.0] * 8 + [5.0, 5.0, 7.0, 0.0], dtype=np.float32)
        encoded = encoder.encode(raw_obs, [], accumulated_reward=0.0)

        # Escape feasibility should be 1.0
        assert encoded[10] == 1.0

        # Obstacle proximity should be clipped (100/10 = 10, clipped to 1.0)
        assert encoded[1] == 1.0

    def test_encode_dead_enemy(self):
        """Test encoding with dead and alive enemies.

        Note: The encoder receives enemy objects as-is; it's the caller's responsibility
        to filter out dead enemies. This test verifies the encoder still works even
        when passed dead enemies (doesn't break), though ideally callers filter them out.
        """
        encoder = SemanticEncoder()
        raw_obs = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0] + [5.0] * 8 + [5.0, 5.0, 7.0, 0.0], dtype=np.float32)

        # Create two enemies
        enemy1 = EnemyNPC(position=(3.0, 5.0), health=50.0, strength=40.0)
        enemy2 = EnemyNPC(position=(7.0, 5.0), health=50.0, strength=40.0)

        encoded_both_alive = encoder.encode(raw_obs, [enemy1, enemy2], accumulated_reward=0.0)

        # Kill first enemy
        enemy1.take_damage(100.0)

        # Encode with one alive, one dead (in real usage, caller would filter)
        encoded_mixed = encoder.encode(raw_obs, [enemy1, enemy2], accumulated_reward=0.0)

        # Threat level might be similar since dead enemy still has strength value
        # The important test is that encoder doesn't crash with dead enemies
        threat_idx = 8
        # Both should be non-zero (enemies are present, even if one is dead)
        assert encoded_both_alive[threat_idx] > 0.0
        assert encoded_mixed[threat_idx] > 0.0


class TestSemanticEncoderDimensionInteraction:
    """Test interactions between different semantic dimensions."""

    def test_threat_affects_tactical(self):
        """Test that high threat level increases retreat bias."""
        encoder = SemanticEncoder()
        raw_obs = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0] + [5.0] * 8 + [5.0, 5.0, 7.0, 0.0], dtype=np.float32)

        # Safe scenario
        encoded_safe = encoder.encode(raw_obs, [], accumulated_reward=100.0)

        # Threat scenario
        strong_enemy = EnemyNPC(position=(5.5, 5.0), health=100.0, strength=90.0)
        encoded_threat = encoder.encode(raw_obs, [strong_enemy], accumulated_reward=100.0)

        # Retreat bias should be higher in threat scenario
        retreat_idx = 16
        assert encoded_threat[retreat_idx] > encoded_safe[retreat_idx]

    def test_distance_affects_goal_commitment(self):
        """Test that close distance to goal increases goal commitment."""
        encoder = SemanticEncoder()

        # Far from goal
        raw_obs_far = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [5.0] * 8 + [10.0, 10.0, 14.0, 0.0], dtype=np.float32)
        encoded_far = encoder.encode(raw_obs_far, [], accumulated_reward=0.0)

        # Close to goal
        raw_obs_close = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [5.0] * 8 + [1.0, 1.0, 1.5, 0.0], dtype=np.float32)
        encoded_close = encoder.encode(raw_obs_close, [], accumulated_reward=0.0)

        # Goal commitment should be higher when close
        commitment_idx = 42
        assert encoded_close[commitment_idx] > encoded_far[commitment_idx]

    def test_reward_enables_learning(self):
        """Test that positive reward increases learning readiness."""
        encoder = SemanticEncoder()
        raw_obs = np.array([0.0] * 18, dtype=np.float32)

        # Positive reward delta
        encoded_learning = encoder.encode(raw_obs, [], accumulated_reward=100.0, prev_accumulated_reward=90.0)

        # No reward delta
        encoded_no_learning = encoder.encode(raw_obs, [], accumulated_reward=100.0, prev_accumulated_reward=100.0)

        # Learning readiness should be higher with positive reward
        learning_idx = 34
        assert encoded_learning[learning_idx] > encoded_no_learning[learning_idx]


class TestSemanticEncoderConsistency:
    """Test consistency across multiple calls."""

    def test_multiple_calls_same_output(self):
        """Test that encoder produces consistent output across multiple calls."""
        encoder = SemanticEncoder()
        raw_obs = np.array([5.0, 5.0, 0.0, 0.5, -0.3, 0.0] + [3.0, 4.0, 2.0, 5.0, 1.0, 6.0, 2.5, 4.5] + [3.0, 3.0, 5.0, 0.0], dtype=np.float32)
        enemies = [
            EnemyNPC(position=(6.0, 5.0), health=50.0, strength=40.0),
            EnemyNPC(position=(4.0, 4.0), health=30.0, strength=25.0),
        ]

        # Run multiple times
        encoded1 = encoder.encode(raw_obs, enemies, accumulated_reward=150.0, prev_accumulated_reward=140.0)
        encoded2 = encoder.encode(raw_obs, enemies, accumulated_reward=150.0, prev_accumulated_reward=140.0)
        encoded3 = encoder.encode(raw_obs, enemies, accumulated_reward=150.0, prev_accumulated_reward=140.0)

        # All should be identical
        assert np.allclose(encoded1, encoded2, atol=1e-6)
        assert np.allclose(encoded2, encoded3, atol=1e-6)

    def test_encoder_has_no_state(self):
        """Test that encoder is stateless (no accumulated state between calls)."""
        encoder = SemanticEncoder()
        raw_obs1 = np.array([0.0] * 18, dtype=np.float32)
        raw_obs2 = np.array([10.0] * 18, dtype=np.float32)

        # Encode in sequence
        encoded1_first = encoder.encode(raw_obs1, [], accumulated_reward=0.0)
        encoded2 = encoder.encode(raw_obs2, [], accumulated_reward=100.0)
        encoded1_second = encoder.encode(raw_obs1, [], accumulated_reward=0.0)

        # Same input should produce same output regardless of prior calls
        assert np.allclose(encoded1_first, encoded1_second, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
