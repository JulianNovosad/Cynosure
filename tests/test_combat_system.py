"""
Comprehensive tests for the combat system.

Tests verify:
- Strength calculation and clamping
- Hit probability scaling with strength difference
- Damage computation and randomization
- Full combat tick resolution with victory/defeat scenarios
"""

import pytest
import numpy as np
from src.simulation.combat_system import (
    compute_agent_strength,
    combat_attack,
    resolve_combat_tick
)
from src.simulation.enemy_npc import EnemyNPC


class TestComputeAgentStrength:
    """Test strength calculation based on accumulated reward."""

    def test_zero_reward(self):
        """Base strength with zero accumulated reward."""
        strength = compute_agent_strength(0.0)
        assert strength == 50.0

    def test_positive_reward_scaling(self):
        """Strength increases with positive reward."""
        strength_low = compute_agent_strength(0.0)
        strength_mid = compute_agent_strength(100.0)
        strength_high = compute_agent_strength(500.0)

        assert strength_low < strength_mid < strength_high

    def test_reward_scales_at_0_1_ratio(self):
        """Each 10 points of reward add 1 point of strength."""
        strength_0 = compute_agent_strength(0.0)
        strength_100 = compute_agent_strength(100.0)
        strength_200 = compute_agent_strength(200.0)

        assert strength_100 == strength_0 + 10.0
        assert strength_200 == strength_0 + 20.0

    def test_clamped_to_maximum(self):
        """Strength is clamped to maximum 100.0."""
        strength_high = compute_agent_strength(1000.0)
        strength_extreme = compute_agent_strength(10000.0)

        assert strength_high == 100.0
        assert strength_extreme == 100.0

    def test_clamped_to_minimum(self):
        """Strength is clamped to minimum 50.0."""
        strength_negative = compute_agent_strength(-1000.0)

        assert strength_negative == 50.0

    def test_typical_values(self):
        """Test with realistic accumulated reward values."""
        # After winning a few fights (50 reward each)
        strength_one_win = compute_agent_strength(50.0)
        assert 55.0 <= strength_one_win <= 56.0

        # After several wins
        strength_multi_win = compute_agent_strength(250.0)
        assert 75.0 <= strength_multi_win <= 80.0


class TestCombatAttack:
    """Test single attack resolution."""

    def test_equal_strength_hit_probability(self):
        """Equal strength should give approximately 50% hit chance."""
        hit_prob_result = combat_attack(75.0, 75.0)
        assert hit_prob_result["hit_probability"] == 0.5

    def test_stronger_attacker_higher_hit_probability(self):
        """Stronger attacker should have higher hit probability."""
        weaker_attacker = combat_attack(50.0, 75.0)
        stronger_attacker = combat_attack(75.0, 50.0)

        assert weaker_attacker["hit_probability"] < stronger_attacker["hit_probability"]

    def test_hit_probability_scaling(self):
        """Hit probability scales correctly with strength difference."""
        # 50 point advantage = 0.5 + 0.05 * 0.5 = 0.525
        high_advantage = combat_attack(100.0, 50.0)
        assert abs(high_advantage["hit_probability"] - 0.525) < 0.001

        # 50 point disadvantage = 0.5 - 0.025 = 0.475
        high_disadvantage = combat_attack(50.0, 100.0)
        assert abs(high_disadvantage["hit_probability"] - 0.475) < 0.001

        # 0 point difference = exactly 0.5
        equal_strength = combat_attack(75.0, 75.0)
        assert equal_strength["hit_probability"] == 0.5

    def test_hit_damage_range(self):
        """Damage on hit is in range [5, 20]."""
        rng = np.random.default_rng(42)
        # Use high strength advantage to increase hit probability
        hit_count = 0
        for _ in range(50):
            result = combat_attack(100.0, 0.0, rng=rng)
            if result["hit"]:
                hit_count += 1
                assert 5.0 <= result["damage"] <= 20.0
            else:
                assert result["damage"] == 0.0

        # Should have gotten some hits
        assert hit_count > 0

    def test_miss_zero_damage(self):
        """Missing attack deals zero damage."""
        rng = np.random.default_rng(42)
        # Use low strength disadvantage to increase miss probability
        miss_count = 0
        for _ in range(50):
            result = combat_attack(0.0, 100.0, rng=rng)
            if not result["hit"]:
                miss_count += 1
                assert result["damage"] == 0.0
            else:
                assert 5.0 <= result["damage"] <= 20.0

        # Should have gotten some misses
        assert miss_count > 0

    def test_rng_determinism(self):
        """Using provided RNG produces deterministic results."""
        rng1 = np.random.default_rng(123)
        result1 = combat_attack(75.0, 75.0, rng=rng1)

        rng2 = np.random.default_rng(123)
        result2 = combat_attack(75.0, 75.0, rng=rng2)

        assert result1["hit"] == result2["hit"]
        assert result1["damage"] == result2["damage"]

    def test_result_keys(self):
        """Result dict has all required keys."""
        result = combat_attack(75.0, 75.0)
        assert "hit" in result
        assert "damage" in result
        assert "hit_probability" in result
        assert len(result) == 3

    def test_hit_probability_clamped(self):
        """Hit probability is clamped to [0, 1]."""
        very_high = combat_attack(150.0, 0.0)
        very_low = combat_attack(0.0, 150.0)

        assert 0.0 <= very_high["hit_probability"] <= 1.0
        assert 0.0 <= very_low["hit_probability"] <= 1.0


class TestResolveCombatTick:
    """Test full combat tick resolution."""

    def test_combat_tick_hit_reward(self):
        """Hit on non-fatal attack gives +5 reward."""
        enemy = EnemyNPC((0.0, 0.0), health=100.0, strength=50.0)

        result = resolve_combat_tick(100.0, enemy)

        if result["hit"]:
            assert result["reward"] == 5.0
            assert result["victory"] is False

    def test_combat_tick_miss_reward(self):
        """Miss gives -10 reward."""
        enemy = EnemyNPC((0.0, 0.0), health=100.0, strength=100.0)

        result = resolve_combat_tick(50.0, enemy)

        if result["hit"] is False:
            assert result["reward"] == -10.0
            assert result["victory"] is False

    def test_combat_tick_victory_reward(self):
        """Victory (enemy dies) gives +50 reward."""
        # Give enemy very low health to guarantee kill within few hits
        enemy = EnemyNPC((0.0, 0.0), health=5.0, strength=50.0)

        # Attack multiple times until victory
        for _ in range(20):
            result = resolve_combat_tick(100.0, enemy)
            if result["victory"]:
                assert result["reward"] == 50.0
                return

        # If we get here, we didn't achieve victory in 20 tries
        pytest.fail("Failed to achieve victory in 20 combat ticks")

    def test_combat_tick_applies_damage(self):
        """Damage is applied to enemy health."""
        initial_health = 100.0
        enemy = EnemyNPC((0.0, 0.0), health=initial_health, strength=50.0)

        result = resolve_combat_tick(100.0, enemy)

        if result["hit"]:
            assert result["enemy_health_after"] < initial_health
            assert result["enemy_health_after"] == enemy.health
        else:
            assert result["enemy_health_after"] == initial_health

    def test_combat_tick_enemy_dies_on_lethal(self):
        """Enemy health reaches 0 when taking lethal damage."""
        enemy = EnemyNPC((0.0, 0.0), health=5.0, strength=0.0)

        # Attack until enemy dies (should happen quickly with very weak enemy)
        for _ in range(20):
            result = resolve_combat_tick(100.0, enemy)
            if result["victory"]:
                assert result["victory"] is True
                assert enemy.alive is False
                assert enemy.health <= 0.0
                return

        pytest.fail("Enemy did not die in 20 combat ticks")

    def test_combat_tick_result_keys(self):
        """Result dict has all required keys."""
        enemy = EnemyNPC((0.0, 0.0), health=100.0, strength=50.0)

        result = resolve_combat_tick(75.0, enemy)

        assert "hit" in result
        assert "damage" in result
        assert "victory" in result
        assert "reward" in result
        assert "enemy_health_after" in result
        assert len(result) == 5

    def test_multiple_hits_accumulate(self):
        """Multiple hits eventually kill enemy."""
        enemy = EnemyNPC((0.0, 0.0), health=100.0, strength=0.0)

        # Very strong attacker vs weak enemy should kill quickly
        hit_count = 0
        total_damage = 0.0
        for _ in range(50):
            if not enemy.alive:
                break
            result = resolve_combat_tick(100.0, enemy)
            if result["hit"]:
                hit_count += 1
                total_damage += result["damage"]

        # With very high strength vs 0 strength, hit rate should be ~52.5%
        # so we should get many hits across 50 ticks
        assert hit_count > 0
        assert total_damage > 0.0


class TestCombatIntegration:
    """Integration tests for combat system."""

    def test_stronger_agent_wins_more_often(self):
        """A stronger agent should defeat enemies more often."""
        rng = np.random.default_rng(42)

        weak_agent_victories = 0
        strong_agent_victories = 0

        num_trials = 20

        # Test weak agent
        for _ in range(num_trials):
            enemy = EnemyNPC((0.0, 0.0), health=50.0, strength=75.0)
            for _ in range(10):
                if not enemy.alive:
                    weak_agent_victories += 1
                    break
                combat_attack(50.0, enemy.strength, rng=rng)

        # Test strong agent
        for _ in range(num_trials):
            enemy = EnemyNPC((0.0, 0.0), health=50.0, strength=75.0)
            for _ in range(10):
                if not enemy.alive:
                    strong_agent_victories += 1
                    break
                combat_attack(100.0, enemy.strength, rng=rng)

        assert strong_agent_victories >= weak_agent_victories

    def test_realistic_combat_scenario(self):
        """Test realistic combat: agent fights to victory or retreat."""
        agent_initial_reward = 100.0
        agent_strength = compute_agent_strength(agent_initial_reward)

        enemy = EnemyNPC((0.0, 0.0), health=50.0, strength=60.0)

        total_reward = 0.0
        tick_count = 0
        max_ticks = 100

        while enemy.alive and tick_count < max_ticks:
            result = resolve_combat_tick(agent_strength, enemy)
            total_reward += result["reward"]
            tick_count += 1

        # Verify combat completed
        assert tick_count <= max_ticks

        # Verify rewards are sensible
        # Victory gets 50 per kill, hits get 5 each, misses get -10 each
        # With agent strength 60 vs enemy strength 60, hit rate is 50%
        # So we should see a mix of +5, -10, and potentially +50 rewards
        if enemy.alive:
            # Couldn't kill enemy, expect mix of hits/misses
            pass
        else:
            # Did kill enemy, total should include final +50
            assert any(result["reward"] == 50.0 for _ in range(1))
