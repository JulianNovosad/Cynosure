"""
Test suite for EnemyNPC class.

Validates behavior state machine (random walk, chase, flee), damage system,
and position dynamics.
"""

import pytest
import numpy as np
from src.simulation.enemy_npc import EnemyNPC


class TestEnemyCreation:
    """Test enemy creation and initialization."""

    def test_enemy_creation_basic(self):
        """Test basic enemy creation with default parameters."""
        pos = (5.0, 5.0)
        health = 50.0
        strength = 10.0

        enemy = EnemyNPC(position=pos, health=health, strength=strength)

        assert enemy.health == health
        assert enemy.strength == strength
        assert enemy.alive is True
        assert enemy.behavior == "random_walk"
        assert np.allclose(enemy.position, np.array(pos))

    def test_enemy_creation_with_custom_speed(self):
        """Test enemy creation with custom max_speed."""
        enemy = EnemyNPC(
            position=(0.0, 0.0),
            health=100.0,
            strength=5.0,
            max_speed=1.5
        )

        assert enemy.max_speed == 1.5

    def test_enemy_properties_stored_correctly(self):
        """Test that all properties are stored as correct types."""
        enemy = EnemyNPC(
            position=(2.5, 3.5),
            health=75.0,
            strength=8.0,
            max_speed=0.8
        )

        assert isinstance(enemy.health, float)
        assert isinstance(enemy.strength, float)
        assert isinstance(enemy.max_speed, float)
        assert isinstance(enemy.position, np.ndarray)


class TestEnemyDamage:
    """Test damage system and death mechanics."""

    def test_take_damage_reduces_health(self):
        """Test that take_damage reduces health correctly."""
        enemy = EnemyNPC(position=(0.0, 0.0), health=100.0, strength=5.0)

        returned = enemy.take_damage(25.0)

        assert enemy.health == 75.0
        assert returned is False
        assert enemy.alive is True

    def test_take_damage_multiple_times(self):
        """Test multiple damage applications."""
        enemy = EnemyNPC(position=(0.0, 0.0), health=100.0, strength=5.0)

        enemy.take_damage(30.0)
        assert enemy.health == 70.0

        enemy.take_damage(20.0)
        assert enemy.health == 50.0

        enemy.take_damage(15.0)
        assert enemy.health == 35.0

    def test_take_damage_at_death_threshold(self):
        """Test that enemy dies when health reaches 0."""
        enemy = EnemyNPC(position=(0.0, 0.0), health=50.0, strength=5.0)

        returned = enemy.take_damage(50.0)

        assert enemy.health == 0.0
        assert returned is True
        assert enemy.alive is False

    def test_take_damage_exceeding_health(self):
        """Test that damage exceeding health doesn't go negative."""
        enemy = EnemyNPC(position=(0.0, 0.0), health=30.0, strength=5.0)

        returned = enemy.take_damage(100.0)

        assert enemy.health == 0.0
        assert returned is True
        assert enemy.alive is False

    def test_take_damage_on_dead_enemy(self):
        """Test that dead enemies stay dead."""
        enemy = EnemyNPC(position=(0.0, 0.0), health=10.0, strength=5.0)

        enemy.take_damage(10.0)
        assert enemy.alive is False

        # Try to damage dead enemy
        returned = enemy.take_damage(5.0)
        assert enemy.health == 0.0
        assert returned is False  # Already dead


class TestEnemyRandomWalk:
    """Test random walk behavior."""

    def test_random_walk_changes_position(self):
        """Test that random walk produces position changes."""
        enemy = EnemyNPC(position=(5.0, 5.0), health=50.0, strength=10.0, max_speed=0.5)
        agent_pos = (10.0, 10.0)  # Far away to trigger random walk

        initial_pos = enemy.position.copy()

        # Run multiple updates to accumulate movement
        for _ in range(10):
            enemy.update(agent_pos, dt=0.1)

        final_pos = enemy.position
        distance_moved = np.linalg.norm(final_pos - initial_pos)

        # Should have moved some distance (not exactly at start)
        assert distance_moved > 0.01

    def test_random_walk_respects_max_speed(self):
        """Test that random walk doesn't exceed max_speed per step."""
        enemy = EnemyNPC(position=(0.0, 0.0), health=50.0, strength=10.0, max_speed=1.0)
        agent_pos = (50.0, 50.0)  # Very far away

        initial_pos = enemy.position.copy()

        for _ in range(5):
            enemy.update(agent_pos, dt=0.1)
            current_pos = enemy.position
            step_distance = np.linalg.norm(current_pos - initial_pos)

            # Maximum possible distance per update: max_speed * dt
            # We allow small tolerance for float precision
            max_allowed = enemy.max_speed * 0.1 * 5  # Multiple steps
            assert step_distance <= max_allowed + 0.01

    def test_random_walk_behavior_selected(self):
        """Test that random walk behavior is selected when agent is far."""
        enemy = EnemyNPC(position=(5.0, 5.0), health=50.0, strength=10.0)
        agent_pos = (100.0, 100.0)  # Very far

        enemy.update(agent_pos, dt=0.1)

        assert enemy.behavior == "random_walk"


class TestEnemyChase:
    """Test chase behavior."""

    def test_chase_moves_toward_agent(self):
        """Test that chase behavior moves toward agent."""
        enemy = EnemyNPC(position=(0.0, 0.0), health=100.0, strength=10.0, max_speed=0.5)
        agent_pos = (1.5, 0.0)  # Within 2.0 units to trigger chase

        # Run update multiple times
        for _ in range(20):
            enemy.update(agent_pos, dt=0.1)

        # Enemy should be closer to agent
        final_distance = np.linalg.norm(np.array(agent_pos) - enemy.position)
        assert final_distance < 1.5

    def test_chase_behavior_selected_when_close(self):
        """Test that chase behavior is selected when agent is close and healthy."""
        enemy = EnemyNPC(position=(5.0, 5.0), health=100.0, strength=10.0)
        agent_pos = (6.0, 5.0)  # 1 unit away, close enough to chase

        enemy.update(agent_pos, dt=0.1)

        assert enemy.behavior == "chase"

    def test_chase_moves_in_agent_direction(self):
        """Test that chase moves in the direction of the agent."""
        enemy = EnemyNPC(position=(0.0, 0.0), health=100.0, strength=10.0, max_speed=1.0)
        agent_pos = (1.5, 0.0)  # Within 2.0 units to trigger chase

        # Run update to trigger chase mode
        enemy.update(agent_pos, dt=0.1)

        # Verify we're in chase mode
        assert enemy.behavior == "chase"

        initial_pos = enemy.position.copy()
        enemy.update(agent_pos, dt=0.1)
        final_pos = enemy.position

        # Movement vector
        movement = final_pos - initial_pos

        # Should move in positive x direction
        assert movement[0] > 0.01


class TestEnemyFlee:
    """Test flee behavior."""

    def test_flee_moves_away_from_agent(self):
        """Test that flee behavior moves away from agent."""
        enemy = EnemyNPC(position=(5.0, 5.0), health=10.0, strength=10.0, max_speed=0.5)

        # Damage enemy to get health below 25% of max
        enemy.take_damage(8.0)  # Health now 2.0

        agent_pos = (5.5, 5.0)  # Very close, and low health -> flee

        initial_distance = np.linalg.norm(np.array(agent_pos) - enemy.position)

        # Run update
        enemy.update(agent_pos, dt=0.1)

        final_distance = np.linalg.norm(np.array(agent_pos) - enemy.position)

        # Distance should increase (or stay same due to discrete stepping)
        assert final_distance >= initial_distance - 0.01

    def test_flee_behavior_selected_when_low_health(self):
        """Test that flee is selected when agent is close and health is low."""
        enemy = EnemyNPC(position=(5.0, 5.0), health=10.0, strength=10.0)

        # Damage enemy to get health below 25% of max
        enemy.take_damage(8.0)  # Health now 2.0, which is 20% of 10

        agent_pos = (6.0, 5.0)  # 1 unit away, close

        enemy.update(agent_pos, dt=0.1)

        # Low health (2 out of 10) with 1 unit distance should trigger flee
        assert enemy.behavior == "flee"

    def test_flee_moves_away_from_agent_direction(self):
        """Test that flee moves away from agent."""
        enemy = EnemyNPC(position=(2.0, 0.0), health=100.0, strength=10.0, max_speed=1.0)

        # Damage enemy to get health below 25% of max
        enemy.take_damage(80.0)  # Health now 20.0, which is 20% of 100

        agent_pos = (0.0, 0.0)  # To the left, within 2.0 units

        # Run multiple updates to ensure consistent flee behavior
        for _ in range(3):
            enemy.update(agent_pos, dt=0.1)

        # After multiple updates in flee mode, distance should increase
        final_distance = np.linalg.norm(np.array(agent_pos) - enemy.position)
        initial_distance = 2.0

        # Distance should be greater than initial
        assert final_distance > initial_distance


class TestEnemyGetState:
    """Test state dictionary output."""

    def test_get_state_returns_dict(self):
        """Test that get_state returns a dictionary."""
        enemy = EnemyNPC(position=(5.0, 5.0), health=50.0, strength=10.0)
        state = enemy.get_state()

        assert isinstance(state, dict)

    def test_get_state_has_required_keys(self):
        """Test that get_state has all required keys."""
        enemy = EnemyNPC(position=(5.0, 5.0), health=50.0, strength=10.0)
        state = enemy.get_state()

        required_keys = {"position", "health", "strength", "behavior", "alive"}
        assert set(state.keys()) == required_keys

    def test_get_state_values_correct(self):
        """Test that get_state returns correct values."""
        pos = (3.5, 4.5)
        health = 75.0
        strength = 12.0

        enemy = EnemyNPC(position=pos, health=health, strength=strength)
        state = enemy.get_state()

        assert state["position"] == pos
        assert state["health"] == health
        assert state["strength"] == strength
        assert state["behavior"] == "random_walk"
        assert state["alive"] is True

    def test_get_state_after_damage(self):
        """Test that get_state reflects damage taken."""
        enemy = EnemyNPC(position=(0.0, 0.0), health=100.0, strength=10.0)

        enemy.take_damage(30.0)
        state = enemy.get_state()

        assert state["health"] == 70.0
        assert state["alive"] is True

    def test_get_state_after_death(self):
        """Test that get_state reflects death."""
        enemy = EnemyNPC(position=(0.0, 0.0), health=50.0, strength=10.0)

        enemy.take_damage(50.0)
        state = enemy.get_state()

        assert state["health"] == 0.0
        assert state["alive"] is False


class TestEnemyIntegration:
    """Integration tests combining multiple features."""

    def test_enemy_full_lifecycle(self):
        """Test enemy creation, movement, damage, and death."""
        enemy = EnemyNPC(position=(5.0, 5.0), health=100.0, strength=10.0, max_speed=0.5)
        agent_pos = (5.5, 5.0)

        # Initial state
        assert enemy.alive is True
        assert enemy.health == 100.0

        # Move toward agent
        for _ in range(5):
            enemy.update(agent_pos, dt=0.1)

        assert enemy.alive is True

        # Take damage
        enemy.take_damage(50.0)
        assert enemy.health == 50.0
        assert enemy.alive is True

        # More damage to death
        enemy.take_damage(50.0)
        assert enemy.health == 0.0
        assert enemy.alive is False

        # Update after death should not move
        final_pos = enemy.position.copy()
        enemy.update(agent_pos, dt=0.1)
        assert np.allclose(enemy.position, final_pos)

    def test_behavior_transitions(self):
        """Test that behavior transitions based on distance and health."""
        enemy = EnemyNPC(position=(5.0, 5.0), health=100.0, strength=10.0)

        # Start far: should be random walk
        agent_pos = (50.0, 50.0)
        enemy.update(agent_pos, dt=0.1)
        assert enemy.behavior == "random_walk"

        # Move close with high health: should chase
        agent_pos = (5.5, 5.0)
        enemy.update(agent_pos, dt=0.1)
        assert enemy.behavior == "chase"

        # Reduce health and stay close: should flee
        enemy.take_damage(80.0)  # Now at 20 health
        enemy.update(agent_pos, dt=0.1)
        assert enemy.behavior == "flee"

        # Move far away: should go back to random walk
        agent_pos = (50.0, 50.0)
        enemy.update(agent_pos, dt=0.1)
        assert enemy.behavior == "random_walk"

    def test_multiple_enemies_independent(self):
        """Test that multiple enemies operate independently."""
        enemy1 = EnemyNPC(position=(0.0, 0.0), health=100.0, strength=5.0)
        enemy2 = EnemyNPC(position=(10.0, 10.0), health=100.0, strength=10.0)

        agent_pos = (0.0, 0.0)

        # Update both
        enemy1.update(agent_pos, dt=0.1)
        enemy2.update(agent_pos, dt=0.1)

        # Damage only enemy1
        enemy1.take_damage(50.0)

        assert enemy1.health == 50.0
        assert enemy2.health == 100.0

        assert enemy1.alive is True
        assert enemy2.alive is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
