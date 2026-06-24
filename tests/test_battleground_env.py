"""
Test suite for BattlegroundEnvironment class.

Validates procedural map generation, enemy spawning, enemy lifecycle management,
behavior updates, and collision detection with enemies.
"""

import pytest
import numpy as np
from src.simulation.battleground_env import BattlegroundEnvironment
from src.simulation.enemy_npc import EnemyNPC


class TestBattlegroundCreation:
    """Test battleground creation and initialization."""

    def test_battleground_creation_default(self):
        """Test battleground creation with default parameters."""
        bg = BattlegroundEnvironment()

        assert bg.grid_size == 20.0
        assert bg.num_obstacles == 5
        assert bg.num_enemies == 3
        assert len(bg.obstacles) == 5
        assert len(bg.enemies) == 0  # No enemies until reset

    def test_battleground_creation_custom(self):
        """Test battleground creation with custom parameters."""
        bg = BattlegroundEnvironment(grid_size=15, num_obstacles=3, num_enemies=5, seed=42)

        assert bg.grid_size == 15.0
        assert bg.num_obstacles == 3
        assert bg.num_enemies == 5
        assert len(bg.obstacles) == 3

    def test_battleground_inherits_from_navigation(self):
        """Test that BattlegroundEnvironment inherits from NavigationEnvironment."""
        from src.simulation.navigation import NavigationEnvironment
        bg = BattlegroundEnvironment()

        assert isinstance(bg, NavigationEnvironment)

    def test_battleground_seed_deterministic(self):
        """Test that same seed produces same obstacle configuration."""
        bg1 = BattlegroundEnvironment(grid_size=20, num_obstacles=5, seed=99)
        bg2 = BattlegroundEnvironment(grid_size=20, num_obstacles=5, seed=99)

        assert len(bg1.obstacles) == len(bg2.obstacles)
        assert np.allclose(bg1.obstacles, bg2.obstacles)


class TestBattlegroundReset:
    """Test battleground reset functionality."""

    def test_reset_returns_start_and_goal(self):
        """Test that reset returns start and goal positions."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=3, seed=42)
        start, goal = bg.reset()

        assert isinstance(start, tuple)
        assert isinstance(goal, tuple)
        assert len(start) == 2  # 2D positions
        assert len(goal) == 2

    def test_reset_start_and_goal_different(self):
        """Test that start and goal are at different locations."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=3, seed=42)
        start, goal = bg.reset()

        distance = np.sqrt((goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2)
        assert distance > 2.0, "Start and goal should be at least 2.0 units apart"

    def test_reset_spawns_enemies(self):
        """Test that reset spawns the correct number of enemies."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=3, seed=42)
        start, goal = bg.reset()

        assert len(bg.enemies) == 3
        assert all(isinstance(e, EnemyNPC) for e in bg.enemies)

    def test_reset_all_enemies_alive(self):
        """Test that all spawned enemies are alive."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=5, seed=42)
        start, goal = bg.reset()

        assert all(e.alive for e in bg.enemies)

    def test_reset_enemies_within_grid(self):
        """Test that enemies spawn within grid boundaries."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=3, seed=42)
        start, goal = bg.reset()

        for enemy in bg.enemies:
            assert 0.0 < enemy.position[0] < 20.0
            assert 0.0 < enemy.position[1] < 20.0

    def test_reset_enemies_have_random_stats(self):
        """Test that enemies have randomized stats."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=3, seed=42)
        start, goal = bg.reset()

        # Reset again with different seed
        bg2 = BattlegroundEnvironment(grid_size=20, num_enemies=3, seed=99)
        start2, goal2 = bg2.reset()

        # Stats should be different (very low probability of collision)
        health_values_1 = [e.health for e in bg.enemies]
        health_values_2 = [e.health for e in bg2.enemies]

        # At least one should be different
        assert not np.allclose(health_values_1, health_values_2)

    def test_reset_enemies_have_valid_stats(self):
        """Test that spawned enemies have valid attribute ranges."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=3, seed=42)
        start, goal = bg.reset()

        for enemy in bg.enemies:
            assert 20.0 <= enemy.health <= 100.0
            assert 20.0 <= enemy.strength <= 80.0
            assert 0.3 <= enemy.max_speed <= 1.0

    def test_reset_enemies_initial_behavior(self):
        """Test that enemies have initial behavior (random_walk or chase)."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=10, seed=42)
        start, goal = bg.reset()

        behaviors = [e.behavior for e in bg.enemies]
        assert all(b in ["random_walk", "chase"] for b in behaviors)

        # Should have mix of behaviors (high probability with 10 enemies)
        has_random = "random_walk" in behaviors
        has_chase = "chase" in behaviors
        assert has_random or has_chase  # At least one type


class TestBattlegroundGetEnemies:
    """Test get_enemies functionality."""

    def test_get_enemies_returns_alive_only(self):
        """Test that get_enemies returns only alive enemies."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=3, seed=42)
        start, goal = bg.reset()

        alive_enemies = bg.get_enemies()
        assert len(alive_enemies) == 3

        # Kill one enemy
        bg.enemies[0].take_damage(bg.enemies[0].health)
        alive_enemies = bg.get_enemies()
        assert len(alive_enemies) == 2
        assert bg.enemies[0] not in alive_enemies

    def test_get_enemies_empty_when_all_dead(self):
        """Test that get_enemies returns empty list when all are dead."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=2, seed=42)
        start, goal = bg.reset()

        # Kill all enemies
        for enemy in bg.enemies:
            enemy.take_damage(enemy.health)

        alive_enemies = bg.get_enemies()
        assert len(alive_enemies) == 0

    def test_get_enemies_returns_copy_behavior(self):
        """Test get_enemies returns list (not reference to internal)."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=3, seed=42)
        start, goal = bg.reset()

        alive1 = bg.get_enemies()
        alive2 = bg.get_enemies()

        # Should be equal but different lists
        assert len(alive1) == len(alive2)
        # Modifying returned list shouldn't affect enemies
        alive1.clear()
        assert len(bg.get_enemies()) == 3


class TestBattlegroundGetEnemyNear:
    """Test get_enemy_near functionality."""

    def test_get_enemy_near_finds_nearby(self):
        """Test that get_enemy_near finds enemies within radius."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=0, seed=42)
        bg.reset()

        # Manually create an enemy at known location
        enemy = EnemyNPC(position=(5.0, 5.0), health=50.0, strength=20.0)
        bg.enemies.append(enemy)

        # Query at same location
        nearby = bg.get_enemy_near((5.0, 5.0), radius=1.0)
        assert len(nearby) == 1
        assert nearby[0] == enemy

    def test_get_enemy_near_radius_boundary(self):
        """Test radius boundary condition."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=0, seed=42)
        bg.reset()

        # Create enemy at (5, 5)
        enemy = EnemyNPC(position=(5.0, 5.0), health=50.0, strength=20.0)
        bg.enemies.append(enemy)

        # Query at distance exactly 3.0, should be included
        query_pos = (5.0 + 3.0, 5.0)
        nearby = bg.get_enemy_near(query_pos, radius=3.0)
        assert len(nearby) == 1

        # Query at distance 3.1, should be excluded
        query_pos = (5.0 + 3.1, 5.0)
        nearby = bg.get_enemy_near(query_pos, radius=3.0)
        assert len(nearby) == 0

    def test_get_enemy_near_multiple_enemies(self):
        """Test finding multiple nearby enemies."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=0, seed=42)
        bg.reset()

        # Create multiple enemies
        enemy1 = EnemyNPC(position=(5.0, 5.0), health=50.0, strength=20.0)
        enemy2 = EnemyNPC(position=(6.0, 5.0), health=50.0, strength=20.0)
        enemy3 = EnemyNPC(position=(15.0, 15.0), health=50.0, strength=20.0)
        bg.enemies.extend([enemy1, enemy2, enemy3])

        # Query near first two
        nearby = bg.get_enemy_near((5.5, 5.0), radius=2.0)
        assert len(nearby) == 2
        assert enemy1 in nearby
        assert enemy2 in nearby
        assert enemy3 not in nearby

    def test_get_enemy_near_excludes_dead(self):
        """Test that get_enemy_near excludes dead enemies."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=0, seed=42)
        bg.reset()

        enemy1 = EnemyNPC(position=(5.0, 5.0), health=50.0, strength=20.0)
        enemy2 = EnemyNPC(position=(6.0, 5.0), health=50.0, strength=20.0)
        bg.enemies.extend([enemy1, enemy2])

        # Kill enemy1
        enemy1.take_damage(50.0)

        nearby = bg.get_enemy_near((5.5, 5.0), radius=5.0)
        assert len(nearby) == 1
        assert nearby[0] == enemy2

    def test_get_enemy_near_default_radius(self):
        """Test default radius of 5.0."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=0, seed=42)
        bg.reset()

        enemy = EnemyNPC(position=(10.0, 10.0), health=50.0, strength=20.0)
        bg.enemies.append(enemy)

        # Query at distance 4.5 with default radius (should find)
        nearby = bg.get_enemy_near((10.0 + 4.5, 10.0))
        assert len(nearby) == 1

        # Query at distance 5.5 with default radius (should not find)
        nearby = bg.get_enemy_near((10.0 + 5.5, 10.0))
        assert len(nearby) == 0


class TestBattlegroundRemoveEnemy:
    """Test remove_enemy functionality."""

    def test_remove_enemy_removes_from_list(self):
        """Test that remove_enemy removes enemy from list."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=3, seed=42)
        start, goal = bg.reset()

        initial_count = len(bg.enemies)
        enemy_to_remove = bg.enemies[0]

        bg.remove_enemy(enemy_to_remove)

        assert len(bg.enemies) == initial_count - 1
        assert enemy_to_remove not in bg.enemies

    def test_remove_enemy_not_in_list(self):
        """Test removing enemy not in list (should not raise)."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=3, seed=42)
        start, goal = bg.reset()

        # Create an enemy not in the list
        other_enemy = EnemyNPC(position=(0.0, 0.0), health=50.0, strength=20.0)

        initial_count = len(bg.enemies)
        bg.remove_enemy(other_enemy)

        # Count should not change
        assert len(bg.enemies) == initial_count

    def test_remove_multiple_enemies(self):
        """Test removing multiple enemies."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=5, seed=42)
        start, goal = bg.reset()

        enemies_to_remove = bg.enemies[:2]

        for enemy in enemies_to_remove:
            bg.remove_enemy(enemy)

        assert len(bg.enemies) == 3
        for enemy in enemies_to_remove:
            assert enemy not in bg.enemies


class TestBattlegroundTickEnemies:
    """Test tick_enemies behavior update functionality."""

    def test_tick_enemies_updates_behavior(self):
        """Test that tick_enemies updates enemy behaviors."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=0, seed=42)
        bg.reset()

        # Create enemy with strong health
        enemy = EnemyNPC(position=(5.0, 5.0), health=100.0, strength=60.0)
        bg.enemies.append(enemy)

        # Initial behavior
        assert enemy.behavior in ["random_walk", "chase"]

        # Tick at close range with high strength -> should chase
        agent_pos = (5.5, 5.0)  # 0.5 units away
        bg.tick_enemies(agent_pos)

        assert enemy.behavior == "chase"

    def test_tick_enemies_chase_behavior(self):
        """Test chase behavior selection: dist < 5.0 AND strength > 50."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=0, seed=42)
        bg.reset()

        enemy = EnemyNPC(position=(5.0, 5.0), health=100.0, strength=60.0)
        bg.enemies.append(enemy)

        # Within 5.0 and strength > 50 -> chase
        bg.tick_enemies((5.0 + 4.0, 5.0))  # 4.0 units away
        assert enemy.behavior == "chase"

    def test_tick_enemies_flee_behavior(self):
        """Test flee behavior selection: dist < 3.0."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=0, seed=42)
        bg.reset()

        enemy = EnemyNPC(position=(5.0, 5.0), health=100.0, strength=20.0)
        bg.enemies.append(enemy)

        # Within 3.0 -> flee (regardless of strength)
        bg.tick_enemies((5.0 + 2.0, 5.0))  # 2.0 units away
        assert enemy.behavior == "flee"

    def test_tick_enemies_random_walk_behavior(self):
        """Test random_walk behavior selection: dist >= 5.0."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=0, seed=42)
        bg.reset()

        enemy = EnemyNPC(position=(5.0, 5.0), health=100.0, strength=60.0)
        bg.enemies.append(enemy)

        # Far away -> random walk
        bg.tick_enemies((15.0, 15.0))  # ~14 units away
        assert enemy.behavior == "random_walk"

    def test_tick_enemies_updates_position(self):
        """Test that tick_enemies moves enemies."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=0, seed=42)
        bg.reset()

        enemy = EnemyNPC(position=(5.0, 5.0), health=100.0, strength=20.0, max_speed=1.0)
        bg.enemies.append(enemy)

        initial_pos = enemy.position.copy()
        agent_pos = (10.0, 10.0)

        bg.tick_enemies(agent_pos)

        # Enemy should have moved
        distance_moved = np.linalg.norm(enemy.position - initial_pos)
        assert distance_moved > 0.01

    def test_tick_enemies_skips_dead(self):
        """Test that tick_enemies doesn't update dead enemies."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=0, seed=42)
        bg.reset()

        enemy = EnemyNPC(position=(5.0, 5.0), health=1.0, strength=20.0)
        bg.enemies.append(enemy)

        # Kill enemy
        enemy.take_damage(1.0)
        dead_pos = enemy.position.copy()

        # Tick enemies
        bg.tick_enemies((5.5, 5.0))

        # Dead enemy should not move
        assert np.allclose(enemy.position, dead_pos)

    def test_tick_enemies_all_enemies(self):
        """Test that tick_enemies updates all enemies."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=3, seed=42)
        start, goal = bg.reset()

        agent_pos = (10.0, 10.0)

        initial_positions = [e.position.copy() for e in bg.enemies]

        bg.tick_enemies(agent_pos)

        # At least some enemies should have moved
        moved = 0
        for i, enemy in enumerate(bg.enemies):
            if not np.allclose(enemy.position, initial_positions[i]):
                moved += 1

        assert moved > 0


class TestBattlegroundCheckCollision:
    """Test collision detection with obstacles and enemies."""

    def test_check_collision_with_obstacle(self):
        """Test collision detection with obstacles."""
        bg = BattlegroundEnvironment(grid_size=20, num_obstacles=1, seed=42)
        bg.reset()

        # Collision should be detected at obstacle center
        obstacle_x, obstacle_y, obstacle_rad = bg.obstacles[0]
        assert bg.check_collision((obstacle_x, obstacle_y), radius=0.1)

    def test_check_collision_with_boundary(self):
        """Test collision detection with boundaries."""
        bg = BattlegroundEnvironment(grid_size=20, num_obstacles=0, seed=42)
        bg.reset()

        # Collision at boundary
        assert bg.check_collision((-0.05, 10.0), radius=0.1)
        assert bg.check_collision((20.05, 10.0), radius=0.1)
        assert bg.check_collision((10.0, -0.05), radius=0.1)
        assert bg.check_collision((10.0, 20.05), radius=0.1)

    def test_check_collision_free_space(self):
        """Test no collision in free space."""
        bg = BattlegroundEnvironment(grid_size=20, num_obstacles=0, seed=42)
        bg.reset()

        assert not bg.check_collision((10.0, 10.0), radius=0.1)

    def test_check_collision_with_enemy(self):
        """Test collision detection with enemies."""
        bg = BattlegroundEnvironment(grid_size=20, num_obstacles=0, seed=42)
        bg.reset()

        # Clear all enemies
        bg.enemies = []

        # Create enemy at known location
        enemy = EnemyNPC(position=(5.0, 5.0), health=50.0, strength=20.0)
        bg.enemies.append(enemy)

        # Should detect collision at enemy position
        assert bg.check_collision((5.0, 5.0), radius=0.1)

    def test_check_collision_enemy_radius(self):
        """Test collision detection respects enemy radius."""
        bg = BattlegroundEnvironment(grid_size=20, num_obstacles=0, seed=42)
        bg.reset()

        bg.enemies = []
        enemy = EnemyNPC(position=(5.0, 5.0), health=50.0, strength=20.0)
        bg.enemies.append(enemy)

        # Enemy radius is 0.2, so distance 0.25 should collide with agent radius 0.1
        assert bg.check_collision((5.0 + 0.25, 5.0), radius=0.1)

        # Distance 0.35 should not collide (0.35 > 0.2 + 0.1)
        assert not bg.check_collision((5.0 + 0.35, 5.0), radius=0.1)

    def test_check_collision_dead_enemy_no_collision(self):
        """Test that dead enemies don't cause collisions."""
        bg = BattlegroundEnvironment(grid_size=20, num_obstacles=0, seed=42)
        bg.reset()

        bg.enemies = []
        enemy = EnemyNPC(position=(5.0, 5.0), health=1.0, strength=20.0)
        bg.enemies.append(enemy)

        # Kill enemy
        enemy.take_damage(1.0)

        # Should not collide with dead enemy
        assert not bg.check_collision((5.0, 5.0), radius=0.1)

    def test_check_collision_multiple_enemies(self):
        """Test collision with multiple enemies."""
        bg = BattlegroundEnvironment(grid_size=20, num_obstacles=0, seed=42)
        bg.reset()

        bg.enemies = []
        enemy1 = EnemyNPC(position=(5.0, 5.0), health=50.0, strength=20.0)
        enemy2 = EnemyNPC(position=(15.0, 15.0), health=50.0, strength=20.0)
        bg.enemies.extend([enemy1, enemy2])

        # Collision with first enemy
        assert bg.check_collision((5.0, 5.0), radius=0.1)

        # Collision with second enemy
        assert bg.check_collision((15.0, 15.0), radius=0.1)

        # No collision in between
        assert not bg.check_collision((10.0, 10.0), radius=0.1)


class TestBattlegroundIntegration:
    """Integration tests combining multiple features."""

    def test_full_battleground_lifecycle(self):
        """Test complete battleground lifecycle."""
        bg = BattlegroundEnvironment(grid_size=20, num_obstacles=5, num_enemies=3, seed=42)

        # Reset and verify
        start, goal = bg.reset()
        assert len(bg.enemies) == 3
        assert all(e.alive for e in bg.enemies)

        # Simulate agent movement and enemy updates
        agent_pos = start
        for step in range(10):
            nearby = bg.get_enemy_near(agent_pos, radius=10.0)
            bg.tick_enemies(agent_pos)

            # Move agent slightly
            agent_pos = (agent_pos[0] + 0.1, agent_pos[1] + 0.1)

        # All enemies should still be in list
        assert len(bg.enemies) == 3

    def test_battleground_with_enemy_death(self):
        """Test battleground with enemy death."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=3, seed=42)
        start, goal = bg.reset()

        # Kill one enemy
        bg.enemies[0].take_damage(bg.enemies[0].health)

        # get_enemies should return 2
        alive = bg.get_enemies()
        assert len(alive) == 2

        # Full list still has 3
        assert len(bg.enemies) == 3

    def test_battleground_behavior_transitions(self):
        """Test behavior transitions as agent moves."""
        bg = BattlegroundEnvironment(grid_size=20, num_enemies=1, seed=42)
        start, goal = bg.reset()

        # Position enemy at known location
        bg.enemies[0].position = np.array([10.0, 10.0], dtype=np.float32)
        bg.enemies[0].strength = 60.0

        # Agent far away
        bg.tick_enemies((0.0, 0.0))
        assert bg.enemies[0].behavior == "random_walk"

        # Agent close, high strength -> chase
        bg.tick_enemies((10.5, 10.0))
        assert bg.enemies[0].behavior == "chase"

        # Agent very close with LOW strength -> flee (distance < 3.0 and low strength)
        bg.enemies[0].strength = 30.0
        bg.tick_enemies((10.1, 10.0))
        assert bg.enemies[0].behavior == "flee"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
