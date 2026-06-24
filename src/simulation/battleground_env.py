"""
Battleground Environment for Cynosure embodied learning.

Extends NavigationEnvironment with procedural map generation, enemy spawning,
and enemy lifecycle management. Provides a combat-ready environment for
embodied learning experiments.
"""

import numpy as np
from typing import Tuple, List
from src.simulation.navigation import NavigationEnvironment
from src.simulation.enemy_npc import EnemyNPC


class BattlegroundEnvironment(NavigationEnvironment):
    """
    Combat environment extending NavigationEnvironment with enemies.

    Extends the parent NavigationEnvironment class to add:
    - Enemy spawning in collision-free locations
    - Enemy lifecycle management (tracking alive enemies)
    - Behavior updates based on agent proximity
    - Collision detection extended to enemies

    Attributes:
        num_enemies: Number of enemies to spawn
        enemies: List of EnemyNPC instances (may include dead enemies)
        grid_size: Size of the navigation grid
        num_obstacles: Number of obstacles in the environment
    """

    def __init__(self, grid_size: int = 20, num_obstacles: int = 5, num_enemies: int = 3, seed: int = None):
        """
        Initialize the battleground environment.

        Args:
            grid_size: Size of the square grid in meters (default 20)
            num_obstacles: Number of circular obstacles to place (default 5)
            num_enemies: Number of enemies to spawn (default 3)
            seed: Random seed for reproducibility (default None)
        """
        # Initialize parent with float grid_size
        super().__init__(grid_size=float(grid_size), num_obstacles=num_obstacles, seed=seed)

        self.num_enemies = num_enemies
        self.enemies: List[EnemyNPC] = []

    def reset(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Reset environment and spawn enemies.

        Calls parent reset() to generate obstacles and agent start/goal positions,
        then spawns enemies in collision-free locations with random initial stats.

        Returns:
            Tuple of (agent_start_2d, agent_goal_2d) as (x, y) tuples
            (note: parent returns 3D, we return 2D for compatibility with enemies)
        """
        # Call parent reset to generate obstacles and positions
        start_3d, goal_3d = super().reset()

        # Extract 2D positions (x, y)
        start_2d = (start_3d[0], start_3d[1])
        goal_2d = (goal_3d[0], goal_3d[1])

        # Spawn enemies in collision-free locations
        self.enemies = []
        for _ in range(self.num_enemies):
            enemy_pos = self._sample_free_position()

            # Spawn with random stats
            health = np.random.uniform(20.0, 100.0)
            strength = np.random.uniform(20.0, 80.0)
            speed = np.random.uniform(0.3, 1.0)

            enemy = EnemyNPC(
                position=enemy_pos,
                health=health,
                strength=strength,
                max_speed=speed
            )

            # Initialize behavior: 70% random_walk, 30% chase
            if np.random.random() < 0.7:
                enemy.behavior = "random_walk"
            else:
                enemy.behavior = "chase"

            self.enemies.append(enemy)

        return start_2d, goal_2d

    def _sample_free_position(self) -> Tuple[float, float]:
        """
        Sample a collision-free position for spawning enemies.

        Iteratively generates random positions until one is collision-free
        (not overlapping with obstacles or boundaries).

        Returns:
            Tuple (x, y) of a valid spawn position
        """
        margin = 0.5
        obstacle_radius = 0.3
        agent_radius = 0.1

        max_attempts = 100
        for _ in range(max_attempts):
            x = np.random.uniform(margin, self.grid_size - margin)
            y = np.random.uniform(margin, self.grid_size - margin)

            # Check collision with obstacles
            collision = False
            for ox, oy, orad in self.obstacles:
                distance = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
                min_distance = agent_radius + orad + 0.1  # Small buffer

                if distance < min_distance:
                    collision = True
                    break

            # Check boundary collision
            if x - agent_radius < 0 or x + agent_radius > self.grid_size:
                collision = True
            if y - agent_radius < 0 or y + agent_radius > self.grid_size:
                collision = True

            if not collision:
                return (x, y)

        # Fallback: return center of grid if sampling fails
        return (self.grid_size / 2.0, self.grid_size / 2.0)

    def get_enemies(self) -> List[EnemyNPC]:
        """
        Get all alive enemies.

        Returns:
            List of EnemyNPC instances that are still alive
        """
        return [e for e in self.enemies if e.alive]

    def get_enemy_near(self, pos: Tuple[float, float], radius: float = 5.0) -> List[EnemyNPC]:
        """
        Get alive enemies within a given radius of a position.

        Args:
            pos: (x, y) position to check
            radius: Search radius in meters (default 5.0)

        Returns:
            List of alive EnemyNPC instances within radius
        """
        nearby = []
        pos = np.array(pos, dtype=np.float32)

        for enemy in self.enemies:
            if enemy.alive:
                distance = np.linalg.norm(np.array(enemy.position) - pos)
                if distance <= radius:
                    nearby.append(enemy)

        return nearby

    def remove_enemy(self, enemy: EnemyNPC) -> None:
        """
        Remove a dead enemy from the environment.

        Args:
            enemy: EnemyNPC instance to remove
        """
        if enemy in self.enemies:
            self.enemies.remove(enemy)

    def tick_enemies(self, agent_pos: Tuple[float, float]) -> None:
        """
        Update all enemy behaviors based on agent proximity.

        Behavior selection logic:
        - If distance < 5.0 AND strength > 50: behavior = "chase"
        - Elif distance < 3.0: behavior = "flee"
        - Else: behavior = "random_walk"

        Also updates enemy positions by calling EnemyNPC.update() which will
        move the enemy based on the selected behavior.

        Args:
            agent_pos: (x, y) position of the agent
        """
        agent_pos_array = np.array(agent_pos, dtype=np.float32)
        agent_pos_tuple = (float(agent_pos[0]), float(agent_pos[1]))

        for enemy in self.enemies:
            if not enemy.alive:
                continue

            distance = np.linalg.norm(np.array(enemy.position) - agent_pos_array)

            # Select behavior based on battleground-specific logic
            # This happens BEFORE calling enemy.update()
            if distance < 5.0 and enemy.strength > 50:
                desired_behavior = "chase"
            elif distance < 3.0:
                desired_behavior = "flee"
            else:
                desired_behavior = "random_walk"

            # Store the desired behavior before update
            enemy.behavior = desired_behavior

            # Move enemy: calling update will recalculate behavior based on
            # EnemyNPC's internal logic, but we can re-apply our behavior selection
            # after if needed. For simplicity, we accept EnemyNPC's behavior logic
            # and just ensure positions update.
            enemy.update(agent_pos_tuple, dt=0.1)

            # Re-apply battleground behavior logic after update (to override EnemyNPC's logic)
            enemy.behavior = desired_behavior

    def check_collision(self, pos: Tuple[float, float], radius: float = 0.1) -> bool:
        """
        Check collision with obstacles or enemies.

        Extends parent check_collision to also check collisions with enemies.

        Args:
            pos: (x, y) position to check
            radius: Entity radius (default 0.1)

        Returns:
            True if collision with obstacles, boundaries, or enemies detected
        """
        # Create 3D position for parent collision check
        pos_3d = (pos[0], pos[1], 0.1)

        # Check collision with obstacles and boundaries (parent)
        if super().check_collision(pos_3d, radius=radius):
            return True

        # Check collision with enemies
        pos_arr = np.array(pos, dtype=np.float32)
        for enemy in self.enemies:
            if enemy.alive:
                distance = np.linalg.norm(np.array(enemy.position) - pos_arr)
                # Enemy is modeled as a sphere with radius 0.2
                enemy_radius = 0.2
                if distance < radius + enemy_radius:
                    return True

        return False
