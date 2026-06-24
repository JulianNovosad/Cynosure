"""
Enemy NPC system for Cynosure embodied battleground.

Simple non-learning NPCs that move via state machine behaviors (random walk, chase, flee)
and can take damage and die. These are the basic antagonists in the embodied learning battleground.
"""

import numpy as np
from typing import Tuple, Dict


class EnemyNPC:
    """
    Simple NPC enemy that moves via state machine behaviors.

    Does not use LRSG learning. Controlled by behavior state machine:
    - random_walk: Move in random direction at max_speed
    - chase: Move toward agent at max_speed
    - flee: Move away from agent at max_speed

    Attributes:
        position: (x, y) tuple of enemy location
        health: Current health points (dies at 0)
        strength: Damage dealt per attack
        max_speed: Maximum movement speed per time step
        behavior: Current behavior state (random_walk, chase, flee)
        alive: Boolean flag indicating if enemy is alive
    """

    def __init__(
        self,
        position: Tuple[float, float],
        health: float,
        strength: float,
        max_speed: float = 0.5
    ):
        """
        Initialize an enemy NPC.

        Args:
            position: (x, y) tuple of starting position
            health: Initial health points
            strength: Damage dealt per attack
            max_speed: Maximum movement speed per time step (default 0.5)
        """
        self.position = np.array(position, dtype=np.float32)
        self.health = float(health)
        self.max_health = float(health)  # Store initial health for health threshold
        self.strength = float(strength)
        self.max_speed = float(max_speed)
        self.behavior = "random_walk"  # Default behavior
        self.alive = True

        # For random walk: track current direction
        self._random_direction = self._generate_random_direction()

    def _generate_random_direction(self) -> np.ndarray:
        """Generate random 2D unit direction vector."""
        angle = np.random.uniform(0, 2 * np.pi)
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

    def update(self, agent_pos: Tuple[float, float], dt: float = 0.1) -> None:
        """
        Update enemy position based on current behavior.

        Behavior selection logic:
        - If agent is within 2.0 units: chase if health > 25% of max_health else flee
        - Otherwise: random walk

        Args:
            agent_pos: (x, y) tuple of agent position
            dt: Time step for integration (default 0.1)
        """
        if not self.alive:
            return

        agent_pos = np.array(agent_pos, dtype=np.float32)
        dist_to_agent = np.linalg.norm(agent_pos - self.position)

        # Select behavior based on distance and health
        if dist_to_agent < 2.0:
            health_ratio = self.health / self.max_health
            if health_ratio > 0.25:  # > 25% health
                self.behavior = "chase"
            else:
                self.behavior = "flee"
        else:
            self.behavior = "random_walk"

        # Update position based on behavior
        if self.behavior == "random_walk":
            self._update_random_walk(dt)
        elif self.behavior == "chase":
            self._update_chase(agent_pos, dt)
        elif self.behavior == "flee":
            self._update_flee(agent_pos, dt)

    def _update_random_walk(self, dt: float) -> None:
        """Random walk: move in current direction, occasionally change direction."""
        # Change direction with some probability per update
        if np.random.random() < 0.1:  # 10% chance per update
            self._random_direction = self._generate_random_direction()

        # Move in current direction
        velocity = self._random_direction * self.max_speed
        self.position = self.position + velocity * dt

    def _update_chase(self, agent_pos: np.ndarray, dt: float) -> None:
        """Chase: move toward agent at max_speed."""
        direction = agent_pos - self.position
        dist = np.linalg.norm(direction)

        if dist > 0.001:  # Avoid division by zero
            direction = direction / dist
        else:
            direction = np.array([0.0, 0.0], dtype=np.float32)

        velocity = direction * self.max_speed
        self.position = self.position + velocity * dt

    def _update_flee(self, agent_pos: np.ndarray, dt: float) -> None:
        """Flee: move away from agent at max_speed."""
        direction = self.position - agent_pos
        dist = np.linalg.norm(direction)

        if dist > 0.001:  # Avoid division by zero
            direction = direction / dist
        else:
            direction = np.array([0.0, 0.0], dtype=np.float32)

        velocity = direction * self.max_speed
        self.position = self.position + velocity * dt

    def take_damage(self, amount: float) -> bool:
        """
        Apply damage to enemy.

        Args:
            amount: Damage amount to apply

        Returns:
            True if enemy dies (health <= 0), False otherwise
        """
        if not self.alive:
            return False

        self.health = max(0.0, self.health - amount)
        if self.health <= 0:
            self.alive = False
            return True
        return False

    def get_state(self) -> Dict:
        """
        Get enemy state as dictionary.

        Returns:
            Dict with position, health, strength, behavior, and alive status
        """
        return {
            "position": (float(self.position[0]), float(self.position[1])),
            "health": self.health,
            "strength": self.strength,
            "behavior": self.behavior,
            "alive": self.alive
        }
