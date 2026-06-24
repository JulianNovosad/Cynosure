"""
Combat system for Cynosure embodied learning battleground.

Provides mechanics for agent-enemy combat including:
- Strength calculation based on accumulated reward
- Attack resolution with hit probability and damage computation
- Full combat tick resolution
"""

from typing import Dict, Optional
import numpy as np


def compute_agent_strength(accumulated_reward: float) -> float:
    """
    Calculate agent's combat strength based on accumulated episode reward.

    Strength scales linearly with reward, representing how much the agent has
    learned and improved through the episode. Base strength is 50.0, scaling
    up at a rate of 0.1 per reward point, clamped to maximum 100.0.

    Args:
        accumulated_reward: Total reward accumulated in current episode

    Returns:
        float: Combat strength in range [50.0, 100.0]
    """
    strength = 50.0 + (accumulated_reward / 10.0)
    return np.clip(float(strength), 50.0, 100.0)


def combat_attack(
    attacker_strength: float,
    defender_strength: float,
    rng: Optional[np.random.Generator] = None
) -> Dict:
    """
    Resolve a single attack in combat.

    Hit probability scales with the difference between attacker and defender
    strength. A 50-point advantage gives 100% hit chance, 50-point disadvantage
    gives 0% hit chance. Damage dealt on hit is uniformly distributed between 5 and 20.

    Args:
        attacker_strength: Combat strength of attacking entity
        defender_strength: Combat strength of defending entity
        rng: Optional numpy random generator for deterministic testing.
             If None, uses default np.random.

    Returns:
        Dict with keys:
            - "hit": bool indicating if attack connected
            - "damage": float damage dealt (0 if missed)
            - "hit_probability": float hit probability used (before roll)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Calculate hit probability based on strength difference
    strength_diff = attacker_strength - defender_strength
    hit_probability = 0.5 + 0.05 * (strength_diff / 100.0)
    hit_probability = np.clip(float(hit_probability), 0.0, 1.0)

    # Roll for hit
    hit = rng.random() < hit_probability

    # Calculate damage if hit
    if hit:
        damage = float(rng.uniform(5.0, 20.0))
    else:
        damage = 0.0

    return {
        "hit": bool(hit),
        "damage": damage,
        "hit_probability": hit_probability
    }


def resolve_combat_tick(
    agent_strength: float,
    enemy,  # EnemyNPC type
    max_damage: float = 20.0
) -> Dict:
    """
    Resolve a complete combat tick where agent attacks enemy.

    Calls combat_attack() to determine hit and damage, applies damage to enemy,
    and determines outcome (victory, hit, or miss) with associated rewards.

    Args:
        agent_strength: Agent's combat strength
        enemy: EnemyNPC instance to attack
        max_damage: Maximum damage in combat_attack (default 20.0, for reference)

    Returns:
        Dict with keys:
            - "hit": bool indicating if attack connected
            - "damage": float damage dealt
            - "victory": bool indicating if enemy died
            - "reward": float reward earned (50 for victory, 5 for hit, -10 for miss)
            - "enemy_health_after": float enemy health remaining after damage
    """
    # Resolve attack
    attack_result = combat_attack(agent_strength, enemy.strength)

    # Apply damage to enemy
    hit = attack_result["hit"]
    damage = attack_result["damage"]
    enemy_died = enemy.take_damage(damage)

    # Determine reward
    if enemy_died:
        reward = 50.0
    elif hit:
        reward = 5.0
    else:
        reward = -10.0

    return {
        "hit": hit,
        "damage": damage,
        "victory": enemy_died,
        "reward": reward,
        "enemy_health_after": enemy.health
    }
