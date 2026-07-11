"""Reward shaping for embodied agent navigation."""
import numpy as np
from typing import Tuple, Dict, Optional


def compute_embodied_reward(
    agent_pos,
    goal_pos,
    prev_distance,
    is_collision,
    is_goal_reached,
    distance_scale=0.5,
    collision_penalty=0.5,
    goal_bonus=10.0
) -> float:
    """
    Compute shaped reward signal for goal-seeking navigation.

    Reward components:
    - Distance reward: (prev_distance - curr_distance) * distance_scale
    - Collision penalty: -collision_penalty (overrides distance reward)
    - Goal bonus: +goal_bonus (overrides everything)

    Args:
        agent_pos: Tuple of (x, y, z) agent position
        goal_pos: Tuple of (x, y, z) goal position
        prev_distance: Previous distance to goal (scalar)
        is_collision: Whether agent collided with obstacle (bool)
        is_goal_reached: Whether agent reached the goal (bool)
        distance_scale: Scale factor for distance-based reward (default 0.5)
        collision_penalty: Magnitude of collision penalty (default 0.5)
        goal_bonus: Bonus for reaching goal (default 10.0)

    Returns:
        Scalar reward value
    """
    # Goal reached takes precedence - return the bonus
    if is_goal_reached:
        return goal_bonus

    # Collision takes second priority - return negative penalty
    if is_collision:
        return -collision_penalty

    # Compute current distance to goal
    agent_arr = np.array(agent_pos)
    goal_arr = np.array(goal_pos)
    curr_distance = np.linalg.norm(agent_arr - goal_arr)

    # Distance-based reward: positive if moving closer, negative if moving away
    distance_reward = (prev_distance - curr_distance) * distance_scale

    return distance_reward


def compute_embodied_reward_enhanced(
    agent_pos: Tuple[float, float],
    goal_pos: Tuple[float, float],
    prev_distance: float,
    is_collision: bool,
    is_goal_reached: bool,
    combat_result: Optional[Dict] = None,
) -> float:
    """
    Compute enhanced reward signal for goal-seeking navigation with combat and survival.

    Reward components (additive):
    1. Navigation reward: reward += 0.1 * (prev_distance - curr_distance)
       - Positive when moving closer to goal, negative when moving away
    2. Collision penalty: If collision, reward -= 5.0
    3. Goal bonus: If goal reached, reward += 100.0
    4. Combat reward: If combat_result provided, reward += combat_result["reward"]

    Args:
        agent_pos: Tuple of (x, y) agent position
        goal_pos: Tuple of (x, y) goal position
        prev_distance: Previous distance to goal (scalar)
        is_collision: Whether agent collided with obstacle (bool)
        is_goal_reached: Whether agent reached the goal (bool)
        combat_result: Optional dict from resolve_combat_tick() with keys:
            - "hit": bool
            - "damage": float
            - "victory": bool
            - "reward": float (50 for victory, 5 for hit, -10 for miss)
            - "enemy_health_after": float

    Returns:
        Scalar reward value (sum of all components)
    """
    reward = 0.0

    # 1. Navigation reward: reward for moving closer to goal, penalize moving away
    curr_distance = np.linalg.norm(np.array(agent_pos) - np.array(goal_pos))
    distance_reduction = prev_distance - curr_distance
    reward += 0.1 * distance_reduction

    # 2. Collision penalty: penalize collisions
    if is_collision:
        reward -= 5.0

    # 3. Goal bonus: reward for reaching goal
    if is_goal_reached:
        reward += 100.0

    # 4. Combat reward: add combat-derived rewards if available
    if combat_result is not None:
        reward += combat_result.get("reward", 0.0)

    return reward
