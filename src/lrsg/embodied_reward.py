"""Reward shaping for embodied agent navigation."""
import numpy as np


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
