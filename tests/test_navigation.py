import pytest
import numpy as np
from src.simulation.navigation import NavigationEnvironment


def test_grid_generation():
    """Test grid generation and reset produces valid start/goal positions."""
    env = NavigationEnvironment(grid_size=10, num_obstacles=3, seed=42)
    start, goal = env.reset()

    assert len(start) == 3, "Start position should be (x, y, z)"
    assert len(goal) == 3, "Goal position should be (x, y, z)"
    assert 0 <= start[0] < 10 and 0 <= start[1] < 10, "Start should be in grid"
    assert 0 <= goal[0] < 10 and 0 <= goal[1] < 10, "Goal should be in grid"
    assert np.linalg.norm(np.array(start) - np.array(goal)) > 2.0, "Start and goal should be far apart"


def test_lidar_reading():
    """Test that LiDAR returns 8 distances bounded by grid size."""
    env = NavigationEnvironment(grid_size=10, num_obstacles=3, seed=42)
    start, goal = env.reset()
    lidar = env.get_lidar(agent_pos=start, num_rays=8)

    assert lidar.shape == (8,), "Lidar should return 8 rays"
    assert np.all(lidar >= 0), "Lidar distances should be non-negative"
    assert np.all(lidar <= 10), "Lidar distances should be bounded by grid size"


def test_collision_detection():
    """Test collision detection at boundaries and with obstacles."""
    env = NavigationEnvironment(grid_size=10, num_obstacles=3, seed=42)
    start, goal = env.reset()

    assert not env.check_collision(start), "Should not collide at start"

    # Check boundary collision - outside grid boundary
    bad_pos = (-0.05, 5, 0.1)
    assert env.check_collision(bad_pos, radius=0.1), "Should collide at boundary"


def test_deterministic_with_seed():
    """Test that same seed produces same start and goal positions."""
    env1 = NavigationEnvironment(grid_size=10, num_obstacles=3, seed=99)
    start1, goal1 = env1.reset()

    env2 = NavigationEnvironment(grid_size=10, num_obstacles=3, seed=99)
    start2, goal2 = env2.reset()

    assert np.allclose(start1, start2), "Same seed should produce same start"
    assert np.allclose(goal1, goal2), "Same seed should produce same goal"
