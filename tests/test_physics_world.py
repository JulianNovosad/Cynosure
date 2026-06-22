import pytest
from src.simulation.physics_world import PhysicsWorld


def test_physics_world_creation():
    """Test that physics world creates with correct agent count."""
    world = PhysicsWorld(num_agents=3, grid_size=50, use_gui=False)

    assert world.num_agents == 3
    assert world.grid_size == 50
    assert len(world.agent_bodies) == 3
    assert world.ground_id is not None

    world.close()


def test_physics_world_step():
    """Test that physics world steps without error."""
    world = PhysicsWorld(num_agents=2, grid_size=50, use_gui=False)

    actions = {0: 0, 1: 1}  # Agent 0: STOP, Agent 1: MOVE_FORWARD
    sensor_readings = world.step(actions)

    assert len(sensor_readings) == 2
    assert 'position' in sensor_readings[0]
    assert 'lidar' in sensor_readings[0]
    assert 'vision' in sensor_readings[0]

    world.close()
