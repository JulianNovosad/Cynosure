"""
Integration test for semantic encoder wiring into simulation loop.
Verifies that semantic state encoding doesn't break learning and simulation stability.
"""

import torch
import pytest
import numpy as np
from src.lrsg.semantic_encoder import SemanticEncoder
from src.simulation.sensors import SensorSimulator
from src.config import Config, SimulationConfig
from src.simulation.loop import Simulation


def test_semantic_encoder_output_shape():
    """Test that semantic encoder produces correct 57D output."""
    encoder = SemanticEncoder(device='cpu')

    # Create a dummy sensor dict
    sensor_dict = {
        'vision': [(10.0, 5.0, 15.0, 1, 0.95), (20.0, 10.0, 30.0, 2, 0.87)],
        'lidar': [(i * 1.0, 5.0 + i * 0.1) for i in range(360)],
        'imu': {'accel': (0.1, -0.05, 9.8), 'gyro': (0.01, 0.02, 0.0), 'mag': (1.0, 0.5, -0.3)},
        'motor_feedback': {'left_speed': 1.0, 'right_speed': 0.95},
        'ram': 75.0,
        'position': (10.0, 5.0, 0.5),
        'time': 100,
    }

    output = encoder.forward(sensor_dict)

    assert isinstance(output, torch.Tensor), "Output must be torch.Tensor"
    assert output.shape == (57,), f"Expected shape (57,), got {output.shape}"
    assert output.dtype == torch.float32, f"Expected dtype float32, got {output.dtype}"
    assert torch.all(output >= -1.0) and torch.all(output <= 1.0), "Output must be in range [-1, 1]"


def test_sensor_simulator_apply_noise():
    """Test that sensor simulator applies noise correctly."""
    sim = SensorSimulator(device='cpu')

    sensor_dict = {
        'vision': [(1.0, 2.0, 3.0, 1, 0.9)],
        'lidar': [(0.0, 10.0), (np.pi, 15.0)],
        'imu': {'accel': (0.0, 0.0, 9.8), 'gyro': (0.0, 0.0, 0.0), 'mag': (1.0, 0.0, 0.0)},
        'motor_feedback': {'left_speed': 1.0, 'right_speed': 0.95},
        'ram': 50.0,
        'position': (10.0, 20.0, 0.5),
        'time': 100,
    }

    noisy = sim.apply_noise(sensor_dict)

    # Should preserve structure
    assert 'vision' in noisy
    assert 'lidar' in noisy
    assert 'imu' in noisy
    assert len(noisy['vision']) == len(sensor_dict['vision'])
    assert len(noisy['lidar']) == len(sensor_dict['lidar'])

    # Lidar distances should remain non-negative
    for angle, dist in noisy['lidar']:
        assert dist >= 0.0, f"Distance {dist} should be non-negative"


def test_semantic_integration_basic():
    """Test that simulation runs with use_semantic=True for 100 ticks."""
    config = Config()
    config.simulation.use_semantic = True
    config.simulation.num_agents = 1
    config.simulation.max_ticks = 100
    config.simulation.headless = True

    sim = Simulation(config)

    # Should initialize semantic encoder
    assert sim.semantic_encoder is not None, "Semantic encoder should be initialized"
    assert sim.sensor_simulator is not None, "Sensor simulator should be initialized"

    # Run 100 ticks without crashing
    for tick in range(100):
        sim.tick()
        assert sim.agents[0].alive, f"Agent should stay alive at tick {tick}"
        assert not torch.isnan(torch.tensor(sim.agents[0].ram)), f"RAM should not be NaN at tick {tick}"


def test_semantic_integration_without_semantic():
    """Test backward compatibility: simulation still works with use_semantic=False."""
    config = Config()
    config.simulation.use_semantic = False
    config.simulation.num_agents = 1
    config.simulation.max_ticks = 50
    config.simulation.headless = True

    sim = Simulation(config)

    # Should NOT initialize semantic encoder
    assert sim.semantic_encoder is None, "Semantic encoder should be None when disabled"
    assert sim.sensor_simulator is None, "Sensor simulator should be None when disabled"

    # Run 50 ticks
    for tick in range(50):
        sim.tick()
        assert sim.agents[0].alive, f"Agent should stay alive at tick {tick}"


def test_context_buffer_update():
    """Test that semantic state properly updates context buffer."""
    config = Config()
    config.simulation.use_semantic = True
    config.simulation.num_agents = 1
    config.simulation.max_ticks = 10
    config.simulation.headless = True

    sim = Simulation(config)
    agent = sim.agents[0]

    # Record initial context buffer
    initial_context = agent.context_buffer.clone()

    # Run 10 ticks
    for tick in range(10):
        sim.tick()

    # Context buffer should have changed
    final_context = agent.context_buffer
    assert not torch.allclose(initial_context, final_context), "Context buffer should change over time"

    # Last 57 dims should be semantic state (in range [-1, 1] after tanh)
    semantic_state = final_context[-57:]
    assert torch.all(semantic_state >= -1.0) and torch.all(semantic_state <= 1.0), \
        "Semantic state in context buffer should be in [-1, 1]"


def test_semantic_integration_no_nans():
    """Test that semantic integration doesn't introduce NaNs."""
    config = Config()
    config.simulation.use_semantic = True
    config.simulation.num_agents = 2
    config.simulation.max_ticks = 50
    config.simulation.headless = True

    sim = Simulation(config)

    for tick in range(50):
        sim.tick()

        for agent in sim.agents:
            if agent.alive:
                assert not torch.isnan(agent.brain.activations).any(), \
                    f"Brain activations contain NaN at tick {tick}"
                assert not torch.isnan(agent.context_buffer).any(), \
                    f"Context buffer contains NaN at tick {tick}"
                assert not torch.isnan(torch.tensor(agent.ram)), \
                    f"RAM is NaN at tick {tick}"
