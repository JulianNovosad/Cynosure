import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.config import Config, SimulationConfig
from src.simulation.loop import Simulation


def test_run_simulation_without_semantic():
    """Test that simulation runs with old behavior (no semantic, no physics)."""
    config = Config()
    config.simulation.use_semantic = False
    config.simulation.use_physics = False
    config.simulation.num_agents = 2
    config.simulation.headless = True

    sim = Simulation(config, device=torch.device("cpu"))

    # Verify initial state
    assert len(sim.agents) == 2
    assert all(agent.alive for agent in sim.agents)

    # Run 50 ticks
    for _ in range(50):
        sim.tick()

    # Verify simulation completed without crashes
    assert sim.tick_count == 50
    assert len(sim.agents) >= 1  # At least one agent should still be alive


def test_run_learning_experiment_without_semantic():
    """Test that learning experiment runs with legacy settings (no semantic, no physics)."""
    config = Config()
    config.simulation.use_semantic = False
    config.simulation.use_physics = False
    config.simulation.num_agents = 2
    config.simulation.headless = True

    sim = Simulation(config, device=torch.device("cpu"))

    # Verify initial state
    assert len(sim.agents) == 2
    for agent in sim.agents:
        assert agent.alive
        assert agent.ram > 0.0

    # Run 100 ticks
    for _ in range(100):
        sim.tick()

    # Verify state is valid
    assert sim.tick_count == 100
    for agent in sim.agents:
        if agent.alive:
            assert agent.ram >= 0.0
            assert isinstance(agent.position, tuple)
            assert len(agent.position) == 3


def test_physics_without_semantic():
    """Test that physics runs independent of semantic encoding."""
    config = Config()
    config.simulation.use_semantic = False
    config.simulation.use_physics = True
    config.simulation.num_agents = 2
    config.simulation.headless = True

    sim = Simulation(config, device=torch.device("cpu"))

    # Verify semantic is disabled but physics is enabled
    assert config.simulation.use_semantic == False
    assert config.simulation.use_physics == True

    # Semantic components should be None
    assert sim.semantic_encoder is None
    assert sim.sensor_simulator is None

    # Run 50 ticks with physics enabled
    for _ in range(50):
        sim.tick()

    # Verify physics simulation completed without errors
    assert sim.tick_count == 50
    assert len(sim.agents) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
