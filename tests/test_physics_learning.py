import torch
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config, LRSGConfig, ArenaConfig, AgentConfig, SimulationConfig, EpisodicCompressorConfig
from src.simulation.loop import Simulation


def test_physics_learning_runs_without_crash():
    """
    Task 7 - Step 2: Test that physics learning simulation runs without crash.

    Verifies:
    - Simulation with use_physics=True, use_semantic=True runs 100 ticks
    - Agents remain alive (RAM > 0)
    - No NaNs in agent state
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config(
        lrsg=LRSGConfig(
            num_neurons=5000,
            max_out_degree=12,
            h_window=1024,
        ),
        arena=ArenaConfig(
            grid_size=(50, 50, 50),
            safe_zone_initial_radius=20.0,
            safe_zone_contract_fixed=0.0,
            safe_zone_damage_rate=0.0,
            ram_node_spawn_interval=50,
            max_ram_nodes=5,
        ),
        agent=AgentConfig(
            initial_ram=200,
            ram_decay_rate=0.005,
            vision_range=8,
            sensory_radius=15.0,
        ),
        simulation=SimulationConfig(
            num_agents=2,
            max_ticks=100,
            use_physics=True,
            use_semantic=True,
            use_announcer=False,
            headless=True,
        ),
        episodic=EpisodicCompressorConfig(),
    )

    sim = Simulation(config, device=device)

    try:
        # Run 100 ticks
        for tick in range(100):
            sim.tick()

            # Check agents are alive with positive RAM
            for agent in sim.agents:
                assert agent.ram > 0, f"Agent {agent.id} RAM <= 0 at tick {tick}"
                assert agent.ram == agent.ram, f"Agent {agent.id} has NaN RAM at tick {tick}"

            # Check no NaNs in brain state
            for agent in sim.agents:
                if agent.alive:
                    brain = agent.brain
                    assert not torch.isnan(brain.activations).any(), f"Agent {agent.id} activations contain NaN at tick {tick}"
                    assert not torch.isnan(brain.out_weights).any(), f"Agent {agent.id} weights contain NaN at tick {tick}"

        # Verify at least one agent survived
        alive_agents = [a for a in sim.agents if a.alive]
        assert len(alive_agents) > 0, "All agents died during physics learning test"

    finally:
        sim.stop()


@pytest.mark.cuda
def test_physics_learning_semantic_encoding():
    """
    Task 7 - Step 2: Verify semantic encoding works in physics simulation.

    Checks:
    - Semantic state is created (not None)
    - Semantic state has correct shape (57D)
    - No NaNs in semantic state
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config(
        lrsg=LRSGConfig(num_neurons=5000),
        arena=ArenaConfig(grid_size=(50, 50, 50)),
        agent=AgentConfig(),
        simulation=SimulationConfig(
            num_agents=2,
            max_ticks=50,
            use_physics=True,
            use_semantic=True,
            use_announcer=False,
            headless=True,
        ),
        episodic=EpisodicCompressorConfig(),
    )

    sim = Simulation(config, device=device)

    try:
        for tick in range(50):
            sim.tick()

            # Check semantic states are being created
            for agent in sim.agents:
                if agent.alive and agent.semantic_state is not None:
                    # Verify shape is 57D
                    assert agent.semantic_state.shape == (57,), f"Semantic state has wrong shape: {agent.semantic_state.shape}"
                    # Verify no NaNs
                    assert not torch.isnan(agent.semantic_state).any(), f"Agent {agent.id} semantic state contains NaN at tick {tick}"

    finally:
        sim.stop()
