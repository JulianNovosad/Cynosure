import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.config import Config, LRSGConfig, ArenaConfig, AgentConfig, SimulationConfig
from src.simulation import Simulation


def test_simulation_initialization():
    cfg = Config(
        lrsg=LRSGConfig(num_neurons=10000, num_input_neurons=100, num_output_neurons=12),
        arena=ArenaConfig(grid_size=(50, 50, 50)),
        simulation=SimulationConfig(num_agents=5, max_ticks=10),
    )
    sim = Simulation(cfg)
    assert len(sim.agents) == 5
    assert sim.tick_count == 0
    assert all(a.alive for a in sim.agents)


@pytest.mark.cuda
def test_simulation_tick():
    cfg = Config(
        lrsg=LRSGConfig(num_neurons=10000, num_input_neurons=100, num_output_neurons=12),
        arena=ArenaConfig(grid_size=(50, 50, 50)),
        simulation=SimulationConfig(num_agents=5, max_ticks=10),
    )
    sim = Simulation(cfg)
    sim.tick()
    assert sim.tick_count == 1


@pytest.mark.cuda
def test_simulation_run():
    cfg = Config(
        lrsg=LRSGConfig(num_neurons=10000, num_input_neurons=100, num_output_neurons=12),
        arena=ArenaConfig(grid_size=(50, 50, 50)),
        simulation=SimulationConfig(num_agents=5, max_ticks=20),
    )
    sim = Simulation(cfg)
    metrics = sim.run()
    assert sim.tick_count > 0
    assert "survival" in metrics


@pytest.mark.cuda
def test_simulation_status():
    cfg = Config(
        lrsg=LRSGConfig(num_neurons=10000, num_input_neurons=100, num_output_neurons=12),
        arena=ArenaConfig(grid_size=(50, 50, 50)),
        simulation=SimulationConfig(num_agents=5, max_ticks=5),
    )
    sim = Simulation(cfg)
    sim.tick()
    status = sim.status
    assert "tick" in status
    assert "agents_alive" in status
    assert "safe_zone_radius" in status


@pytest.mark.cuda
def test_simulation_all_dead():
    cfg = Config(
        lrsg=LRSGConfig(num_neurons=10000, num_input_neurons=100, num_output_neurons=12),
        arena=ArenaConfig(grid_size=(50, 50, 50)),
        simulation=SimulationConfig(num_agents=2, max_ticks=1000),
    )
    sim = Simulation(cfg)
    metrics = sim.run()
    alive_end = metrics["survival"][-1] if metrics["survival"] else 0
    assert sim.tick_count > 0


@pytest.mark.cuda
def test_linguistic_tick():
    cfg = Config(
        lrsg=LRSGConfig(num_neurons=10000, num_input_neurons=100, num_output_neurons=12),
        arena=ArenaConfig(grid_size=(50, 50, 50)),
        simulation=SimulationConfig(num_agents=3, max_ticks=20, use_announcer=True),
    )
    sim = Simulation(cfg)
    for _ in range(10):
        sim.tick()
    assert len(sim.announcer.history) >= 1


if __name__ == "__main__":
    test_simulation_initialization()
    print("[PASS] test_simulation_initialization")
    test_simulation_tick()
    print("[PASS] test_simulation_tick")
    test_simulation_run()
    print("[PASS] test_simulation_run")
    test_simulation_status()
    print("[PASS] test_simulation_status")
    test_simulation_all_dead()
    print("[PASS] test_simulation_all_dead")
    test_linguistic_tick()
    print("[PASS] test_linguistic_tick")
    print("All simulation tests passed!")
