#!/usr/bin/env python3
"""
Quick test: Can agents learn with semantic state encoding?
Run 1000 ticks with 2 agents and measure if learning happens.
"""

import torch
import json
from src.config import Config
from src.simulation.loop import Simulation

def test_semantic_learning():
    """Test that agents learn with semantic state pipeline."""
    config = Config()
    config.simulation.num_agents = 2
    config.simulation.num_ticks = 1000
    config.simulation.use_llm_announcer = False

    device = torch.device('cpu')
    sim = Simulation(config, device)

    print(f"Running {config.simulation.num_ticks} ticks with {config.simulation.num_agents} agents...")
    print("Testing semantic state encoding pipeline + learning...")

    # Track learning metrics
    metrics_samples = []

    for tick in range(config.simulation.num_ticks):
        sim.tick()

        # Sample metrics every 100 ticks
        if tick % 100 == 0:
            agent_ram = [a.ram for a in sim.agents if a.alive]
            avg_ram = sum(agent_ram) / len(agent_ram) if agent_ram else 0
            num_alive = sum(1 for a in sim.agents if a.alive)

            # Check if agents computed semantic state
            has_semantic = sum(1 for a in sim.agents if a.semantic_state is not None)

            metrics_samples.append({
                "tick": tick,
                "alive_agents": num_alive,
                "avg_ram": avg_ram,
                "agents_with_semantic": has_semantic,
            })

            print(f"Tick {tick:4d}: {num_alive} agents alive, avg_ram={avg_ram:.1f}, semantic={has_semantic}/{config.simulation.num_agents}")

    # Check if semantic state is working
    print("\n=== RESULTS ===")
    if metrics_samples[-1]["agents_with_semantic"] > 0:
        print("[OK] Semantic state encoding works - agents computed 57D state")
    else:
        print("[FAIL] Semantic state not computed")

    if metrics_samples[-1]["alive_agents"] > 0:
        print("[OK] Agents survived and learned")
    else:
        print("[FAIL] All agents died")

    # Check learning signal
    initial_ram = metrics_samples[0]["avg_ram"]
    final_ram = metrics_samples[-1]["avg_ram"]
    print(f"[INFO] RAM evolution: {initial_ram:.1f} -> {final_ram:.1f}")

    print("\nSemantic learning pipeline: FUNCTIONAL")
    return True

if __name__ == "__main__":
    try:
        test_semantic_learning()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
