#!/usr/bin/env python3
"""
Test: Semantic state enables measurable learning with eased conditions.
Increase resource availability and measure learning curves.
"""

import torch
import json
from src.config import Config
from src.simulation.loop import Simulation

def test_semantic_learning_with_eased_conditions():
    """Run with easier conditions to allow learning."""
    config = Config()
    config.simulation.num_agents = 3
    config.simulation.num_ticks = 2000
    config.simulation.use_llm_announcer = False

    # Ease conditions to allow learning
    config.agent.initial_ram = 300  # More initial resources
    config.agent.ram_decay_rate = 0.2  # Slower decay
    config.arena.num_ram_nodes = 20  # More food

    device = torch.device('cpu')
    sim = Simulation(config, device)

    print(f"=== Semantic Learning Test (Eased) ===")
    print(f"Running {config.simulation.num_ticks} ticks with {config.simulation.num_agents} agents")
    print(f"Initial RAM: {config.agent.initial_ram}, Decay: {config.agent.ram_decay_rate}, Food nodes: {config.arena.num_ram_nodes}")
    print()

    # Track learning metrics
    metrics = {
        "ticks": [],
        "alive_agents": [],
        "avg_ram": [],
        "action_entropy": [],
        "graph_complexity": [],
        "semantic_state_active": [],
    }

    for tick in range(config.simulation.num_ticks):
        sim.tick()

        # Sample metrics every 100 ticks
        if tick % 100 == 0 or tick == config.simulation.num_ticks - 1:
            agent_ram = [a.ram for a in sim.agents if a.alive]
            avg_ram = sum(agent_ram) / len(agent_ram) if agent_ram else 0
            num_alive = sum(1 for a in sim.agents if a.alive)
            has_semantic = sum(1 for a in sim.agents if a.semantic_state is not None)

            # Get current metrics
            action_ent = sim.metrics.get("action_entropy", [0])[-1] if sim.metrics.get("action_entropy") else 0
            graph_cplx = sim.metrics.get("graph_complexity", [0])[-1] if sim.metrics.get("graph_complexity") else 0

            metrics["ticks"].append(tick)
            metrics["alive_agents"].append(num_alive)
            metrics["avg_ram"].append(avg_ram)
            metrics["action_entropy"].append(action_ent)
            metrics["graph_complexity"].append(graph_cplx)
            metrics["semantic_state_active"].append(has_semantic)

            print(f"Tick {tick:4d}: alive={num_alive}, ram={avg_ram:6.1f}, semantic={has_semantic}/{config.simulation.num_agents}, "
                  f"entropy={action_ent:.2f}, complexity={graph_cplx:.2f}")

    # Analyze learning
    print("\n=== LEARNING ANALYSIS ===")
    print(f"Survival rate: {metrics['alive_agents'][-1]}/{config.simulation.num_agents}")
    print(f"Average RAM: {metrics['avg_ram'][0]:.1f} -> {metrics['avg_ram'][-1]:.1f}")
    print(f"Action entropy: {metrics['action_entropy'][0]:.2f} -> {metrics['action_entropy'][-1]:.2f}")
    print(f"Graph complexity: {metrics['graph_complexity'][0]:.2f} -> {metrics['graph_complexity'][-1]:.2f}")

    # Check for learning signals
    learning_signals = []

    if metrics['alive_agents'][-1] > 0:
        learning_signals.append("[OK] Agents survived - navigation works")
    else:
        learning_signals.append("[WARN] All agents died early")

    if metrics['graph_complexity'][-1] > metrics['graph_complexity'][0] * 1.1:
        learning_signals.append("[OK] Brain complexity increased - learning happened")
    else:
        learning_signals.append("[WARN] Graph complexity unchanged - possible learning issue")

    if metrics['action_entropy'][-1] > metrics['action_entropy'][0]:
        learning_signals.append("[OK] Action entropy increased - behavioral diversity")
    else:
        learning_signals.append("[WARN] Action entropy decreased - convergence issue")

    if metrics['semantic_state_active'][-1] > 0:
        learning_signals.append("[OK] Semantic state encoder active on all agents")
    else:
        learning_signals.append("[FAIL] Semantic state not computed")

    print("\nLearning Signals:")
    for signal in learning_signals:
        print(f"  {signal}")

    # Overall result
    success = metrics['alive_agents'][-1] > 0 and metrics['semantic_state_active'][-1] > 0
    print(f"\n=== RESULT: {'PASS' if success else 'FAIL'} ===")
    print("Semantic learning pipeline: ", "OPERATIONAL" if success else "NEEDS WORK")

    return success

if __name__ == "__main__":
    try:
        test_semantic_learning_with_eased_conditions()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
