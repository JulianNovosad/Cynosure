#!/usr/bin/env python3
"""run_physics_learning_experiment.py

Physics-based learning experiment: tests whether LRSG Hebbian learning with
physics simulation enables agents to learn resource collection over episodes.

Task 7 - Step 1: Create physics-based learning experiment runner
Output: experiment_results/physics_learning.json

Configuration:
- 2 agents in physics-based simulation (Pybullet)
- use_physics=True enables physics world
- use_semantic=True enables semantic state encoding
- Agents learn to approach and collect resources via Hebbian plasticity
- Tracks: collection_rate, final_ram per seed
"""

import json
import math
import os
import random
import sys
import time
import traceback
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config, LRSGConfig, ArenaConfig, AgentConfig, SimulationConfig, EpisodicCompressorConfig
from src.simulation.loop import Simulation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def run_experiment(seed, num_ticks, use_physics, use_semantic):
    """
    Run physics-based learning experiment.

    Args:
        seed: Random seed
        num_ticks: Number of ticks to run
        use_physics: If True, use Pybullet physics simulation
        use_semantic: If True, enable semantic state encoding

    Returns:
        dict with keys: seed, ticks, use_physics, collection_rate, final_ram
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Create configuration
    config = Config(
        lrsg=LRSGConfig(
            num_neurons=10000,
            max_out_degree=12,
            h_window=2048,
        ),
        arena=ArenaConfig(
            grid_size=(50, 50, 50),
            safe_zone_initial_radius=20.0,
            safe_zone_contract_fixed=0.0,
            safe_zone_damage_rate=0.0,
            safe_zone_weight_corruption_rate=0.0,
            ram_node_spawn_interval=50,
            max_ram_nodes=10,
        ),
        agent=AgentConfig(
            initial_ram=200,
            ram_decay_rate=0.005,
            vision_range=8,
            sensory_radius=15.0,
        ),
        simulation=SimulationConfig(
            num_agents=2,
            max_ticks=num_ticks,
            use_physics=use_physics,
            use_semantic=use_semantic,
            use_announcer=False,
            headless=True,
        ),
        episodic=EpisodicCompressorConfig(),
    )

    # Run simulation
    sim = Simulation(config, device=device)

    collection_counts = {agent.id: 0 for agent in sim.agents}
    prev_ram = {agent.id: agent.ram for agent in sim.agents}

    try:
        for tick in range(num_ticks):
            sim.tick()

            # Track collection: if RAM increased and agent performed COLLECT action
            for agent in sim.agents:
                if agent.alive and agent.last_action == 2:  # COLLECT action
                    ram_delta = agent.ram - prev_ram[agent.id]
                    if ram_delta > 0:  # Successful collection
                        collection_counts[agent.id] += 1
                prev_ram[agent.id] = agent.ram
    except Exception as e:
        print(f"  Exception during tick {tick}: {e}")
        traceback.print_exc()
    finally:
        sim.stop()

    # Compute metrics
    final_ram_values = [agent.ram for agent in sim.agents if agent.alive or True]
    final_ram = sum(final_ram_values) / len(final_ram_values) if final_ram_values else 0.0

    total_collections = sum(collection_counts.values())
    collection_rate = total_collections / (num_ticks * len(sim.agents)) if num_ticks > 0 else 0.0

    return {
        "seed": seed,
        "ticks": num_ticks,
        "use_physics": use_physics,
        "use_semantic": use_semantic,
        "collection_rate": round(collection_rate, 6),
        "final_ram": round(final_ram, 2),
        "total_collections": total_collections,
        "num_agents": len(sim.agents),
    }


def main():
    parser = argparse.ArgumentParser(description="Physics-based learning experiment")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to run")
    parser.add_argument("--ticks", type=int, default=500, help="Ticks per seed")
    parser.add_argument("--use-physics", action="store_true", default=True, help="Enable physics simulation")
    parser.add_argument("--use-semantic", action="store_true", default=True, help="Enable semantic encoding")
    parser.add_argument("--output", type=str, default="experiment_results/physics_learning.json", help="Output file")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Clean up previous results
    if os.path.exists(args.output):
        os.remove(args.output)

    results = []
    completed = 0
    failed = 0

    print(f"Device: {device}")
    print(f"Running {args.seeds} seeds × {args.ticks} ticks")
    print(f"use_physics={args.use_physics}, use_semantic={args.use_semantic}")
    print()

    for seed in range(args.seeds):
        t0 = time.time()
        try:
            result = run_experiment(
                seed=seed,
                num_ticks=args.ticks,
                use_physics=args.use_physics,
                use_semantic=args.use_semantic,
            )
            results.append(result)
            completed += 1

            t1 = time.time()
            coll_rate = result["collection_rate"]
            final_ram = result["final_ram"]
            print(f"Seed {seed:2d} | collection_rate={coll_rate:.4f} | final_ram={final_ram:.1f} | {t1-t0:.1f}s | ok={completed} fail={failed}")
        except Exception as e:
            failed += 1
            print(f"Seed {seed:2d} FAILED: {e}")
            traceback.print_exc()

    # Save results
    if results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {args.output}")

        # Print summary
        collection_rates = [r["collection_rate"] for r in results]
        final_rams = [r["final_ram"] for r in results]

        print(f"\nSummary ({completed} seeds completed, {failed} failed):")
        print(f"  Collection rate: {np.mean(collection_rates):.6f} ± {np.std(collection_rates):.6f}")
        print(f"  Final RAM: {np.mean(final_rams):.2f} ± {np.std(final_rams):.2f}")
    else:
        print(f"No results to save.")


if __name__ == "__main__":
    main()
