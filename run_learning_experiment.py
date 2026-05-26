#!/usr/bin/env python3
"""run_learning_experiment.py

Controlled learning experiment: tests whether LRSG Hebbian learning enables
a single agent to learn to navigate to a RAM node over repeated episodes.

Output: experiment_results/learning_experiment.json

Runs 100 seeds × 2 conditions × 10 episodes × 500 ticks.
"""

import json
import math
import os
import random
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import LRSGConfig, ArenaConfig, AgentConfig
from src.lrsg import LRSGraph, forward_pass, hebbian_update
from src.lrsg.io import encode_input, decode_output
from src.arena.environment import Environment
from src.arena.ram_node import RAMNode
from src.agent import Agent

# Experiment parameters
NUM_SEEDS = 100
NUM_EPISODES = 10
MAX_TICKS = 500
NUM_NEURONS = 100_000
COLLECT_DIST = 4.0
GRID = (10, 10, 10)
SPAWN = (2.0, 2.0, 2.0)
TARGET = (8.0, 8.0, 8.0)
DIRS = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
OUTPUT_DIR = "experiment_results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "learning_experiment.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print(f"Device: {device}")


def make_structures(seed):
    """Create shared graph structure (positions, targets) with a fresh weight init."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cfg = LRSGConfig(
        num_neurons=NUM_NEURONS, max_out_degree=12, h_window=4096,
        disable_language=True,
    )
    brain = LRSGraph(cfg, device)
    # Save the structural template for reuse across conditions
    struct = {
        "out_targets": brain.out_targets.detach().clone(),
        "positions": brain.positions.detach().clone(),
        "block_ids": brain.block_ids.detach().clone(),
    }
    weights_init = brain.out_weights.detach().clone()
    return brain, cfg, struct, weights_init


def apply_structure(brain, struct):
    """Apply saved structure to a brain."""
    brain.out_targets.copy_(struct["out_targets"])
    brain.positions.copy_(struct["positions"])
    brain.block_ids.copy_(struct["block_ids"])


def run_episode(agent, env, lrsg_cfg, is_learning):
    """Run one episode. Returns (ticks_to_collect, collected, final_distance)."""
    agent.position = list(SPAWN)
    agent._tick_count = 0
    agent.last_action = -1
    agent.last_direction = -1
    agent.brain.reset_activations()

    for tick in range(MAX_TICKS):
        agent._compute_tactical_state(env, [])
        vision = agent._compute_vision(env, []).to(device)
        vitals = agent.vitals.to(device)
        whisper = agent.whisper_buffer.to(device)
        encoded = encode_input(vision, vitals, whisper, lrsg_cfg)

        agent.brain.reset_activations()
        agent.brain.inject_input(encoded)
        forward_pass(agent.brain, lrsg_cfg, theta=agent.theta, last_action=agent.last_action)

        decoded = decode_output(agent.brain.read_output(), lrsg_cfg, tick=tick, graph=agent.brain)
        agent.last_action = decoded["action"]
        agent.last_direction = decoded["direction"]

        if agent.last_action == 3 and 0 <= agent.last_direction < 6:
            d = DIRS[agent.last_direction]
            agent.position = tuple(
                max(0, min(GRID[i] - 1, agent.position[i] + d[i])) for i in range(3)
            )

        agent._tick_count += 1

        dist = math.sqrt(sum((agent.position[i] - TARGET[i])**2 for i in range(3)))
        reward = 1.0 if dist <= COLLECT_DIST else 0.0

        if reward > 0:
            if is_learning:
                hebbian_update(agent.brain, lrsg_cfg, reward,
                               learning_rate=agent.learning_rate, theta=agent.theta)
            return tick + 1, True, dist

        # Only call Hebbian when reward > 0 (reward=0 does nothing anyway)
        # Baseline skips entirely (LR=0 => no weight change)

    final_dist = math.sqrt(sum((agent.position[i] - TARGET[i])**2 for i in range(3)))
    return MAX_TICKS, False, final_dist


def run_seed(seed):
    """Run one seed: both learning and baseline conditions."""
    brain, lrsg_cfg, struct, weights_init = make_structures(seed)

    arena_cfg = ArenaConfig(
        grid_size=GRID, safe_zone_initial_radius=20.0, safe_zone_contract_fixed=0.0,
        safe_zone_damage_rate=0.0, safe_zone_weight_corruption_rate=0.0,
        ram_node_spawn_interval=10_000_000, max_ram_nodes=0,
    )
    agent_cfg = AgentConfig(
        initial_ram=200, ram_decay_rate=0.0, vision_range=10, sensory_radius=15.0,
    )

    # Create agent once to capture randomized params, then reuse
    agent = Agent(0, SPAWN, brain, agent_cfg, lrsg_cfg)
    base_theta = agent.theta
    base_lr = agent.learning_rate

    env = Environment(arena_cfg)
    env.ram_nodes[:] = [RAMNode(TARGET, 0)]

    results = {"seed": seed, "learning": [], "baseline": []}

    for condition, is_learning in [("learning", True), ("baseline", False)]:
        # Reset to initial state
        brain.out_weights.data.copy_(weights_init)
        apply_structure(brain, struct)
        agent.theta = base_theta
        agent.learning_rate = base_lr if is_learning else 0.0
        agent.position = list(SPAWN)
        agent.brain.reset_activations()

        ep_records = []
        for ep in range(NUM_EPISODES):
            ticks, collected, dist = run_episode(agent, env, lrsg_cfg, is_learning)
            ep_records.append({
                "episode": ep + 1,
                "ticks_to_collect": ticks,
                "collected": collected,
                "final_distance": dist,
            })
        results[condition] = ep_records

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Resume: find already-completed seeds
    completed_seeds = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        completed_seeds.add(rec["seed"])
                    except json.JSONDecodeError:
                        pass
        print(f"Resuming: {len(completed_seeds)} seeds already completed")

    completed = len(completed_seeds)
    failed = 0

    for seed in range(NUM_SEEDS):
        if seed in completed_seeds:
            continue

        t0 = time.time()
        try:
            result = run_seed(seed)
            completed += 1

            for cond in ("learning", "baseline"):
                record = {
                    "seed": seed,
                    "condition": cond,
                    "episodes": result[cond],
                }
                with open(OUTPUT_FILE, "a") as f:
                    f.write(json.dumps(record) + "\n")

            lr_coll = sum(1 for e in result["learning"] if e["collected"])
            bl_coll = sum(1 for e in result["baseline"] if e["collected"])
            t1 = time.time()
            print(f"Seed {seed:3d} | L:{lr_coll}/10 B:{bl_coll}/10 | {t1-t0:.1f}s | ok={completed} fail={failed}")
        except Exception as e:
            failed += 1
            print(f"Seed {seed:3d} FAILED: {e}")
            traceback.print_exc()

    print(f"\nDone. {completed} seeds completed, {failed} failed.")
    if completed < 80:
        print(f"WARNING: Only {completed} seeds completed (< 80). Architecture unstable.")

    # Rewrite as a single JSON array
    records = []
    with open(OUTPUT_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    with open(OUTPUT_FILE, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Results: {OUTPUT_FILE} ({len(records)} records)")


if __name__ == "__main__":
    main()
