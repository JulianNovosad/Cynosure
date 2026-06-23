#!/usr/bin/env python3
"""run_learning_experiment.py

Controlled learning experiment: tests whether LRSG Hebbian learning enables
a single agent to learn to navigate to a RAM node over repeated episodes.

Output: experiment_results/learning_experiment_v3.json
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

from src.config import LRSGConfig, ArenaConfig, AgentConfig
from src.lrsg import LRSGraph, forward_pass, hebbian_update
from src.lrsg.io import encode_input, decode_output
from src.arena.environment import Environment
from src.arena.ram_node import RAMNode
from src.agent import Agent

class MockSanitizer:
    def check_probe(self, text, agent_id, tick): return "PASSED"
    def sanitize_agent_whisper(self, text, agent_id, tick): return text

# Constants
COLLECT_DIST = 1.5
GRID = (5, 5, 5)
SPAWN = (1.0, 1.0, 1.0)
TARGET = (3.5, 3.5, 3.5)
DIRS = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
COLLECTION_REWARD = 1.0
# FIX (22 Jun 2026): Increase shaped reward 10x to provide stronger learning signal
# Previous: 0.3 made distance-based reward too small (0.09 per step at best)
# New: 3.0 makes immediate feedback much stronger (0.9 per step)
SHAPED_REWARD_SCALE = 3.0  # Per-unit-distance shaped reward coefficient (CHANGED: 0.3 → 3.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def make_structures(seed, num_neurons, trace_decay, trace_cap):
    """Create shared graph structure (positions, targets) with a fresh weight init."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cfg = LRSGConfig(
        num_neurons=num_neurons, max_out_degree=12, h_window=1024,
        disable_language=True,
    )
    brain = LRSGraph(cfg, device)
    brain.trace_decay = trace_decay
    brain.trace_cap = trace_cap
    
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


def run_episode(agent, env, lrsg_cfg, is_learning, max_ticks, learning_rate):
    """Run one episode. Returns (ticks_to_collect, collected, final_distance)."""
    agent.position = list(SPAWN)
    agent._tick_count = 0
    agent.last_action = -1
    agent.last_direction = -1
    agent.brain.reset_activations()
    agent.brain.reset_traces()

    prev_dist = math.sqrt(sum((SPAWN[i] - TARGET[i])**2 for i in range(3)))
    accumulated_reward = 0.0

    for tick in range(max_ticks):
        agent._compute_tactical_state(env, [])
        vision = agent._compute_vision(env, []).to(device)
        vitals = agent.vitals.to(device)
        whisper = agent.whisper_buffer.to(device)
        encoded = encode_input(vision, vitals, whisper, lrsg_cfg)

        agent.brain.reset_activations()
        agent.brain.inject_input(encoded)
        forward_pass(agent.brain, lrsg_cfg, theta=agent.theta, last_action=agent.last_action, energy=agent.ram, learning_rate=learning_rate)

        decoded = decode_output(agent.brain.read_output(), lrsg_cfg, tick=tick, graph=agent.brain)
        agent.last_action = decoded["action"]
        agent.last_direction = decoded["direction"]

        if agent.last_action == 3 and 0 <= agent.last_direction < 6:
            d = DIRS[agent.last_direction]
            agent.position = tuple(
                max(0, min(GRID[i] - 0.1, agent.position[i] + d[i])) for i in range(3)
            )

        agent._tick_count += 1

        dist = math.sqrt(sum((agent.position[i] - TARGET[i])**2 for i in range(3)))
        collected = dist <= COLLECT_DIST

        # Accumulate shaped reward: positive for approach, negative for retreat
        accumulated_reward += (prev_dist - dist) * SHAPED_REWARD_SCALE
        prev_dist = dist

        if collected:
            accumulated_reward += COLLECTION_REWARD
            # Always reinforce collection with at least the collection reward
            effective_reward = max(COLLECTION_REWARD, accumulated_reward)
            if is_learning:
                hebbian_update(agent.brain, lrsg_cfg, effective_reward,
                               learning_rate=learning_rate, theta=agent.theta)
            return tick + 1, True, dist

    # End of episode: only reinforce positive outcomes; negative accumulated
    # reward means the agent wandered away — don't disrupt learned navigation.
    if is_learning and accumulated_reward > 1e-6:
        hebbian_update(agent.brain, lrsg_cfg, accumulated_reward,
                       learning_rate=learning_rate, theta=agent.theta)

    final_dist = math.sqrt(sum((agent.position[i] - TARGET[i])**2 for i in range(3)))
    return max_ticks, False, final_dist


def run_seed(seed, args):
    """Run one seed: both learning and baseline conditions."""
    # FIX (22 Jun 2026): Scale learning rate 10x to compensate for slow weight initialization
    # Previous: eta=0.01 was 100-1000x too weak with old weight scale
    # With 5x weight scale increase, we still need 2-10x learning rate boost
    adjusted_eta = args.eta * 10.0 if args.eta < 0.1 else args.eta

    brain, lrsg_cfg, struct, weights_init = make_structures(seed, args.neurons, args.lambda_val, args.trace_cap)

    arena_cfg = ArenaConfig(
        grid_size=GRID, safe_zone_initial_radius=10.0, safe_zone_contract_fixed=0.0,
        safe_zone_damage_rate=0.0, safe_zone_weight_corruption_rate=0.0,
        ram_node_spawn_interval=10_000_000, max_ram_nodes=0,
    )
    agent_cfg = AgentConfig(
        initial_ram=200, ram_decay_rate=0.0, vision_range=10, sensory_radius=10.0,
    )

    agent = Agent(0, SPAWN, brain, agent_cfg, lrsg_cfg, MockSanitizer())
    base_theta = agent.theta

    env = Environment(arena_cfg)
    env.ram_nodes[:] = [RAMNode(TARGET, 0)]

    results = {"seed": seed, "learning": [], "baseline": []}

    for condition, is_learning in [("learning", True), ("baseline", False)]:
        brain.out_weights.data.copy_(weights_init)
        apply_structure(brain, struct)
        agent.theta = base_theta
        agent.position = list(SPAWN)
        agent.brain.reset_activations()
        agent.brain.reset_traces()

        ep_records = []
        for ep in range(args.episodes):
            ticks, collected, dist = run_episode(agent, env, lrsg_cfg, is_learning, args.ticks, adjusted_eta)
            ep_records.append({
                "episode": ep + 1,
                "ticks_to_collect": ticks,
                "collected": collected,
                "final_distance": dist,
            })
        results[condition] = ep_records

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--neurons", type=int, default=5000)
    parser.add_argument("--lambda_val", type=float, default=0.95, help="Trace decay")
    parser.add_argument("--trace_cap", type=float, default=50.0, help="Trace safety cap")
    parser.add_argument("--eta", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--output", type=str, default="experiment_results/learning_experiment_v5.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if os.path.exists(args.output):
        os.remove(args.output)

    completed = 0
    failed = 0

    print(f"Device: {device}")
    print(f"Running {args.seeds} seeds, {args.episodes} episodes, lambda={args.lambda_val}, eta={args.eta}")

    for seed in range(args.seeds):
        t0 = time.time()
        try:
            result = run_seed(seed, args)
            completed += 1

            for cond in ("learning", "baseline"):
                record = {
                    "seed": seed,
                    "condition": cond,
                    "episodes": result[cond],
                }
                with open(args.output, "a") as f:
                    f.write(json.dumps(record) + "\n")

            lr_coll = sum(1 for e in result["learning"] if e["collected"])
            bl_coll = sum(1 for e in result["baseline"] if e["collected"])
            t1 = time.time()
            print(f"Seed {seed:3d} | L:{lr_coll}/{args.episodes} B:{bl_coll}/{args.episodes} | {t1-t0:.1f}s | ok={completed} fail={failed}")
        except Exception as e:
            failed += 1
            print(f"Seed {seed:3d} FAILED: {e}")
            traceback.print_exc()

    print(f"\nDone. {completed} seeds completed, {failed} failed.")

    records = []
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        with open(args.output, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Results: {args.output} ({len(records)} records)")


if __name__ == "__main__":
    main()
