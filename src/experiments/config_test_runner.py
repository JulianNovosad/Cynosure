#!/usr/bin/env python3
"""config_test_runner.py

Test three Cynosure AI configurations to measure performance, learning, and stability.

Configurations:
1. Compact:    2^20 (1M) neurons, max_out_degree 24 (~24M synapses)
2. Standard:   2^22 (4M) neurons, max_out_degree 24 (~96M synapses)
3. Large:      2^25 (32M) neurons, max_out_degree 12 (~384M synapses, reduced degree to fit)

Metrics measured:
- Initialization time and VRAM usage
- Tick latency (physics + linguistics)
- Learning effectiveness (collection rate on a navigation task)
- Network stability (no NaNs, graph complexity, rewire rate)
- Agent survival and RAM economy
"""

import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch

# Add project root to path
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, project_root)

from src.config import Config, LRSGConfig, ArenaConfig, AgentConfig, SimulationConfig
from src.simulation import Simulation
from src.lrsg import LRSGraph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0


# Configuration definitions
CONFIG_1 = {
    "name": "Compact",
    "neurons": 2**20,          # 1,048,576
    "max_out_degree": 24,
    "grid_size": (64, 64, 64),
    "num_agents": 2,
    "max_ticks": 500,
}

CONFIG_2 = {
    "name": "Standard",
    "neurons": 2**22,          # 4,194,304
    "max_out_degree": 24,
    "grid_size": (128, 128, 128),
    "num_agents": 3,
    "max_ticks": 500,
}

CONFIG_3 = {
    "name": "Large",
    "neurons": 2**25,          # 33,554,432
    "max_out_degree": 12,      # Reduced to fit in VRAM
    "grid_size": (256, 256, 256),
    "num_agents": 2,
    "max_ticks": 200,          # Fewer ticks due to size
}

CONFIGS = [CONFIG_1, CONFIG_2, CONFIG_3]


def calculate_synapse_count(num_neurons: int, max_out_degree: int) -> int:
    """Calculate approximate synapse count (neurons × avg_out_degree)."""
    return int(num_neurons * max_out_degree * 0.7)  # ~70% sparsity


def run_config_test(config: Dict) -> Dict:
    """Run a single configuration test. Returns metrics dict."""
    print(f"\n{'='*70}")
    print(f"Testing: {config['name']}")
    print(f"{'='*70}")

    result = {
        "config_name": config["name"],
        "neurons": config["neurons"],
        "max_out_degree": config["max_out_degree"],
        "approx_synapses": calculate_synapse_count(config["neurons"], config["max_out_degree"]),
        "num_agents": config["num_agents"],
        "max_ticks": config["max_ticks"],
        "results": {},
        "error": None,
    }

    try:
        # Create configuration
        lrsg_cfg = LRSGConfig(
            num_neurons=config["neurons"],
            max_out_degree=config["max_out_degree"],
            h_window=min(4096, config["neurons"] // 4),
            disable_language=True,  # Disable language to focus on core performance
        )

        arena_cfg = ArenaConfig(grid_size=config["grid_size"])
        sim_cfg = SimulationConfig(
            num_agents=config["num_agents"],
            max_ticks=config["max_ticks"],
        )
        full_cfg = Config(lrsg=lrsg_cfg, arena=arena_cfg, simulation=sim_cfg)

        # 1. INITIALIZATION METRICS
        print(f"Initializing {config['num_agents']} agents x {config['neurons']:,} neurons...")
        torch.cuda.empty_cache()
        vram_before = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0

        t0 = time.time()
        sim = Simulation(full_cfg, device=device)
        init_time = time.time() - t0

        vram_after = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        vram_used = vram_after - vram_before

        print(f"  Init time: {init_time:.2f}s")
        print(f"  VRAM used: {vram_used:.2f}GB (total: {vram_after:.2f}GB / {total_vram:.2f}GB)")

        result["results"]["init_time_sec"] = init_time
        result["results"]["vram_used_gb"] = vram_used
        result["results"]["vram_total_gb"] = vram_after

        # 2. SIMULATION METRICS
        print(f"Running {config['max_ticks']} ticks...")

        # Suppress dashboard output during simulation
        import logging
        logging.getLogger().setLevel(logging.ERROR)

        t0 = time.time()
        try:
            metrics = sim.run()
            run_time = time.time() - t0

            avg_tick_ms = (run_time / config["max_ticks"]) * 1000
            ticks_per_sec = 1.0 / (run_time / config["max_ticks"]) if run_time > 0 else 0

            print(f"  Run time: {run_time:.2f}s")
            print(f"  Avg tick: {avg_tick_ms:.1f}ms ({ticks_per_sec:.1f} TPS)")

            result["results"]["run_time_sec"] = run_time
            result["results"]["avg_tick_ms"] = avg_tick_ms
            result["results"]["ticks_per_sec"] = ticks_per_sec

            # 3. STABILITY METRICS
            final_status = sim.status
            result["results"]["agents_alive"] = final_status.get("agents_alive", 0)
            result["results"]["total_agents"] = config["num_agents"]

            if "metrics" in final_status:
                for key, value in final_status["metrics"].items():
                    if isinstance(value, (int, float, bool)):
                        result["results"][f"final_{key}"] = value

            # Check for NaN/Inf in final states
            nan_count = 0
            for agent in sim.agents:
                if torch.isnan(agent.brain.activations).any():
                    nan_count += 1

            result["results"]["agents_with_nan"] = nan_count

            print(f"  Agents alive: {final_status.get('agents_alive', 0)}/{config['num_agents']}")
            print(f"  NaN agents: {nan_count}")

        except Exception as e:
            print(f"  ⚠️  Simulation error: {str(e)}")
            result["results"]["simulation_error"] = str(e)
            raise

        vram_final = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        result["results"]["vram_final_gb"] = vram_final

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        result["error"] = str(e)
        traceback.print_exc()

    return result


def compile_results(results: List[Dict]) -> None:
    """Compile and print results comparison."""
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")

    # Create comparison table
    print(f"{'Config':<12} {'Neurons':>12} {'Synapses':>15} {'Init (s)':>10} {'VRAM (GB)':>10}")
    print("-" * 70)

    for r in results:
        if r["error"] is None:
            neurons = r["neurons"]
            synapses = r["approx_synapses"]
            init_time = r["results"].get("init_time_sec", 0)
            vram = r["results"].get("vram_used_gb", 0)
            print(f"{r['config_name']:<12} {neurons:>12,} {synapses:>15,} {init_time:>10.2f} {vram:>10.2f}")
        else:
            print(f"{r['config_name']:<12} {'ERROR':>12} {r['error']:<40}")

    print()
    print(f"{'Config':<12} {'TPS':>8} {'Agents':>8} {'Survivors':>10} {'NaN':>6}")
    print("-" * 70)

    for r in results:
        if r["error"] is None:
            tps = r["results"].get("ticks_per_sec", 0)
            agents = r["num_agents"]
            survivors = r["results"].get("agents_alive", 0)
            nans = r["results"].get("agents_with_nan", 0)
            print(f"{r['config_name']:<12} {tps:>8.2f} {agents:>8} {survivors:>10} {nans:>6}")
        else:
            print(f"{r['config_name']:<12} {'ERROR':>8}")

    # Save full results
    output_file = "experiment_results/config_comparison.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "device": gpu_name,
            "total_vram_gb": total_vram,
            "results": results,
        }, f, indent=2)

    print(f"\n✅ Full results saved to {output_file}")


if __name__ == "__main__":
    print(f"Cynosure Configuration Comparison Test")
    print(f"Device: {gpu_name}")
    print(f"VRAM: {total_vram:.1f}GB\n")

    results = []
    for config in CONFIGS:
        result = run_config_test(config)
        results.append(result)

    compile_results(results)
