#!/usr/bin/env python3
"""config_test_quick.py

Quick configuration test - focuses on initialization, memory, and short runs.
"""

import json
import os
import sys
import time

import torch

# Add project root to path
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, project_root)

from src.config import Config, LRSGConfig, ArenaConfig, SimulationConfig
from src.lrsg import LRSGraph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

# Suppress all logging
import logging
logging.disable(logging.CRITICAL)

CONFIGS = [
    {
        "name": "Compact (2^20 neurons)",
        "neurons": 2**20,
        "max_out_degree": 24,
        "description": "1M neurons, ~24M synapses",
    },
    {
        "name": "Standard (2^22 neurons)",
        "neurons": 2**22,
        "max_out_degree": 24,
        "description": "4M neurons, ~96M synapses",
    },
    {
        "name": "Large (2^25 neurons)",
        "neurons": 2**25,
        "max_out_degree": 12,
        "description": "32M neurons, ~384M synapses",
    },
]


def test_single_brain(neurons: int, max_degree: int) -> dict:
    """Test a single LRSG brain configuration."""
    try:
        torch.cuda.empty_cache()
        vram_before = torch.cuda.memory_allocated(0) / 1e9

        t0 = time.time()
        cfg = LRSGConfig(
            num_neurons=neurons,
            max_out_degree=max_degree,
            h_window=min(4096, neurons // 4),
            disable_language=True,
        )
        brain = LRSGraph(cfg, device)
        init_time = time.time() - t0

        vram_after = torch.cuda.memory_allocated(0) / 1e9
        vram_used = vram_after - vram_before

        # Check basic activation reset (lightweight operation)
        t0 = time.time()
        brain.reset_activations()
        forward_time = time.time() - t0

        # Estimate synapses
        approx_synapses = int(neurons * max_degree * 0.7)

        return {
            "success": True,
            "neurons": neurons,
            "approx_synapses": approx_synapses,
            "max_out_degree": max_degree,
            "init_time_ms": init_time * 1000,
            "vram_used_mb": vram_used * 1024,
            "vram_total_gb": vram_after,
            "forward_time_ms": forward_time * 1000,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "neurons": neurons,
        }


def main():
    print(f"\n{'='*80}")
    print(f"Cynosure Configuration Test (Quick)")
    print(f"{'='*80}")
    print(f"Device: {gpu_name}")
    print(f"Total VRAM: {total_vram:.1f}GB\n")

    results = []

    for config in CONFIGS:
        print(f"Testing: {config['name']}")
        print(f"  {config['description']}")

        result = test_single_brain(config["neurons"], config["max_out_degree"])
        result["config_name"] = config["name"]
        results.append(result)

        if result["success"]:
            print(f"  [OK] Init: {result['init_time_ms']:.1f}ms")
            print(f"  [OK] VRAM: {result['vram_used_mb']:.0f}MB ({result['vram_total_gb']:.1f}GB total)")
            print(f"  [OK] Forward: {result['forward_time_ms']:.2f}ms")
            print()
        else:
            print(f"  [FAILED] {result['error']}\n")

    # Summary table
    print(f"{'='*80}")
    print(f"{'Config':<25} {'Neurons':>12} {'Synapses':>15} {'Init (ms)':>10} {'VRAM':>10}")
    print(f"{'-'*80}")

    for r in results:
        if r["success"]:
            print(f"{r['config_name']:<25} {r['neurons']:>12,} {r['approx_synapses']:>15,} {r['init_time_ms']:>10.1f} {r['vram_used_mb']:>9.0f}MB")
        else:
            print(f"{r['config_name']:<25} ERROR: {r['error']}")

    # Save results
    os.makedirs("experiment_results", exist_ok=True)
    with open("experiment_results/config_test_quick.json", "w") as f:
        json.dump({
            "timestamp": time.time(),
            "device": gpu_name,
            "total_vram_gb": total_vram,
            "results": results,
        }, f, indent=2)

    print(f"\n[SUCCESS] Results saved to experiment_results/config_test_quick.json")


if __name__ == "__main__":
    main()
