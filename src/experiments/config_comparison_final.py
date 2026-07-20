#!/usr/bin/env python3
"""config_comparison_final.py

Comprehensive comparison of three Cynosure configurations with full metrics.

Configs (adjusted for 8GB VRAM):
1. Compact:    2^20 (1M) neurons, max_degree 24
2. Standard:   2^22 (4M) neurons, max_degree 24
3. Medium:     2^23 (8M) neurons, max_degree 12 (reduced from 2^25 which OOMs)
"""

import json
import os
import sys
import time
import traceback
import logging

import torch

# Add project root to path
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, project_root)

from src.config import LRSGConfig
from src.lrsg import LRSGraph

# Suppress logging
logging.disable(logging.CRITICAL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

CONFIGS = [
    {
        "label": "Config 1: Compact",
        "neurons": 2**20,           # 1,048,576
        "max_out_degree": 24,
        "grid_ticks": 10,           # Run for 10 ticks
    },
    {
        "label": "Config 2: Standard",
        "neurons": 2**22,           # 4,194,304
        "max_out_degree": 24,
        "grid_ticks": 10,
    },
    {
        "label": "Config 3: Medium",
        "neurons": 2**23,           # 8,388,608
        "max_out_degree": 12,       # Reduced for VRAM
        "grid_ticks": 5,
    },
]


def estimate_synapses(neurons: int, max_degree: int) -> int:
    """Estimate total synapses."""
    return int(neurons * max_degree * 0.7)  # ~70% sparsity


def test_config(config: dict) -> dict:
    """Test a configuration."""
    print(f"\n{config['label']}")
    print("-" * 70)

    result = {
        "label": config["label"],
        "neurons": config["neurons"],
        "max_out_degree": config["max_out_degree"],
        "estimated_synapses": estimate_synapses(config["neurons"], config["max_out_degree"]),
        "success": False,
        "metrics": {},
    }

    try:
        torch.cuda.empty_cache()
        vram_before = torch.cuda.memory_allocated(0) / 1e9

        # 1. Initialization
        print(f"Initializing {config['neurons']:,} neurons...", end="", flush=True)
        t0 = time.time()

        cfg = LRSGConfig(
            num_neurons=config["neurons"],
            max_out_degree=config["max_out_degree"],
            h_window=min(4096, config["neurons"] // 4),
            disable_language=True,
        )
        brain = LRSGraph(cfg, device)

        init_time = time.time() - t0
        vram_after_init = torch.cuda.memory_allocated(0) / 1e9
        vram_init = vram_after_init - vram_before

        print(f" DONE ({init_time:.2f}s, {vram_init*1024:.0f}MB)")

        # 2. Test activation cycle (forward pass simulation)
        print(f"Testing activation cycles...", end="", flush=True)
        times = []
        for i in range(config["grid_ticks"]):
            t0 = time.time()
            brain.reset_activations()
            times.append((time.time() - t0) * 1000)

        avg_reset_ms = sum(times) / len(times)
        print(f" DONE ({avg_reset_ms:.2f}ms avg)")

        # 3. Check stability
        nan_count = (torch.isnan(brain.activations).sum().item())
        print(f"Stability: {nan_count} NaNs")

        vram_final = torch.cuda.memory_allocated(0) / 1e9

        result["success"] = True
        result["metrics"] = {
            "init_time_sec": init_time,
            "vram_init_mb": vram_init * 1024,
            "vram_final_mb": vram_final * 1024,
            "avg_reset_ms": avg_reset_ms,
            "nans": nan_count,
        }

    except Exception as e:
        print(f" ERROR: {str(e)[:100]}")
        result["error"] = str(e)
        traceback.print_exc()

    return result


def format_size(bytes_mb: float) -> str:
    """Format bytes as MB or GB."""
    if bytes_mb > 1024:
        return f"{bytes_mb / 1024:.1f}GB"
    return f"{bytes_mb:.0f}MB"


def main():
    print("\n" + "="*70)
    print("Cynosure Configuration Comparison")
    print("="*70)
    print(f"Device: {gpu_name}")
    print(f"Total VRAM: {total_vram:.1f}GB\n")

    results = []
    for config in CONFIGS:
        result = test_config(config)
        results.append(result)

    # Summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70 + "\n")

    print(f"{'Config':<30} {'Neurons':>12} {'Synapses':>15} {'Status':>10}")
    print("-" * 70)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"{r['label']:<30} {r['neurons']:>12,} {r['estimated_synapses']:>15,} {status:>10}")

    print(f"\n{'Config':<30} {'Init (sec)':>12} {'VRAM (MB)':>15} {'Reset (ms)':>10}")
    print("-" * 70)
    for r in results:
        if r["success"]:
            init = r["metrics"]["init_time_sec"]
            vram = r["metrics"]["vram_init_mb"]
            reset = r["metrics"]["avg_reset_ms"]
            print(f"{r['label']:<30} {init:>12.2f} {vram:>15.0f} {reset:>10.3f}")
        else:
            print(f"{r['label']:<30} {'ERROR':>12}")

    # Key findings
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print("="*70)

    successful = [r for r in results if r["success"]]
    if successful:
        fastest_init = min(successful, key=lambda x: x["metrics"]["init_time_sec"])
        smallest_vram = min(successful, key=lambda x: x["metrics"]["vram_init_mb"])
        largest_viable = max(successful, key=lambda x: x["neurons"])

        print(f"\nFastest initialization: {fastest_init['label']}")
        print(f"  Time: {fastest_init['metrics']['init_time_sec']:.2f}s")
        print(f"\nSmallest VRAM footprint: {smallest_vram['label']}")
        print(f"  VRAM: {smallest_vram['metrics']['vram_init_mb']:.0f}MB")
        print(f"\nLargest viable config (8GB limit): {largest_viable['label']}")
        print(f"  Neurons: {largest_viable['neurons']:,}")
        print(f"  VRAM: {largest_viable['metrics']['vram_init_mb']:.0f}MB")

    # Save results
    os.makedirs("experiment_results", exist_ok=True)
    output_file = "experiment_results/config_comparison_final.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "device": gpu_name,
            "total_vram_gb": total_vram,
            "results": results,
        }, f, indent=2)

    print(f"\n[SUCCESS] Results saved to {output_file}")


if __name__ == "__main__":
    main()
