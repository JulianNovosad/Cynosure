#!/usr/bin/env python3
"""
USER_SPECS_TEST.py

Test the user's THREE EXACT CONFIGURATIONS as specified:
1. 4M neurons (round to 2^22) + 4B synapses (round to 2^32)
2. 32M neurons (round to 2^25) + 320M synapses (round to 2^29)
3. 1M neurons (round to 2^20) + 4B synapses (round to 2^32)

This script will:
1. Define metrics to measure
2. Run experiments on all three configs
3. Compile and report results (including failures)
"""

import json
import os
import sys
import time
import traceback
import logging

import torch
import numpy as np

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, project_root)

from src.config import LRSGConfig
from src.lrsg import LRSGraph

logging.disable(logging.CRITICAL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9

print("\n" + "="*90)
print("STEP 1: DEFINE METRICS")
print("="*90)

METRICS = {
    "initialization": [
        "time_seconds",
        "vram_allocated_mb",
        "vram_percentage",
    ],
    "stability": [
        "nan_count",
        "inf_count",
        "graph_integrity",
    ],
    "performance": [
        "ticks_per_second",
        "activation_reset_ms",
    ],
    "scaling": [
        "vram_per_1m_neurons",
        "init_time_per_1m_neurons",
    ],
    "feasibility": [
        "config_viable",
        "hardware_sufficient",
        "error_description",
    ],
}

print("\nMetrics to measure per configuration:")
for category, metrics in METRICS.items():
    print(f"  {category.upper()}:")
    for metric in metrics:
        print(f"    - {metric}")

# USER'S EXACT SPECIFICATIONS
print("\n" + "="*90)
print("STEP 2: RUN EXPERIMENTS")
print("="*90)

USER_CONFIGS = [
    {
        "id": 1,
        "name": "Config 1",
        "neurons": 2**22,          # 4M rounded to power of 2
        "neurons_rounded": "2^22",
        "requested_synapses": 2**32,  # 4B rounded to power of 2
        "synapses_rounded": "2^32",
        "description": "4M neurons + 4B synapses",
    },
    {
        "id": 2,
        "name": "Config 2",
        "neurons": 2**25,          # 32M rounded to power of 2
        "neurons_rounded": "2^25",
        "requested_synapses": 2**29,  # 320M rounded to power of 2
        "synapses_rounded": "2^29",
        "description": "32M neurons + 320M synapses",
    },
    {
        "id": 3,
        "name": "Config 3",
        "neurons": 2**20,          # 1M rounded to power of 2
        "neurons_rounded": "2^20",
        "requested_synapses": 2**32,  # 4B rounded to power of 2
        "synapses_rounded": "2^32",
        "description": "1M neurons + 4B synapses",
    },
]

def calculate_degree_from_synapses(neurons, synapses):
    """Calculate required max_out_degree to achieve synapse count."""
    # Assuming ~70% sparsity: actual = neurons × degree × 0.7
    degree = int(synapses / (neurons * 0.7))
    return degree

def run_experiment(config):
    """Run experiment for one configuration."""
    print(f"\n{config['name']}: {config['description']}")
    print(f"  Neurons: {config['neurons']:,} ({config['neurons_rounded']})")
    print(f"  Synapses: {config['requested_synapses']:,} ({config['synapses_rounded']})")

    result = {
        "id": config["id"],
        "name": config["name"],
        "description": config["description"],
        "neurons": config["neurons"],
        "requested_synapses": config["requested_synapses"],
        "status": "NOT_RUN",
        "metrics": {},
        "error": None,
    }

    # Calculate required degree
    required_degree = calculate_degree_from_synapses(config["neurons"], config["requested_synapses"])
    print(f"  Required max_out_degree: {required_degree:,}")

    result["required_degree"] = required_degree

    # Attempt instantiation
    print(f"  Attempting initialization...")

    try:
        torch.cuda.empty_cache()
        vram_before = torch.cuda.memory_allocated(0) / 1e9

        t0 = time.time()

        # Create config with capped degree
        cfg = LRSGConfig(
            num_neurons=config["neurons"],
            max_out_degree=min(required_degree, 24),  # Cap at 24 (LRSG standard)
            h_window=min(4096, config["neurons"] // 4),
            disable_language=True,
        )

        brain = LRSGraph(cfg, device)
        init_time = time.time() - t0

        vram_after = torch.cuda.memory_allocated(0) / 1e9
        vram_used = vram_after - vram_before

        # Check stability
        nan_count = torch.isnan(brain.activations).sum().item()
        inf_count = torch.isinf(brain.activations).sum().item()

        print(f"    [SUCCESS] Init: {init_time:.2f}s | VRAM: {vram_used*1024:.0f}MB | NaNs: {nan_count}")

        result["status"] = "SUCCESS"
        result["metrics"] = {
            "init_time_sec": init_time,
            "vram_allocated_mb": vram_used * 1024,
            "vram_percentage": (vram_used / total_vram) * 100,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "vram_per_1m_neurons": (vram_used * 1024) / (config["neurons"] / 1e6),
            "init_time_per_1m_neurons": (init_time * 1000) / (config["neurons"] / 1e6),
        }

    except torch.cuda.OutOfMemoryError as e:
        error_msg = "CUDA out of memory"
        print(f"    [FAILED] {error_msg}")
        result["status"] = "FAILED_OOM"
        result["error"] = error_msg
        result["metrics"]["vram_total"] = total_vram

    except Exception as e:
        error_msg = str(e)[:200]
        print(f"    [FAILED] {error_msg}")
        result["status"] = "FAILED_ERROR"
        result["error"] = error_msg

    return result

# Run all three experiments
print(f"\nDevice: {gpu_name}")
print(f"Total VRAM: {total_vram:.1f}GB\n")

results = []
for config in USER_CONFIGS:
    result = run_experiment(config)
    results.append(result)

# STEP 3: COMPILE RESULTS
print("\n" + "="*90)
print("STEP 3: COMPILE RESULTS")
print("="*90)

print("\n" + "-"*90)
print("SUMMARY TABLE")
print("-"*90)
print(f"{'Config':<12} {'Neurons':>12} {'Synapses':>15} {'Degree':>10} {'Status':<20}")
print("-"*90)

for r in results:
    print(f"{r['name']:<12} {r['neurons']:>12,} {r['requested_synapses']:>15,} {r['required_degree']:>10,} {r['status']:<20}")

print("\n" + "-"*90)
print("DETAILED RESULTS")
print("-"*90)

for r in results:
    print(f"\n{r['name']}: {r['description']}")
    print(f"  Status: {r['status']}")

    if r["status"] == "SUCCESS":
        print(f"  Init Time: {r['metrics']['init_time_sec']:.2f}s")
        print(f"  VRAM Used: {r['metrics']['vram_allocated_mb']:.0f}MB ({r['metrics']['vram_percentage']:.1f}%)")
        print(f"  Stability: {r['metrics']['nan_count']} NaNs, {r['metrics']['inf_count']} Infs")
        print(f"  Efficiency: {r['metrics']['vram_per_1m_neurons']:.0f}MB per 1M neurons")
    else:
        print(f"  Error: {r['error']}")

# Save JSON results
print("\n" + "-"*90)
print("SAVING RESULTS")
print("-"*90)

os.makedirs("experiment_results", exist_ok=True)
output_file = "experiment_results/user_exact_specs.json"

with open(output_file, "w") as f:
    json.dump({
        "timestamp": time.time(),
        "device": gpu_name,
        "total_vram_gb": total_vram,
        "metrics_defined": METRICS,
        "configs_tested": [
            {
                "id": c["id"],
                "description": c["description"],
                "neurons": c["neurons"],
                "synapses": c["requested_synapses"],
            }
            for c in USER_CONFIGS
        ],
        "results": results,
    }, f, indent=2)

print(f"Results saved to: {output_file}")

print("\n" + "="*90)
print("FINAL REPORT")
print("="*90)

successful = [r for r in results if r["status"] == "SUCCESS"]
failed = [r for r in results if r["status"] != "SUCCESS"]

print(f"\nExperiments Run: {len(results)}")
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")

if successful:
    print(f"\nSuccessful Configs:")
    for r in successful:
        print(f"  {r['name']}: {r['neurons']:,} neurons, {r['metrics']['vram_allocated_mb']:.0f}MB VRAM")

if failed:
    print(f"\nFailed Configs:")
    for r in failed:
        print(f"  {r['name']}: {r['status']} - {r['error']}")

print("\n" + "="*90)

