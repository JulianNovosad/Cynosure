#!/usr/bin/env python3
"""
user_specs_test.py

Test the user's THREE EXACT CONFIGURATIONS as specified:
1. 4M neurons (2^22) + 4B synapses (2^32)
2. 32M neurons (2^25) + 320M synapses (2^29)
3. 1M neurons (2^20) + 4B synapses (2^32)

STEP 1: Define metrics
STEP 2: Run experiments on all three configs
STEP 3: Compile and report results (including failures)
"""

import json
import os
import sys
import time
import logging

import torch

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
    "initialization": ["time_seconds", "vram_allocated_mb", "vram_percentage"],
    "stability": ["nan_count", "inf_count", "graph_integrity"],
    "performance": ["activation_reset_ms"],
    "scaling": ["vram_per_1m_neurons", "init_time_per_1m_neurons"],
    "feasibility": ["config_viable", "error_description"],
}

print("\nMetrics to measure per configuration:")
for category, metrics in METRICS.items():
    print(f"  {category.upper()}: {', '.join(metrics)}")

# USER'S EXACT SPECIFICATIONS
print("\n" + "="*90)
print("STEP 2: RUN EXPERIMENTS")
print("="*90)

USER_CONFIGS = [
    {
        "id": 1,
        "name": "Config 1",
        "neurons": 2**22,
        "neurons_spec": "4M -> 2^22",
        "requested_synapses": 2**32,
        "synapses_spec": "4B -> 2^32",
    },
    {
        "id": 2,
        "name": "Config 2",
        "neurons": 2**25,
        "neurons_spec": "32M -> 2^25",
        "requested_synapses": 2**29,
        "synapses_spec": "320M -> 2^29",
    },
    {
        "id": 3,
        "name": "Config 3",
        "neurons": 2**20,
        "neurons_spec": "1M -> 2^20",
        "requested_synapses": 2**32,
        "synapses_spec": "4B -> 2^32",
    },
]

def calc_degree(neurons, synapses):
    return int(synapses / (neurons * 0.7))

def run_experiment(config):
    """Run experiment for one configuration."""
    print(f"\n{config['name']}")
    print(f"  Spec: {config['neurons_spec']} neurons, {config['synapses_spec']} synapses")
    print(f"  Actual: {config['neurons']:,} neurons, {config['requested_synapses']:,} synapses")

    result = {
        "id": config["id"],
        "name": config["name"],
        "neurons": config["neurons"],
        "synapses": config["requested_synapses"],
        "status": "NOT_RUN",
        "metrics": {},
        "error": None,
    }

    required_degree = calc_degree(config["neurons"], config["requested_synapses"])
    print(f"  Required degree: {required_degree:,}")
    result["required_degree"] = required_degree

    try:
        torch.cuda.empty_cache()
        vram_before = torch.cuda.memory_allocated(0) / 1e9

        t0 = time.time()

        cfg = LRSGConfig(
            num_neurons=config["neurons"],
            max_out_degree=min(required_degree, 24),
            h_window=min(4096, config["neurons"] // 4),
            disable_language=True,
        )

        brain = LRSGraph(cfg, device)
        init_time = time.time() - t0

        vram_after = torch.cuda.memory_allocated(0) / 1e9
        vram_used = vram_after - vram_before

        nan_count = torch.isnan(brain.activations).sum().item()
        inf_count = torch.isinf(brain.activations).sum().item()

        print(f"  [OK] Init: {init_time:.2f}s, VRAM: {vram_used*1024:.0f}MB, NaNs: {nan_count}")

        result["status"] = "SUCCESS"
        result["metrics"] = {
            "init_time_sec": init_time,
            "vram_mb": vram_used * 1024,
            "vram_pct": (vram_used / total_vram) * 100,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "mb_per_1m": (vram_used * 1024) / (config["neurons"] / 1e6),
            "ms_per_1m": (init_time * 1000) / (config["neurons"] / 1e6),
        }

    except torch.cuda.OutOfMemoryError:
        print(f"  [FAILED] CUDA Out of Memory")
        result["status"] = "FAILED_OOM"
        result["error"] = "CUDA out of memory"

    except Exception as e:
        msg = str(e)[:100]
        print(f"  [FAILED] {msg}")
        result["status"] = "FAILED"
        result["error"] = msg

    return result

print(f"\nDevice: {gpu_name} | Total VRAM: {total_vram:.1f}GB\n")

results = []
for config in USER_CONFIGS:
    result = run_experiment(config)
    results.append(result)

# STEP 3: COMPILE RESULTS
print("\n" + "="*90)
print("STEP 3: COMPILE RESULTS")
print("="*90)

print(f"\n{'Config':<12} {'Neurons':>12} {'Synapses':>15} {'Status':<20}")
print("-"*90)
for r in results:
    status = r['status']
    print(f"{r['name']:<12} {r['neurons']:>12,} {r['synapses']:>15,} {status:<20}")

print(f"\n{'Config':<12} {'Init (sec)':>12} {'VRAM (MB)':>12} {'NaNs':>6} {'Status':<20}")
print("-"*90)
for r in results:
    if r["status"] == "SUCCESS":
        print(f"{r['name']:<12} {r['metrics']['init_time_sec']:>12.2f} {r['metrics']['vram_mb']:>12.0f} {r['metrics']['nan_count']:>6} {r['status']:<20}")
    else:
        print(f"{r['name']:<12} {'N/A':>12} {'N/A':>12} {'N/A':>6} {r['status']:<20}")

# Save results
os.makedirs("experiment_results", exist_ok=True)
output_file = "experiment_results/user_exact_specs.json"

with open(output_file, "w") as f:
    json.dump({
        "timestamp": time.time(),
        "device": gpu_name,
        "vram_gb": total_vram,
        "metrics": METRICS,
        "results": results,
    }, f, indent=2)

print(f"\n[DONE] Results saved to: {output_file}")

successful = [r for r in results if r["status"] == "SUCCESS"]
print(f"\nSummary: {len(successful)}/{len(results)} configs passed")
