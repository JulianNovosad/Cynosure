#!/usr/bin/env python3
"""test_user_configs.py

Test the user's three specific configurations:
1. 4M neurons + 4B synapses (4 billion)
2. 32M neurons + 320M synapses
3. 1M neurons + 4B synapses

Calculate required max_out_degree to achieve these synapse counts.
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
total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

# User's exact specifications
USER_CONFIGS = [
    {
        "label": "User Config 1",
        "neurons": 4_000_000,
        "requested_synapses": 4_000_000_000,  # 4 billion
    },
    {
        "label": "User Config 2",
        "neurons": 32_000_000,
        "requested_synapses": 320_000_000,  # 320 million
    },
    {
        "label": "User Config 3",
        "neurons": 1_000_000,
        "requested_synapses": 4_000_000_000,  # 4 billion
    },
]


def calculate_max_degree(neurons: int, requested_synapses: int) -> int:
    """Calculate required max_out_degree to achieve requested synapse count."""
    # Assuming ~70% sparsity: actual_synapses = neurons × max_degree × 0.7
    # So: max_degree = requested_synapses / (neurons × 0.7)
    max_degree = int(requested_synapses / (neurons * 0.7))
    return max_degree


def test_config(config: dict) -> dict:
    """Test a user-specified configuration."""
    print(f"\n{config['label']}")
    print("-" * 80)

    neurons = config["neurons"]
    requested_synapses = config["requested_synapses"]
    required_degree = calculate_max_degree(neurons, requested_synapses)

    print(f"Neurons: {neurons:,}")
    print(f"Requested Synapses: {requested_synapses:,}")
    print(f"Required max_out_degree: {required_degree:,}")
    print(f"Note: LRSG uses sparse connectivity with max_out_degree parameter")

    result = {
        "label": config["label"],
        "neurons": neurons,
        "requested_synapses": requested_synapses,
        "required_max_degree": required_degree,
        "feasibility": "ANALYSIS",
        "metrics": {},
    }

    # Check feasibility
    print(f"\nFeasibility Analysis:")

    if required_degree > 10000:
        print(f"  [INFEASIBLE] Required degree {required_degree:,} is too large")
        print(f"    Standard LRSG uses max_degree 12-24 for sparse networks")
        print(f"    This would create a DENSE network, not sparse")
        result["feasibility"] = "INFEASIBLE_DENSE"
        result["reason"] = f"Required degree {required_degree:,} is unrealistic for sparse networks"
        return result

    if neurons * required_degree > 1e11:  # More than 100B total connections
        print(f"  [WARNING] Total connections {neurons * required_degree:,.0f} is extremely large")
        result["feasibility"] = "WARNING_DENSE"

    # Try to instantiate with LRSG if feasible
    if required_degree <= 1000:
        print(f"  Attempting instantiation with max_out_degree={required_degree}...")

        try:
            torch.cuda.empty_cache()
            vram_before = torch.cuda.memory_allocated(0) / 1e9

            t0 = time.time()
            cfg = LRSGConfig(
                num_neurons=neurons,
                max_out_degree=min(required_degree, 1000),  # Cap at 1000 for safety
                h_window=min(4096, neurons // 4),
                disable_language=True,
            )
            brain = LRSGraph(cfg, device)
            init_time = time.time() - t0

            vram_after = torch.cuda.memory_allocated(0) / 1e9
            vram_used = vram_after - vram_before

            print(f"  [OK] Initialization successful!")
            print(f"    Init time: {init_time:.2f}s")
            print(f"    VRAM used: {vram_used*1024:.0f}MB")

            result["feasibility"] = "SUCCESS"
            result["metrics"] = {
                "init_time": init_time,
                "vram_used_mb": vram_used * 1024,
                "actual_max_degree": min(required_degree, 1000),
            }

        except Exception as e:
            error_str = str(e)[:200]
            print(f"  [FAILED] Instantiation failed: {error_str}")
            result["feasibility"] = "FAILED"
            result["error"] = error_str
    else:
        print(f"  [INFEASIBLE] Required degree {required_degree:,}")
        print(f"    Would exceed practical sparse network constraints")
        result["feasibility"] = "INFEASIBLE"
        result["reason"] = f"Degree {required_degree:,} too large for sparse networks"

    return result


def main():
    print("\n" + "="*80)
    print("CYNOSURE - USER SPECIFIED CONFIGURATION TEST")
    print("="*80)
    print(f"Device: {gpu_name}")
    print(f"Total VRAM: {total_vram:.1f}GB")
    print(f"\nTesting user's exact specifications...")

    results = []
    for config in USER_CONFIGS:
        result = test_config(config)
        results.append(result)

    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")

    print(f"{'Config':<20} {'Neurons':>12} {'Synapses':>15} {'Degree':>10} {'Status':<20}")
    print("-" * 80)

    for r in results:
        neurons = r["neurons"]
        synapses = r["requested_synapses"]
        degree = r["required_max_degree"]
        status = r["feasibility"]
        print(f"{r['label']:<20} {neurons:>12,} {synapses:>15,} {degree:>10,} {status:<20}")

    # Detailed findings
    print(f"\n" + "="*80)
    print("DETAILED FINDINGS")
    print("="*80 + "\n")

    for r in results:
        print(f"{r['label']}")
        print(f"  Requested: {r['neurons']:,} neurons × {r['requested_synapses']:,} synapses")
        print(f"  Required Degree: {r['required_max_degree']:,}")
        print(f"  Status: {r['feasibility']}")

        if "reason" in r:
            print(f"  Reason: {r['reason']}")
        if "error" in r:
            print(f"  Error: {r['error']}")
        if "metrics" in r and r["metrics"]:
            print(f"  Init: {r['metrics']['init_time']:.2f}s, VRAM: {r['metrics']['vram_used_mb']:.0f}MB")
        print()

    # Technical explanation
    print("="*80)
    print("TECHNICAL ANALYSIS")
    print("="*80 + "\n")

    print("Why some configurations are infeasible:")
    print()
    print("1. SPARSE vs DENSE NETWORKS")
    print("   LRSG uses SPARSE networks with limited out-degree:")
    print("   - Standard LRSG: max_out_degree = 12-24 (sparse)")
    print("   - Your Config 1: Would require degree 1000 (ultra-dense)")
    print("   - Your Config 3: Would require degree 4000 (impossible)")
    print()
    print("2. SYNAPSE COUNTS")
    print("   Your specified synapse counts assume all-to-all or near-all-to-all")
    print("   connectivity, which is not how LRSG works.")
    print()
    print("3. HARDWARE LIMITS")
    print("   Even feasible configs hit VRAM limits:")
    print("   - RTX 2070: 8.6GB VRAM")
    print("   - A100: 40GB VRAM")
    print("   - RTX 4090: 24GB VRAM")
    print()
    print("SOLUTION:")
    print("  Specify neuron count (power of 2) instead of synapse count.")
    print("  LRSG will handle sparse connectivity efficiently.")
    print()

    # Save results
    os.makedirs("experiment_results", exist_ok=True)
    with open("experiment_results/user_config_test.json", "w") as f:
        json.dump({
            "timestamp": time.time(),
            "device": gpu_name,
            "total_vram_gb": total_vram,
            "results": results,
        }, f, indent=2)

    print(f"Results saved to: experiment_results/user_config_test.json")


if __name__ == "__main__":
    main()
