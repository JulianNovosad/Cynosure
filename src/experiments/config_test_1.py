"""Config Test 1: 2^22 neurons (4.19M), 2^32 synapses (sparse, ~1024 degree)"""
import os
import sys
import time
import json
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
from src.config import Config
from src.simulation.loop import Simulation

# Configuration 1: Large neuron count, high connectivity
cfg = Config()
cfg.lrsg.num_neurons = 2**22  # 4,194,304 neurons
cfg.lrsg.max_out_degree = 1024  # ~1024 connections per neuron = 4.3B synapses
cfg.simulation.num_agents = 3
cfg.simulation.max_ticks = 500
cfg.simulation.use_llm_announcer = False  # Disable LLM to speed up runs

print("=" * 70)
print("CONFIG TEST 1: 2^22 Neurons (4.19M) × ~1024 Degree (4.3B Synapses)")
print(f"  Agents:       {cfg.simulation.num_agents}")
print(f"  Neurons:      {cfg.lrsg.num_neurons:,} (2^22)")
print(f"  Degree:       {cfg.lrsg.max_out_degree}")
print(f"  Est. Edges:   ~{(cfg.lrsg.num_neurons - cfg.lrsg.num_output_neurons) * cfg.lrsg.max_out_degree:,.0e}")
print(f"  Ticks:        {cfg.simulation.max_ticks:,}")
print(f"  Device:       cuda")
print("=" * 70)

try:
    t_start = time.time()
    sim = Simulation(cfg, torch.device("cuda"))
    t_init = time.time()
    vram_init = torch.cuda.memory_allocated() / 1e9
    print(f"\nInit: {t_init - t_start:.1f}s, VRAM: {vram_init:.2f}GB")

    metrics = sim.run()
    t_end = time.time()
    elapsed = t_end - t_init
    avg_ms = (elapsed / max(1, sim.tick_count)) * 1000
    vram_peak = torch.cuda.max_memory_allocated() / 1e9
    vram_final = torch.cuda.memory_allocated() / 1e9

    print(f"\n{'=' * 70}")
    print(f"COMPLETE - {sim.tick_count:,} ticks in {elapsed:.0f}s ({avg_ms:.1f}ms/tick)")
    print(f"  VRAM Peak: {vram_peak:.2f}GB | Final: {vram_final:.2f}GB")
    print(f"  Status: {sim.status}")

    # Aggregate metrics
    result = {
        "config_id": 1,
        "num_neurons": cfg.lrsg.num_neurons,
        "max_out_degree": cfg.lrsg.max_out_degree,
        "num_agents": cfg.simulation.num_agents,
        "ticks_completed": sim.tick_count,
        "elapsed_seconds": elapsed,
        "avg_ms_per_tick": avg_ms,
        "vram_peak_gb": vram_peak,
        "vram_final_gb": vram_final,
        "metrics": {}
    }

    for key, values in metrics.items():
        if values:
            result["metrics"][key] = {
                "min": float(min(values)),
                "max": float(max(values)),
                "mean": float(sum(values) / len(values)),
                "last": float(values[-1])
            }

    # Save result
    os.makedirs("experiment_results", exist_ok=True)
    with open("experiment_results/config_test_1.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n✓ Results saved to experiment_results/config_test_1.json")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
