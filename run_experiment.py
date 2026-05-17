"""Exp 1: Baseline Calibration — 10K-tick full-scale run."""
import os
import sys
import time
import torch
sys.stdout.reconfigure(encoding='utf-8')
from src.config import Config
from src.simulation.loop import Simulation

cfg = Config()
cfg.simulation.use_llm_announcer = True
cfg.simulation.max_ticks = 100_000

print("=" * 60)
print("CYNOSURE — Exp 1: Baseline Calibration")
print(f"  Agents:   {cfg.simulation.num_agents}")
print(f"  Neurons:  {cfg.lrsg.num_neurons:,}")
print(f"  Edges:    ~{(cfg.lrsg.num_neurons - cfg.lrsg.num_output_neurons) * cfg.lrsg.max_out_degree:,}")
print(f"  Model:    {cfg.simulation.announcer_model}")
print(f"  Ticks:    {cfg.simulation.max_ticks:,}")
print(f"  Device:   cuda")
print(f"  VRAM budget: 8GB")
print("=" * 60)

t_start = time.time()
sim = Simulation(cfg, torch.device("cuda"))
t_init = time.time()
vram_init = torch.cuda.memory_allocated() / 1e9
print(f"\nInit: {t_init - t_start:.1f}s, VRAM: {vram_init:.2f}GB")

metrics = sim.run()
t_end = time.time()
elapsed = t_end - t_init
avg_ms = (elapsed / max(1, sim.tick_count)) * 1000
vram_final = torch.cuda.memory_allocated() / 1e9

print(f"\n{'=' * 60}")
print(f"COMPLETE - {sim.tick_count:,} ticks in {elapsed:.0f}s ({avg_ms:.1f}ms/tick)")
print(f"  VRAM: {vram_init:.2f}GB -> {vram_final:.2f}GB")
print(f"  Final status: {sim.status}")

# Summary statistics
for key, values in metrics.items():
    if values:
        print(f"  {key}: min={min(values):.4f}  max={max(values):.4f}  mean={sum(values)/len(values):.4f}  last={values[-1]}")

print(f"\nFiles written:")
for f in ["telemetry.jsonl", "conflicts.jsonl", "security_audit.log"]:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f"  {f}: {size/1024:.1f}KB")
