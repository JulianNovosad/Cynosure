"""5000-tick linguistic curriculum — test scale (10K neurons, 5 agents)."""
import os
import sys
import time
import torch
sys.stdout.reconfigure(encoding='utf-8')
from src.config import Config, LRSGConfig, ArenaConfig, AgentConfig, SimulationConfig
from src.simulation.loop import Simulation

cfg = Config(
    lrsg=LRSGConfig(num_neurons=10_000, max_out_degree=4, num_input_neurons=100, num_output_neurons=527),
    arena=ArenaConfig(grid_size=(64, 64, 64)),
    simulation=SimulationConfig(num_agents=5, max_ticks=5_000, use_llm_announcer=False),
)

for f in ["telemetry.jsonl", "agent_whispers.jsonl", "conflicts.jsonl"]:
    if os.path.exists(f):
        os.remove(f)

print("=" * 60)
print("CYNOSURE — Linguistic Curriculum (Test Scale, LLM-free)")
print(f"  Agents:   {cfg.simulation.num_agents}")
print(f"  Neurons/agent:  {cfg.lrsg.num_neurons:,}")
print(f"  Ticks:    {cfg.simulation.max_ticks:,}")
print(f"  Grid:     {cfg.arena.grid_size}")
print("=" * 60)

t_start = time.time()
sim = Simulation(cfg, torch.device("cuda"))
t_init = time.time()
vram_init = torch.cuda.memory_allocated() / 1e9
print(f"\nInit: {t_init - t_start:.1f}s, VRAM: {vram_init:.2f}GB")
print("Starting 5000-tick run...")

metrics = sim.run()
t_end = time.time()
elapsed = t_end - t_init
avg_ms = (elapsed / max(1, sim.tick_count)) * 1000
vram_final = torch.cuda.memory_allocated() / 1e9

print(f"\n{'=' * 60}")
print(f"COMPLETE - {sim.tick_count:,} ticks in {elapsed:.0f}s ({avg_ms:.1f}ms/tick avg)")
print(f"  VRAM: {vram_init:.2f}GB -> {vram_final:.2f}GB")
print(f"  Final status: {sim.status}")

for key, values in metrics.items():
    if values:
        print(f"  {key}: min={min(values):.4f}  max={max(values):.4f}  mean={sum(values)/len(values):.4f}  last={values[-1]}")

print(f"\nFiles written:")
for f in ["telemetry.jsonl", "conflicts.jsonl", "agent_whispers.jsonl", "security_audit.log"]:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f"  {f}: {size/1024:.1f}KB")