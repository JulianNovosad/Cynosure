"""Episodic Memory Compressor Experiment — 2000-tick run.

Tests:
1. Do agents with similar recent experiences produce similar summary vectors?
2. Do they alter behavior after receiving another agent's summary?
"""
import os
import sys
import time
import torch
import json
import math
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
from src.config import Config, LRSGConfig, ArenaConfig, AgentConfig, SimulationConfig, EpisodicCompressorConfig
from src.simulation.loop import Simulation

for f in ["episodic_results.jsonl", "security_audit.log", "telemetry.jsonl", "conflicts.jsonl", "transitions.jsonl", "agent_whispers.jsonl"]:
    if os.path.exists(f):
        os.remove(f)

cfg = Config(
    lrsg=LRSGConfig(
        num_neurons=262_144,
        max_out_degree=12,
        context_size=256,
        whisper_vocab_size=512,
        whisper_latent_dim=512,
        probe_base_reward=0.3,
        whisper_base_reward=0.2,
        manipulate_base_reward=0.1,
    ),
    arena=ArenaConfig(grid_size=(32, 32, 32)),
    agent=AgentConfig(initial_ram=200, vision_range=8, sensory_radius=10.0, memory_decay_ticks=50),
    simulation=SimulationConfig(num_agents=8, max_ticks=2000, use_llm_announcer=False),
    episodic=EpisodicCompressorConfig(
        input_dim=10,
        hidden_dim=50,
        latent_dim=32,
        buffer_max_size=100,
        train_interval=10,
        summary_interval=10,
        context_inject_weight=0.3,
    ),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Episodic Memory Compressor Experiment on {device}")
print(f"Autoencoder: 10D → 50D → 32D → 50D → 10D")
print(f"Agents: {cfg.simulation.num_agents}, Ticks: {cfg.simulation.max_ticks}")
print(f"Summary dim: 10D (position[3] + RAM + safe_zone + proximity + last_action + nearest_ram + nearest_agent + tick)")

start = time.perf_counter()
sim = Simulation(cfg, device=device)
init_time = time.perf_counter() - start
print(f"Init time: {init_time:.1f}s")

tick_times = []
for _ in range(cfg.simulation.max_ticks):
    t0 = time.perf_counter()
    sim.tick()
    t1 = time.perf_counter()
    tick_times.append((t1 - t0) * 1000)

    if sim.tick_count % 200 == 0:
        avg = sum(tick_times[-200:]) / min(200, len(tick_times)) if tick_times else 0
        alive = sum(1 for a in sim.agents if a.alive)
        vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        last_sim = sim._episodic_metrics["summary_cosine_similarities"][-1] if sim._episodic_metrics["summary_cosine_similarities"] else 0
        last_bcr = sim._episodic_metrics["behavioral_changes"][-1] if sim._episodic_metrics["behavioral_changes"] else 0
        print(f"Tick {sim.tick_count}: avg={avg:.1f}ms, alive={alive}, vram={vram:.1f}GB, summary_sim={last_sim:.4f}, behavior_change={last_bcr:.4f}")

    if not any(a.alive for a in sim.agents):
        break

elapsed = time.perf_counter() - start
print(f"\nTotal ticks: {sim.tick_count}, Time: {elapsed:.1f}s, Avg: {sum(tick_times)/len(tick_times):.1f}ms/tick")

print("\n" + "=" * 70)
print("EPISODIC MEMORY COMPRESSOR — ANALYSIS REPORT")
print("=" * 70)

results = []
with open("episodic_results.jsonl", "r") as f:
    for line in f:
        results.append(json.loads(line))

if results:
    sims = [r["avg_summary_similarity"] for r in results]
    bcrs = [r["behavioral_change_rate"] for r in results]

    early = sims[:20] if len(sims) >= 20 else sims
    mid = sims[len(sims)//4:3*len(sims)//4] if len(sims) >= 40 else sims[len(sims)//2:]
    late = sims[-20:] if len(sims) >= 20 else sims

    print("\nQ1: Do agents with similar experiences produce similar summary vectors?")
    print(f"  Early-phase avg similarity: {sum(early)/len(early):.4f}")
    print(f"  Mid-phase avg similarity:   {sum(mid)/len(mid):.4f}")
    print(f"  Late-phase avg similarity:  {sum(late)/len(late):.4f}")
    print(f"  Overall avg similarity:    {sum(sims)/len(sims):.4f}")
    print(f"  Max similarity:             {max(sims):.4f} (tick {results[sims.index(max(sims))]['tick']})")
    print(f"  Min similarity:             {min(sims):.4f} (tick {results[sims.index(min(sims))]['tick']})")

    increasing = sum(1 for i in range(1, len(sims)) if sims[i] > sims[i-1])
    print(f"  Similarity trend: {'increasing' if increasing > len(sims)*0.5 else 'decreasing/stable' if increasing < len(sims)*0.3 else 'fluctuating'} ({increasing}/{len(sims)-1} positive steps)")

    print("\nQ2: Do agents alter behavior after receiving another agent's summary?")
    print(f"  Early-phase avg behavior change rate: {sum(bcrs[:20])/max(1, min(20, len(bcrs))):.4f}")
    print(f"  Mid-phase avg behavior change rate:   {sum(bcrs[len(bcrs)//4:3*len(bcrs)//4])/max(1, len(bcrs)//2):.4f}")
    print(f"  Late-phase avg behavior change rate:  {sum(bcrs[-20:])/max(1, min(20, len(bcrs))):.4f}")
    print(f"  Overall avg behavior change rate:     {sum(bcrs)/max(1, len(bcrs)):.4f}")
    print(f"  Max behavior change:                 {max(bcrs):.4f} (tick {results[bcrs.index(max(bcrs))]['tick']})")

    total_whispers = 0
    with open("telemetry.jsonl", "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("action_name") == "WHISPER":
                    total_whispers += 1
            except:
                pass
    print(f"\n  Total whispers sent:           {total_whispers}")
    print(f"  Whisper-to-behavior-change ratio: {sum(bcrs)/max(1, total_whispers):.4f}")

    print("\n--- Per-agent summary stats ---")
    for agent in sim.agents:
        if agent._episodic_summary is not None:
            summary = agent.get_episodic_summary()
            buf_size = agent._episodic_buffer.size() if agent._episodic_buffer else 0
            recv_count = len(agent._received_summaries)
            print(f"  Agent {agent.id}: buffer={buf_size}, received_summaries={recv_count}, "
                  f"summary_mean={summary.mean().item():.4f}, summary_std={summary.std().item():.4f}")

    print("\n--- Correlation: similarity vs behavioral change ---")
    if len(sims) == len(bcrs) and len(sims) > 10:
        mean_sim = sum(sims) / len(sims)
        mean_bcr = sum(bcrs) / len(bcrs)
        cov = sum((s - mean_sim) * (b - mean_bcr) for s, b in zip(sims, bcrs)) / max(1, len(sims) - 1)
        std_sim = math.sqrt(sum((s - mean_sim)**2 for s in sims) / max(1, len(sims) - 1))
        std_bcr = math.sqrt(sum((b - mean_bcr)**2 for b in bcrs) / max(1, len(bcrs) - 1))
        corr = cov / max(1e-10, std_sim * std_bcr)
        print(f"  Pearson correlation (similarity ↔ behavior change): {corr:.4f}")
        print(f"  Interpretation: {'Strong positive' if corr > 0.5 else 'Strong negative' if corr < -0.5 else 'Weak/moderate correlation'}")

print("\n--- FINAL ANSWERS ---")
if results:
    overall_sim = sum(sims)/len(sims) if results else 0
    overall_bcr = sum(bcrs)/len(bcrs) if results else 0
    print(f"Q1: Similarity among agents with similar experiences: {overall_sim:.4f} avg cosine similarity")
    if overall_sim > 0.3:
        print("  → YES: Agents in similar states tend to produce similar 10D summary vectors.")
    elif overall_sim > 0.1:
        print("  → PARTIAL: Summary vectors show moderate similarity, suggesting some common experiential patterns.")
    else:
        print("  → NO: Summary vectors remain dissimilar, suggesting each agent develops unique representations.")

    print(f"\nQ2: Behavioral change after receiving summary: {overall_bcr:.4f} avg change rate")
    if overall_bcr > 0.2:
        print("  → YES: Agents visibly alter their behavior after receiving other agents' summary vectors.")
    elif overall_bcr > 0.05:
        print("  → PARTIAL: Some behavioral adjustment observed after summary reception.")
    else:
        print("  → NO: Agents do not significantly change behavior after receiving summaries.")
else:
    print("No episodic results recorded — simulation may have been too short or crashed.")

print(f"\nFiles written: episodic_results.jsonl, telemetry.jsonl, conflicts.jsonl")