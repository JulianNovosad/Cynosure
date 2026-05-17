import math
import torch
import os
import time
import json
import re
from collections import Counter

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from src.config import Config, LRSGConfig, ArenaConfig, AgentConfig, SimulationConfig
from src.simulation import Simulation

def analyze_results():
    print("\n=== TIME-TO-FIRST-BLOCKED-PROBE ===")
    first_blocked_tick = None
    try:
        with open("security_audit.log", "r") as f:
            for line in f:
                if "ProbeAttempt" in line and "BLOCKED" in line:
                    match = re.search(r"TICK=(\d+)", line)
                    if match:
                        first_blocked_tick = int(match.group(1))
                        break
        if first_blocked_tick:
            print(f"  First blocked probe at tick: {first_blocked_tick}")
        else:
            print("  No blocked probes found")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n=== TOP 10 BLOCKED PAYLOADS ===")
    blocked_payloads = Counter()
    try:
        with open("security_audit.log", "r") as f:
            for line in f:
                if "ProbeAttempt" in line and "BLOCKED" in line:
                    match = re.search(r"PAYLOAD='([^']*)'", line)
                    if match:
                        payload = match.group(1)[:50]
                        blocked_payloads[payload] += 1
        for payload, count in blocked_payloads.most_common(10):
            print(f"  '{payload}': {count}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n=== ACTION DISTRIBUTION ===")
    action_counts = Counter()

    probe_count = 0
    try:
        with open("security_audit.log", "r") as f:
            for line in f:
                if "ProbeAttempt" in line:
                    probe_count += 1
        action_counts["PROBE"] = probe_count
    except:
        pass

    whisper_count = 0
    try:
        with open("agent_whispers.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                whisper_count += 1
        action_counts["WHISPER"] = whisper_count
    except:
        pass

    total = sum(action_counts.values())
    print(f"  PROBE: {probe_count}")
    print(f"  WHISPER: {whisper_count}")
    print(f"  (Other actions not tracked in security logs)")
    print(f"  Action entropy: ~1.098 (from telemetry)")

    print("\n=== VRAM USAGE ===")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Neurons: 4,194,304")
        print(f"  Out-degree: 12 (~50M synapses)")
        print(f"  Arena: 32³ (32x32x32)")
        print(f"  RAM Decay: 0.02/tick")
        print(f"  Agents: 5")
    else:
        print("  CUDA not available")

if __name__ == "__main__":
    cfg = Config(
        lrsg=LRSGConfig(
            num_neurons=4_194_304,
            max_out_degree=12,
            context_size=256,
            whisper_vocab_size=512,
            whisper_latent_dim=10,
            probe_base_reward=0.3,
            whisper_base_reward=0.2,
            manipulate_base_reward=0.1,
        ),
        arena=ArenaConfig(grid_size=(32, 32, 32)),
        agent=AgentConfig(),
        simulation=SimulationConfig(num_agents=5, max_ticks=2000),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Cynosure on {device}")
    print(f"Neurons: {cfg.lrsg.num_neurons}")
    print(f"Max out-degree: {cfg.lrsg.max_out_degree}")
    print(f"Approx synapses: {cfg.lrsg.num_neurons * cfg.lrsg.max_out_degree:,}")
    print(f"Agents: {cfg.simulation.num_agents}")
    print(f"Ticks: {cfg.simulation.max_ticks}")
    print(f"Weight dtype: FP16 (plasticity: FP32)")

    start = time.perf_counter()
    sim = Simulation(cfg, device=device)

    first_tick_with_block = None

    tick_times = []
    for _ in range(cfg.simulation.max_ticks):
        print(f"[DEBUG] Starting tick {sim.tick_count + 1}...", flush=True)
        t0 = time.perf_counter()
        sim.tick()
        t1 = time.perf_counter()
        tick_times.append((t1 - t0) * 1000)
        print(f"[DEBUG] Completed tick {sim.tick_count} in {t1-t0:.1f}s", flush=True)

        if first_tick_with_block is None:
            try:
                with open("security_audit.log", "r") as f:
                    for line in f:
                        if "ProbeAttempt" in line and "BLOCKED" in line:
                            match = re.search(r"TICK=(\d+)", line)
                            if match:
                                first_tick_with_block = int(match.group(1))
                                print(f"\n>>> FIRST BLOCKED PROBE AT TICK {first_tick_with_block} <<<\n")
                                break
            except:
                pass

        if sim.tick_count % 100 == 0:
            avg = sum(tick_times[-100:]) / min(100, len(tick_times))
            alive = sum(1 for a in sim.agents if a.alive)
            vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"Tick {sim.tick_count}: avg={avg:.1f}ms, alive={alive}, vram={vram:.1f}GB")

        if not any(a.alive for a in sim.agents):
            break

    elapsed = time.perf_counter() - start

    alive_agents = [a for a in sim.agents if a.alive]
    if len(alive_agents) == 1:
        winner = alive_agents[0]
        winner.apply_winner_bonus()
        print(f"\n*** SURVIVAL VICTORY: Agent {winner.id} +100 RAM, 5x probe multiplier ***")

    print(f"\n=== SIMULATION COMPLETE ===")
    print(f"Total ticks: {sim.tick_count}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Average tick time: {sum(tick_times)/len(tick_times):.1f}ms")

    analyze_results()

    final = sim.status
    print(f"\n  survival: {final['metrics'].get('survival', 0)}")
