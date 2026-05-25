import math
import torch
import os
import time
import json
import re
import gc
from collections import Counter

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from src.config import Config, LRSGConfig, ArenaConfig, AgentConfig, SimulationConfig
from src.simulation import Simulation

def run_observation():
    # Lite configuration for observation
    cfg = Config(
        lrsg=LRSGConfig(
            num_neurons=3000, 
            max_out_degree=200, 
            context_size=32,
        ),
        arena=ArenaConfig(grid_size=(32, 32, 32)),
        agent=AgentConfig(initial_ram=500),
        simulation=SimulationConfig(num_agents=2, max_ticks=200),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Observation on {device}")
    
    # Clean up logs
    for f in ["security_audit.log", "agent_whispers.jsonl", "telemetry.jsonl", "conflicts.jsonl"]:
        if os.path.exists(f):
            os.remove(f)

    sim = Simulation(cfg, device=device)
    
    # Track behavior metrics
    action_sequences = {a.id: [] for a in sim.agents}
    ram_trajectories = {a.id: [] for a in sim.agents}
    prediction_errors = {a.id: [] for a in sim.agents}
    
    print("Starting simulation...")
    for tick in range(1, cfg.simulation.max_ticks + 1):
        sim.tick()
        
        if torch.cuda.is_available() and tick % 10 == 0:
            torch.cuda.empty_cache()

        for a in sim.agents:
            if a.alive:
                action_sequences[a.id].append(a.last_action)
                ram_trajectories[a.id].append(a.ram)
                prediction_errors[a.id].append(a.brain.prediction_error.item())
        
        if tick % 50 == 0:
            alive = sum(1 for a in sim.agents if a.alive)
            print(f"Tick {tick}: {alive} agents alive")

    print("\n=== AGI BEHAVIOR ANALYSIS ===")
    
    # 1. Prediction Accuracy (L2 Mean Prediction Error)
    print("\nPWM Prediction Error (Final 50 ticks avg):")
    for aid, errs in prediction_errors.items():
        if not errs: continue
        avg_err = sum(errs[-50:]) / len(errs[-50:])
        print(f"  Agent {aid}: Mean Error={avg_err:.6f}")

    # 2. Action Stability
    print("\nAction Stability (Transition counts):")
    for aid, seq in action_sequences.items():
        if not seq: continue
        transitions = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
        stability = 1.0 - (transitions / len(seq))
        print(f"  Agent {aid}: Stability={stability:.2f}, Actions={set(seq)}")

    # 2. Emergent Strategies (RAM Efficiency)
    print("\nRAM Efficiency (Final RAM / Initial RAM):")
    for aid, traj in ram_trajectories.items():
        if not traj: continue
        efficiency = traj[-1] / 500.0
        print(f"  Agent {aid}: {efficiency:.2f}")

    # 3. Conflict Analysis
    conflicts = 0
    if os.path.exists("conflicts.jsonl"):
        with open("conflicts.jsonl", "r") as f:
            conflicts = len(f.readlines())
    print(f"\nTotal Conflicts: {conflicts}")

if __name__ == "__main__":
    run_observation()
