import sys, os
sys.path.insert(0, r"C:\Cynosure")
sys.stdout.reconfigure(encoding='utf-8')

import torch, random, json, time

from src.config import Config, LRSGConfig, ArenaConfig, AgentConfig, SimulationConfig, EpisodicCompressorConfig
from src.simulation.loop import Simulation

for f in ['telemetry.jsonl','conflicts.jsonl','episodic_results.jsonl','detailed_metrics.jsonl','security_audit.log']:
    if os.path.exists(f):
        os.remove(f)

cfg = Config(
    lrsg=LRSGConfig(num_neurons=32768, max_out_degree=8, context_size=128),
    arena=ArenaConfig(grid_size=(16,16,16)),
    agent=AgentConfig(initial_ram=200, vision_range=4, sensory_radius=6.0),
    simulation=SimulationConfig(num_agents=2, max_ticks=20, use_llm_announcer=False),
    episodic=EpisodicCompressorConfig(input_dim=10, hidden_dim=20, latent_dim=16),
)

torch.manual_seed(0)
random.seed(0)
sim = Simulation(cfg, torch.device('cpu'))
for _ in range(20):
    sim.tick()

print('Smoke test: tick', sim.tick_count, 'conflict_count', sim._conflict_count)

rows = []
if os.path.exists('detailed_metrics.jsonl'):
    with open('detailed_metrics.jsonl') as f:
        for line in f:
            try: rows.append(json.loads(line))
            except: pass

print('Detailed rows:', len(rows))
if rows:
    r = rows[0]
    print('Keys:', list(r.keys()))
    print('action_pct len:', len(r['action_pct']))
    print('prox matrix shape:', len(r['proximity_matrix']), 'x', len(r['proximity_matrix'][0]) if r['proximity_matrix'] else 0)
    print('ae_sim:', r['ae_similarity'])
    print('behavioral_change_rate:', r['behavioral_change_rate'])
    print('token_diversity:', r['token_diversity'])
print('Smoke test PASSED')