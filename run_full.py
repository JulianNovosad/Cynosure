import time
import torch
from src.config import Config, LRSGConfig, ArenaConfig, AgentConfig, SimulationConfig
from src.simulation import Simulation

cfg = Config(
    lrsg=LRSGConfig(num_neurons=4_194_304),
    arena=ArenaConfig(grid_size=(256, 256, 256)),
    simulation=SimulationConfig(num_agents=15, max_ticks=100),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

print(f"Cynosure: 15 agents x 4.19M neurons on {gpu_name}")
print(f"VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f}GB / {total_vram:.1f}GB")

t0 = time.time()
sim = Simulation(cfg, device=device)
init_time = time.time() - t0
vram_used = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
print(f"Init: {init_time:.1f}s | VRAM: {vram_used:.1f}GB")

t0 = time.time()
metrics = sim.run()
run_time = time.time() - t0

print(f"\n=== Results ===")
print(f"Run 100 ticks: {run_time:.1f}s ({run_time/100*1000:.0f}ms/tick)")
final = sim.status
print(f"Survivors: {final['agents_alive']}/{cfg.simulation.num_agents}")
for k, v in final["metrics"].items():
    print(f"  {k}: {v}")

vram_final = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
print(f"Final VRAM: {vram_final:.1f}GB")
