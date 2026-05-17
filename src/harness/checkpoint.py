import os
import json
import tempfile
import torch


class Checkpointer:
    def __init__(self, directory: str = "checkpoints", interval_ticks: int = 100):
        self.directory = directory
        self.interval = interval_ticks
        os.makedirs(directory, exist_ok=True)

    def save(self, tick: int, sim_state: dict) -> str:
        tmp = os.path.join(self.directory, f"ckpt_{tick}.tmp")
        final = os.path.join(self.directory, f"ckpt_{tick}.pt")
        torch.save(sim_state, tmp)
        os.replace(tmp, final)
        return final

    def load(self, path: str) -> dict:
        return torch.load(path, map_location="cpu")

    def latest(self) -> str:
        files = [f for f in os.listdir(self.directory) if f.endswith(".pt")]
        if not files:
            return None
        files.sort(key=lambda f: int(f.split("_")[1].split(".")[0]))
        return os.path.join(self.directory, files[-1])

    def build_state(self, sim) -> dict:
        agent_states = []
        for agent in sim.agents:
            agent_states.append({
                "id": agent.id,
                "position": agent.position,
                "ram": agent.ram,
                "alive": agent.alive,
                "last_action": agent.last_action,
                "last_direction": agent.last_direction,
                "last_intent": agent.last_intent,
                "action_history": agent.action_history[-500:],
                "brain_out_targets": agent.brain.out_targets.cpu().clone(),
                "brain_out_weights": agent.brain.out_weights.cpu().clone(),
                "brain_activations": agent.brain.activations.cpu().clone(),
                "brain_activity_history": agent.brain.activity_history.cpu().clone(),
            })
        return {
            "tick": sim.tick_count,
            "config": sim.config,
            "env_safe_zone_center": sim.env.safe_zone.center,
            "env_safe_zone_radius": sim.env.safe_zone.radius,
            "env_ram_nodes": [
                {"position": n.position, "yield_amount": n.yield_amount}
                for n in sim.env.active_ram_nodes
            ],
            "agents": agent_states,
        }
