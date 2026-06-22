import os
import json
import io
import hmac
import hashlib
import torch

_CHECKPOINT_SIGNING_KEY: bytes = None

def set_checkpoint_signing_key(key: bytes):
    global _CHECKPOINT_SIGNING_KEY
    if len(key) < 16:
        raise ValueError("Signing key must be at least 16 bytes")
    _CHECKPOINT_SIGNING_KEY = key

def _sign_bytes(data: bytes) -> str:
    if _CHECKPOINT_SIGNING_KEY is None:
        return ""
    return hmac.new(_CHECKPOINT_SIGNING_KEY, data, hashlib.sha256).hexdigest()

def _verify_signature(data: bytes, sig: str) -> bool:
    if _CHECKPOINT_SIGNING_KEY is None:
        return True
    expected = hmac.new(_CHECKPOINT_SIGNING_KEY, data, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)


class Checkpointer:
    def __init__(self, directory: str = "checkpoints", interval_ticks: int = 100):
        self.directory = directory
        self.interval = interval_ticks
        os.makedirs(directory, exist_ok=True)

    def save(self, tick: int, sim_state: dict) -> str:
        tmp = os.path.join(self.directory, f"ckpt_{tick}.tmp")
        final = os.path.join(self.directory, f"ckpt_{tick}.pt")
        signed = dict(sim_state)
        buf = io.BytesIO()
        torch.save(signed, buf)
        raw = buf.getvalue()
        sig = _sign_bytes(raw)
        if sig:
            signed["_integrity_sig"] = sig
            buf = io.BytesIO()
            torch.save(signed, buf)
            raw = buf.getvalue()
        with open(tmp, "wb") as f:
            f.write(raw)
        os.replace(tmp, final)
        return final

    def load(self, path: str) -> dict:
        with open(path, "rb") as f:
            raw = f.read()
        state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        sig = state.pop("_integrity_sig", "")
        if sig and not _verify_signature(raw, sig):
            raise ValueError(f"Checkpoint integrity check FAILED for {path}")
        return state

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
