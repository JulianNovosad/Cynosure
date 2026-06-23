"""
State persistence mechanism for continuous learning across sessions.
Maintains memory, learned dynamics, and learned policies without explicit checkpoint restore.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pickle


class StatePersistence:
    """Manages saving and loading agent state across sessions."""

    def __init__(self, storage_dir: str = ".agent_state"):
        """Initialize persistence layer."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # State files
        self.agent_state_file = self.storage_dir / "agent_state.pkl"
        self.dynamics_model_file = self.storage_dir / "dynamics_model.pkl"
        self.memory_file = self.storage_dir / "memory.json"
        self.checkpoint_file = self.storage_dir / "checkpoint.json"

    def save_agent_state(self, agent_state: Dict[str, Any]) -> None:
        """
        Save agent learning state (policy parameters, weights, etc.).
        Crucial for transfer to physical deployment without redesign.
        """
        with open(self.agent_state_file, "wb") as f:
            pickle.dump(agent_state, f)

    def load_agent_state(self) -> Optional[Dict[str, Any]]:
        """Load previously saved agent state."""
        if self.agent_state_file.exists():
            with open(self.agent_state_file, "rb") as f:
                return pickle.load(f)
        return None

    def save_dynamics_model(self, model_state: Dict[str, Any]) -> None:
        """
        Save learned environment dynamics model.
        This allows agent to predict consequences of actions across time steps.
        """
        with open(self.dynamics_model_file, "wb") as f:
            pickle.dump(model_state, f)

    def load_dynamics_model(self) -> Optional[Dict[str, Any]]:
        """Load previously learned dynamics model."""
        if self.dynamics_model_file.exists():
            with open(self.dynamics_model_file, "rb") as f:
                return pickle.load(f)
        return None

    def save_memory(self, memory: Dict[str, Any]) -> None:
        """
        Save agent memory: state summaries, action histories, outcome traces.
        Used for long-horizon tasks.
        """
        with open(self.memory_file, "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            memory_serializable = self._make_serializable(memory)
            json.dump(memory_serializable, f, indent=2)

    def load_memory(self) -> Optional[Dict[str, Any]]:
        """Load previously saved memory."""
        if self.memory_file.exists():
            with open(self.memory_file, "r") as f:
                return json.load(f)
        return None

    def save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Save recovery checkpoint for mutation recovery protocol.
        Stores timestamp, agent state hash, learning progress.
        """
        checkpoint["timestamp"] = datetime.now().isoformat()
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load recovery checkpoint."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r") as f:
                return json.load(f)
        return None

    def restore_continuous_state(self) -> Dict[str, Any]:
        """
        Restore full continuous state for session resumption.
        Critical invariant: Resetting state without explicit checkpoint restore
        breaks continuity and counts as failure for evaluation runs.
        """
        state = {
            "agent_state": self.load_agent_state(),
            "dynamics_model": self.load_dynamics_model(),
            "memory": self.load_memory(),
            "checkpoint": self.load_checkpoint(),
        }

        # Validate continuity: at least one component must be present
        if not any(state.values()):
            raise RuntimeError(
                "No continuous state found. "
                "Cannot reset state without explicit checkpoint restore. "
                "Continuity broken."
            )

        return state

    def is_first_session(self) -> bool:
        """Check if this is the first session (no saved state exists)."""
        return not self.agent_state_file.exists()

    def clear_state(self) -> None:
        """Clear all saved state (use only for explicit reset)."""
        for f in [self.agent_state_file, self.dynamics_model_file,
                  self.memory_file, self.checkpoint_file]:
            if f.exists():
                f.unlink()

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable types to JSON-compatible format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: StatePersistence._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [StatePersistence._make_serializable(item) for item in obj]
        else:
            return obj


class ContinuousMemory:
    """
    Agent memory system for long-horizon tasks.
    Stores state summaries, action histories, and outcome traces.
    """

    def __init__(self, max_history: int = 1000):
        """Initialize memory with fixed capacity."""
        self.max_history = max_history
        self.state_history = []
        self.action_history = []
        self.outcome_history = []
        self.episode_summaries = []

    def record_step(self, state: np.ndarray, action: Dict[str, float], outcome: float) -> None:
        """Record one interaction step."""
        if len(self.state_history) >= self.max_history:
            self.state_history.pop(0)
            self.action_history.pop(0)
            self.outcome_history.pop(0)

        self.state_history.append(state.tolist() if isinstance(state, np.ndarray) else state)
        self.action_history.append(action)
        self.outcome_history.append(outcome)

    def record_episode_summary(self, summary: Dict[str, Any]) -> None:
        """Record summary of completed episode."""
        self.episode_summaries.append(summary)
        # Keep only recent summaries
        if len(self.episode_summaries) > 50:
            self.episode_summaries.pop(0)

    def get_recent_experience(self, lookback: int = 100) -> Dict[str, Any]:
        """Retrieve recent experience for learning."""
        n = min(lookback, len(self.state_history))
        return {
            "states": self.state_history[-n:],
            "actions": self.action_history[-n:],
            "outcomes": self.outcome_history[-n:],
        }

    def get_episode_patterns(self) -> list:
        """Retrieve patterns from episode summaries for long-horizon planning."""
        return self.episode_summaries

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to serializable format."""
        return {
            "state_history": self.state_history,
            "action_history": self.action_history,
            "outcome_history": self.outcome_history,
            "episode_summaries": self.episode_summaries,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ContinuousMemory":
        """Restore memory from serialized format."""
        mem = ContinuousMemory()
        mem.state_history = data.get("state_history", [])
        mem.action_history = data.get("action_history", [])
        mem.outcome_history = data.get("outcome_history", [])
        mem.episode_summaries = data.get("episode_summaries", [])
        return mem
