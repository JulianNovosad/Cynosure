"""
Event-driven cluster scheduler.

Per-tick execution (from plans/rewrite.md):
  1. receive SensorPacket → convert to input activations
  2. select active clusters by priority (top-N, incoming weight × signal)
  3. load clusters into VRAM if not resident
  4. run sparse matmul + recurrent update + gating per cluster
  5. propagate outputs through cross-cluster routing table
  6. evict least-recently-active clusters
  7. return ActionPacket

Scale targets:
  active clusters in VRAM: 4–12
  total stored clusters:   10K–50K
  active VRAM footprint:   ~6–15 MB cluster data + ~200MB compute buffers
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.cluster.cluster import Cluster, cluster_forward, hebbian_update, create_cluster
from src.cluster.packet import SensorPacket, ActionPacket, CAMERA_FEATURE_DIM, SERVO_CHANNELS


@dataclass
class _ClusterSlot:
    """Tracks a cluster resident in VRAM (active pool)."""
    cluster: Cluster
    last_active_tick: int = 0
    total_incoming_signal: float = 0.0


@dataclass
class _RoutingEntry:
    """Cross-cluster routing: src_cluster output neuron → tgt_cluster input neuron."""
    src_cluster_id: int
    src_neuron: int
    tgt_cluster_id: int
    tgt_neuron: int
    weight: float


class ClusterScheduler:
    """
    Manages a pool of clusters: loads, runs, routes, evicts.

    max_active: maximum clusters resident in VRAM simultaneously (4–12 per spec)
    """

    def __init__(
        self,
        max_active: int = 8,
        rng: np.random.Generator | None = None,
    ) -> None:
        assert 4 <= max_active <= 12, f"max_active must be 4–12 per spec, got {max_active}"
        self.max_active = max_active
        self._rng = rng if rng is not None else np.random.default_rng()

        # All clusters known to scheduler (disk/RAM store)
        self._cluster_store: dict[int, Cluster] = {}

        # Clusters currently loaded in VRAM
        self._active: dict[int, _ClusterSlot] = {}

        # Cross-cluster routing table: maps cluster_id → list of RoutingEntry
        self._routing: dict[int, list[_RoutingEntry]] = {}

        self._tick: int = 0

        # Pending input injections from routing: cluster_id → {neuron_idx: signal}
        self._pending_injections: dict[int, dict[int, float]] = {}

    # ------------------------------------------------------------------
    # Cluster registration
    # ------------------------------------------------------------------

    def register_cluster(self, cluster: Cluster) -> None:
        """Add a cluster to the scheduler's store (not yet loaded into VRAM)."""
        self._cluster_store[cluster.cluster_id] = cluster
        self._routing[cluster.cluster_id] = []

    def add_routing(
        self,
        src_cluster_id: int,
        src_neuron: int,
        tgt_cluster_id: int,
        tgt_neuron: int,
        weight: float = 1.0,
    ) -> None:
        """Add a cross-cluster routing entry."""
        assert src_cluster_id in self._cluster_store
        assert tgt_cluster_id in self._cluster_store
        self._routing[src_cluster_id].append(
            _RoutingEntry(src_cluster_id, src_neuron, tgt_cluster_id, tgt_neuron, weight)
        )

    @property
    def n_clusters(self) -> int:
        return len(self._cluster_store)

    @property
    def n_active(self) -> int:
        return len(self._active)

    # ------------------------------------------------------------------
    # Priority scoring
    # ------------------------------------------------------------------

    def _score_cluster(self, cluster_id: int, signal_mag: float) -> float:
        """Priority = sum(|incoming routing weights|) × incoming signal magnitude."""
        routes = self._routing.get(cluster_id, [])
        if not routes:
            return signal_mag  # standalone cluster: use raw signal
        incoming_weight_sum = sum(abs(r.weight) for r in routes)
        return incoming_weight_sum * signal_mag

    def _select_top_n(self, sensor_signal_mag: float) -> list[int]:
        """Select top-N cluster IDs by priority score."""
        if not self._cluster_store:
            return []

        scores = [
            (-self._score_cluster(cid, sensor_signal_mag), cid)
            for cid in self._cluster_store
        ]
        heapq.heapify(scores)
        n = min(self.max_active, len(scores))
        selected = []
        for _ in range(n):
            _, cid = heapq.heappop(scores)
            selected.append(cid)
        return selected

    # ------------------------------------------------------------------
    # VRAM load / evict
    # ------------------------------------------------------------------

    def _load(self, cluster_id: int) -> None:
        """Load cluster into active pool if not already resident."""
        if cluster_id in self._active:
            return
        cluster = self._cluster_store[cluster_id]
        self._active[cluster_id] = _ClusterSlot(cluster=cluster, last_active_tick=self._tick)

    def _evict_lru(self, keep: set[int]) -> None:
        """Evict least-recently-used clusters not in keep set until within max_active."""
        while len(self._active) > self.max_active:
            evict_id = min(
                (cid for cid in self._active if cid not in keep),
                key=lambda cid: self._active[cid].last_active_tick,
                default=None,
            )
            if evict_id is None:
                break
            del self._active[evict_id]

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def step(self, packet: SensorPacket, reward: Optional[float] = None) -> ActionPacket:
        """
        Process one sensor packet and return a motor action.

        Steps per plans/rewrite.md scheduler spec.
        """
        self._tick += 1

        # --- 1. Convert packet to input activations ---
        input_vec = packet.to_activation_vector()
        signal_mag = float(np.linalg.norm(input_vec.astype(np.float32)))

        # --- 2. Select top-N clusters ---
        selected_ids = self._select_top_n(signal_mag)
        selected_set = set(selected_ids)

        # --- 3. Load selected clusters ---
        for cid in selected_ids:
            self._load(cid)

        # --- 4. Run each active cluster ---
        cluster_outputs: dict[int, np.ndarray] = {}
        for cid in selected_ids:
            slot = self._active[cid]
            slot.last_active_tick = self._tick

            # Merge pending injections into input
            cluster_input = input_vec.copy()
            if cid in self._pending_injections:
                for neuron_idx, sig in self._pending_injections[cid].items():
                    if neuron_idx < len(cluster_input):
                        cluster_input[neuron_idx] = np.float16(
                            float(cluster_input[neuron_idx]) + sig
                        )
                del self._pending_injections[cid]

            out = cluster_forward(slot.cluster, cluster_input)
            cluster_outputs[cid] = out

            if reward is not None:
                hebbian_update(slot.cluster, reward)

        # --- 5. Propagate outputs through routing table ---
        self._pending_injections.clear()
        for src_cid, out in cluster_outputs.items():
            for route in self._routing.get(src_cid, []):
                tgt = route.tgt_cluster_id
                if tgt not in self._pending_injections:
                    self._pending_injections[tgt] = {}
                src_val = float(out[route.src_neuron]) if route.src_neuron < len(out) else 0.0
                prev = self._pending_injections[tgt].get(route.tgt_neuron, 0.0)
                self._pending_injections[tgt][route.tgt_neuron] = prev + src_val * route.weight

        # --- 6. Evict LRU clusters ---
        self._evict_lru(keep=selected_set)

        # --- 7. Aggregate action from cluster outputs ---
        if not cluster_outputs:
            return ActionPacket.stop()

        # Average all active cluster outputs (first MOTOR_CHANNELS dims drive motors)
        stacked = np.stack(
            [out for out in cluster_outputs.values()], axis=0
        ).astype(np.float32)
        agg = stacked.mean(axis=0).astype(np.float16)
        return ActionPacket.from_activation(agg)

    def broadcast_reward(self, reward: float) -> None:
        """Apply reward to all currently active clusters (called outside tick if needed)."""
        for slot in self._active.values():
            hebbian_update(slot.cluster, reward)

    def vram_footprint_bytes(self) -> int:
        """Total bytes used by active cluster data (not counting compute buffers)."""
        return sum(slot.cluster.memory_bytes() for slot in self._active.values())

    def active_cluster_ids(self) -> list[int]:
        return list(self._active.keys())


def build_default_scheduler(
    n_clusters: int = 8,
    n_neurons: int = 4096,
    n_edges: int = 40000,
    max_active: int = 8,
    rng: np.random.Generator | None = None,
) -> ClusterScheduler:
    """
    Create a scheduler populated with n_clusters randomly initialized clusters.

    Adds a linear routing chain: 0 → 1 → 2 → … → n-1 (first neuron feeds into next).
    """
    if rng is None:
        rng = np.random.default_rng()

    sched = ClusterScheduler(max_active=max_active, rng=rng)
    for i in range(n_clusters):
        c = create_cluster(i, n_neurons=n_neurons, n_edges=n_edges, rng=rng)
        sched.register_cluster(c)

    # Wire clusters in a chain for cross-cluster signal propagation
    for i in range(n_clusters - 1):
        sched.add_routing(
            src_cluster_id=i,
            src_neuron=0,
            tgt_cluster_id=i + 1,
            tgt_neuron=0,
            weight=0.5,
        )

    return sched
