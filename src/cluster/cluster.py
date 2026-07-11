"""
Single-cluster runtime: sparse matmul + recurrent update + Hebbian eligibility traces.

Cluster spec (from plans/rewrite.md):
  neurons:         2K–8K
  edges:           20K–80K (sparse only, no dense fallback)
  edge storage:    src uint16, tgt uint16, weight float16 = 6 bytes/edge
  neuron state:    activation float16 + recurrent float16 = 4 bytes/neuron
  eligibility:     float16 per edge = 2 bytes/edge
  total:           ~500KB–1.2MB per cluster
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Cluster:
    """
    A single sparse cluster.

    Edges stored as parallel arrays (COO format without scipy dependency).
    All weight/state arrays are float16 to match spec byte budgets.
    """
    cluster_id: int
    n_neurons: int                  # 2K–8K
    src: np.ndarray                 # uint16, shape (n_edges,)
    tgt: np.ndarray                 # uint16, shape (n_edges,)
    weight: np.ndarray              # float16, shape (n_edges,)
    activation: np.ndarray          # float16, shape (n_neurons,)
    recurrent: np.ndarray           # float16, shape (n_neurons,)
    trace: np.ndarray               # float16, shape (n_edges,) — eligibility trace

    # Tunable learning hyperparams stored per-cluster
    trace_decay: float = 0.95       # multiplicative decay per tick (~200–500ms window)
    lr: float = 0.01                # Hebbian learning rate
    weight_decay: float = 0.9999    # global weight decay per tick (prevents unbounded growth)
    recurrent_decay: float = 0.8    # recurrent state decay per tick

    def n_edges(self) -> int:
        return len(self.src)

    def memory_bytes(self) -> int:
        """Approximate cluster memory footprint."""
        edge_bytes = self.n_edges() * (2 + 2 + 2 + 2)  # src, tgt, weight, trace
        neuron_bytes = self.n_neurons * (2 + 2)          # activation, recurrent
        return edge_bytes + neuron_bytes


def create_cluster(
    cluster_id: int,
    n_neurons: int = 4096,
    n_edges: int = 40000,
    rng: np.random.Generator | None = None,
) -> Cluster:
    """Create a randomly initialized sparse cluster within spec bounds."""
    assert 2000 <= n_neurons <= 8192, f"n_neurons must be 2K–8K, got {n_neurons}"
    assert 20000 <= n_edges <= 80000, f"n_edges must be 20K–80K, got {n_edges}"

    if rng is None:
        rng = np.random.default_rng()

    n = np.uint16(n_neurons)
    src = rng.integers(0, n_neurons, size=n_edges, dtype=np.uint16)
    tgt = rng.integers(0, n_neurons, size=n_edges, dtype=np.uint16)
    # Avoid self-loops
    mask = src == tgt
    tgt[mask] = (tgt[mask] + 1) % n_neurons

    # Xavier-style scaling by actual in-degree (fan-in), not n_neurons.
    # For a sparse graph, avg fan-in = n_edges / n_neurons (~10 for spec defaults),
    # which is far smaller than n_neurons — scaling by 1/sqrt(n_neurons) here would
    # under-scale weights by ~sqrt(n_neurons / fan_in) and leave pre-activation
    # sums too close to zero for tanh to ever leave its near-linear regime.
    #
    # k=0.15 is chosen empirically: combined with the recurrent_decay=0.8 direct
    # self-feedback term below, this keeps the recurrent loop's effective spectral
    # gain under 1 (state decays to ~0 within 50 zero-input ticks — see
    # TestClusterForward::test_recurrent_decays_to_zero_on_zero_input). Larger k
    # produces bigger single-tick signals but pushes the loop into a runaway
    # regime that saturates and never decays absent input, which violates the
    # "stable internal dynamics" requirement. Motor responsiveness is recovered
    # downstream via an output gain in ActionPacket.from_activation instead of
    # by destabilizing the recurrent dynamics.
    fan_in = max(1.0, n_edges / n_neurons)
    k = 0.15
    scale = np.float16(k / np.sqrt(fan_in))
    weight = (rng.standard_normal(n_edges) * scale).astype(np.float16)

    return Cluster(
        cluster_id=cluster_id,
        n_neurons=n_neurons,
        src=src,
        tgt=tgt,
        weight=weight,
        activation=np.zeros(n_neurons, dtype=np.float16),
        recurrent=np.zeros(n_neurons, dtype=np.float16),
        trace=np.zeros(n_edges, dtype=np.float16),
    )


def cluster_forward(
    cluster: Cluster,
    input_vec: np.ndarray,
) -> np.ndarray:
    """
    Run one forward tick through a cluster.

    Steps:
      1. Write input activations into first len(input_vec) neurons
      2. Sparse matmul: accumulate weighted pre-activations into post-neurons
      3. Add recurrent contribution
      4. Apply tanh gating (keeps values in [−1, 1], prevents float16 overflow)
      5. Update recurrent state
      6. Update eligibility traces (pre × post co-activation)

    Returns: activation vector (float16, shape (n_neurons,))
    """
    n = cluster.n_neurons
    input_len = min(len(input_vec), n)

    # --- 1. Inject input ---
    pre_activation = cluster.recurrent.copy()
    pre_activation[:input_len] = input_vec[:input_len].astype(np.float16)

    # --- 2. Sparse matmul (COO scatter-add via bincount) ---
    # new_act[tgt] += weight * pre_activation[src]
    # bincount is ~10× faster than np.add.at for large edge counts
    contribution = cluster.weight.astype(np.float32) * pre_activation[cluster.src].astype(np.float32)
    new_act_f32 = np.bincount(
        cluster.tgt.astype(np.intp), weights=contribution.astype(np.float64), minlength=n
    ).astype(np.float32)

    # --- 3. Add recurrent ---
    new_act_f32 += cluster.recurrent.astype(np.float32) * cluster.recurrent_decay

    # --- 4. Gate with tanh ---
    new_act = np.tanh(new_act_f32).astype(np.float16)

    # --- 5. Update eligibility traces: trace = decay * trace + pre * post ---
    pre_act = pre_activation[cluster.src].astype(np.float32)
    post_act = new_act[cluster.tgt].astype(np.float32)
    cluster.trace = (
        cluster.trace_decay * cluster.trace.astype(np.float32)
        + pre_act * post_act
    ).astype(np.float16)

    # --- 6. Store updated states ---
    cluster.recurrent = new_act.copy()
    cluster.activation = new_act

    return new_act


def hebbian_update(cluster: Cluster, reward: float) -> None:
    """
    Apply reward-modulated Hebbian update to all edges in cluster.

    Δw = lr × trace × reward
    Then apply global weight decay to prevent unbounded growth.

    Called when a reward signal arrives (scalar, global broadcast).
    """
    delta = np.float16(cluster.lr) * cluster.trace * np.float16(reward)
    cluster.weight = (
        cluster.weight.astype(np.float32) + delta.astype(np.float32)
    ).astype(np.float16)

    # Global weight decay
    cluster.weight = (cluster.weight.astype(np.float32) * cluster.weight_decay).astype(np.float16)

    # Clamp weights to prevent float16 overflow (max ~65504)
    np.clip(cluster.weight, -10.0, 10.0, out=cluster.weight)
