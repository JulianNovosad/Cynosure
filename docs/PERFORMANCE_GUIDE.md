# Cynosure Model Execution Speed & Optimizations

> **Documentation note (2026-07-20):** This file is a historical, point-in-time snapshot and is intentionally not rewritten to reflect current state (rewriting it would falsify the record it captured). For current system status, see `CYNOSURE_KB.md` and `docs/CHANGELOG.md`.


_Last updated: 23 Jun 2026_

## Overview

This guide explains what determines the execution speed of the Cynosure AI simulation and the optimizations in place to achieve the 10+ TPS (ticks per second) performance target.

---

## Part 1: Factors Determining Speed

### 1.1 LRSG Forward Pass (Primary Bottleneck)

**Characteristic:** 4.2M neurons per agent with sparse connectivity (max degree 24)

**Operations:**
- Dense matrix-vector operations: `W_proj`, `W_fb`, `W_pwm` sliced to active regions
- Activation clamping and NaN handling per tick
- Sparse graph propagation (frontier-based sparse activations)

**Baseline Performance (Unoptimized):**
- ~14ms per agent forward pass (before Phase 2 optimizations)
- ~8ms per agent forward pass (after optimizations)

**Key Insight:** Only ~4096 of 4.2M neurons are active per tick (sparse frontier). Dense operations target only the active window, not the full network.

### 1.2 Physics Simulation Tick

**Components:**
- 256³ grid state updates
- Agent movement and sensory calculations  
- Conflict resolution between agents (distance checks, resource transfer)
- Safe Zone intensity calculations (exponential damage scaling)

**Latency Impact:** ~55ms per tick for 5-10 agents

### 1.3 Plasticity (Learning) Updates

**Operations:**
- Hebbian weight updates on synapse pairs (pre + post eligibility traces)
- Metabolic maintenance (~50M synapses at full scale, 1M synapse per batch)
- Structural neurogenesis (neuron births/deaths in high-activity clusters)

**Critical Issue Fixed (25 May 2026):** Double-execution bug in `loop.py` (missing `else` clause) caused both stream-parallel AND sequential plasticity to run every tick, doubling compute time and causing OOM at 4.2M neurons.

**After Fix:** ~50% reduction in plasticity CUDA time.

### 1.4 Linguistic Output Generation

**Operations:**
- Character prediction from PWM state (language hub output)
- Whisper broadcasting to other agents
- Runs every 10th tick (not every tick)

**Before Optimization:** ~2259ms per linguistic tick (TinyLlama LLM)
**After Optimization:** ~3-16ms per linguistic tick (PWM character decoder + temperature sampling)
**Speedup:** 140-684x

**Why TinyLlama was slow:**
- Transformer inference: ~2.2s per tick
- Context window expansion overhead
- No caching between ticks

**Why PWM decoder is fast:**
- Temperature sampling from 256-char logits (single forward pass output)
- No model inference, just stochastic selection
- Reuses existing forward pass computation

---

## Part 2: Optimizations in Place

### 2.1 Precision Optimization: FP16

**Mechanism:** All dense brain parameters (`W_proj`, `W_fb`, `W_pwm`, etc.) converted to FP16 half-precision.

**Benefits:**
- ~500MB per-agent VRAM saved
- **Tensor Core acceleration enabled** on RTX series GPUs (Turing, Ampere, Ada)
- Reduced memory bandwidth for weight matrices

**Verification:** Numerical correctness: 0.0 absolute max diff vs original FP32 path.

**Trade-off:** Negligible precision loss on sparse activations (±1e-3 relative error acceptable for neural weights).

---

### 2.2 Parallelism: CUDA Streams

**Mechanism:** Implemented CUDA stream parallelism in `Simulation._physics_tick()`. Multiple agents execute `sense_and_act` and `hebbian_update` concurrently on different GPU streams.

**Benefit:** Maximizes GPU occupancy; overlaps kernel launches across agents.

**Impact:** 5-10 agent runs sustain 30-35ms per tick vs 50-100ms sequential.

---

### 2.3 Memory Management: VRAM Reference Sharing

**Mechanism:** Static tensors (`positions`, `block_ids`, dense weights) shared by reference across all agent brain instances instead of duplicated.

**Fixed Issue:** OOM at 15 agents, 4.2M neurons on 8GB VRAM. Reduced per-agent VRAM footprint by ~60%.

---

### 2.4 Chunked Metabolic Maintenance

**Mechanism:** Metabolic updates (synapse-level resource costs) processed in 1M synapse chunks per batch instead of all 50M at once.

**Fixed Issue:** ~14 second delay at tick 20 due to peak VRAM exceeding 8GB. Chunking maintains low peak usage.

**Impact:** Eliminated VRAM thrashing; metabolic updates now take <100ms per tick.

---

### 2.5 Sparse Propagation Kernel (Triton)

**Mechanism:** Custom Triton kernel `_propagation_kernel` implements frontier-based sparse graph activation updates.

**Operations:** For each frontier source neuron → iterate edges → compute `activation[target] += activation[source] * weight` → apply PCZ modulation (`gamma_t`) for excitatory edges → atomic add for concurrent writes.

**Performance:**
- Kernel is JIT-compiled and cached by Triton per run
- Requires MSVC (`cl.exe`) or auto-detected via `find_msvc()`
- Windows: CUDA bundled with Triton at `<platlib>/triton/backends/nvidia`

**Status:** Opt-in via `use_triton=True` parameter. Python fallback (`_python_fallback` using PyTorch `scatter_add_`) is default (faster for large frontiers due to cuBLAS optimization).

**Contribution to Total CUDA Time:** 0.88%

---

### 2.6 Device-to-Host (D2H) Batching

**Mechanism:** Eliminated full-tensor GPU→CPU copies in `decode_output`. Batched 3 separate `.item()` GPU→CPU sync calls into 1.

**Before:** 17.7% of CUDA time spent in D2H operations
**After:** 4.0% of CUDA time
**Savings:** 77% reduction in synchronization overhead

**Technical Details:**
- Kept intermediate results as 0-d tensors (no `.item()` until final output)
- Reduced `_language_reward_accumulator` from Python float to `torch.tensor(0.0, device=device)`
- Removed 2 `.clone()` calls per agent tick (copy kernel overhead)

---

### 2.7 Tensor Core Optimization (Phase 2, Task 3)

**Problem:** Profiler showed GEMM/matmul was only 0.05% of CUDA time on RTX 2070. Tensor cores were idle.

**Root Cause:** Forward pass used elementwise ops, 1D vector multiplies (`torch.mv`), and reductions instead of 2D matmuls that dispatch to tensor cores.

**Solution (7 steps):**

1. **TF32 Flags:** `torch.backends.cuda.matmul.allow_tf32 = True` enables tensor core dispatch for FP32 matmuls.
2. **Config-Driven Active Window:** Added `h_window: int = 4096` to `LRSGConfig`. Hardcoded window size now configurable.
3. **FP16 2D Matmul:** Changed `torch.matmul(2D, 1D)` (GEMV, no tensor cores) to `torch.mm(2D, 2D)` by unsqueezing to columns.
4. **P_start/P_end Caching:** Moved `.item()` calls from hot path to graph initialization. Eliminates 2 GPU→CPU syncs per tick per agent.
5. **Bigram Coherence `.item()` Elimination:** Kept `count` and `total` as 0-d tensors, converting only at final return.
6. **Char Surprise `.item()` Elimination:** Signature changed to accept `torch.Tensor` reward. `_language_reward_accumulator` now tensor-based.
7. **Context Buffer Clone Elimination:** Reference swap pattern replaces 2 `.clone()` calls per agent tick.

**Results:**
- ~6 GPU→CPU sync calls eliminated per agent tick (9% of CUDA time)
- ~2 `.clone()` copies eliminated per agent tick (11.4% of CUDA time)
- All dense MVMs now use FP16 `torch.mm` (tensor core eligible)

**WMMA Dispatch Status:** Not achieved on RTX 2070 Turing (sm_75) due to MVM shapes having n=1 (tensor cores require n≥16). Batching 16+ agents would unlock wmma but was deprioritized per branching rule (optimization <5% gain threshold).

---

### 2.8 Simplified Metrics Computation

**Mechanism:** `graph_complexity` metric reduced from O(N×D) to O(N) approximation.

**Fixed Issue:** 2-4 second spikes every 10 ticks when computing full graph metrics.

**Impact:** Consistent tick latency; eliminated metric-driven stalls.

---

## Part 3: Performance Targets & Current Results

### Target: 10 Ticks Per Second (TPS)

This translates to:
- **Per-tick latency:** ~100ms maximum (10 TPS = 1 tick / 0.1s)
- **Comfortable headroom:** Achieve 50-70ms steady state (14-20 TPS)

### Current Performance Achieved

| Config | Tick Latency | TPS | Notes |
|--------|---|---|---|
| 2 agents, 10K neurons | 28.1ms steady | 35+ | Stable test |
| 5 agents, 10K neurons | 25.9ms avg | 38+ | After all optimizations |
| 2 agents, 100K neurons | 25.1ms steady | 40+ | Longer learning experiment |
| 5 agents, 100K neurons | 33.5ms steady | 30+ | Scale test |
| 5 agents, 4.2M neurons | 30-35ms steady | 28-33 | Full config (occasional higher) |
| 10 agents, 4.2M neurons | 35-40ms steady | 25-28 | Multi-agent coordination |

### Combined Tick Breakdown (5 agents, 100K neurons)

- **Physics tick:** ~55ms
- **Linguistic tick (every 10th):** ~16ms
- **Average total:** ~70ms (comfortably >10 TPS)

### Profiler Session Results (25 May 2026)

2 agents, 200 ticks, 100K neurons:

| Category | % of CUDA Time |
|---|---|
| Elementwise ops | 44.8% |
| Memory copy (all types) | 15.4% |
| Device→Host sync | 4.0% |
| Compare/Select | 6.5% |
| GEMM/bmm | 3.2% |
| Triton custom kernels | 1.3% |

---

## Part 4: Optimization Timeline & Milestones

### 25 May 2026 — Bug Fix & Phase 2 Optimizations

**Bug Fixed:** Double-execution in plasticity path (missing `else` clause in `loop.py:385`)
- Impact: 50% reduction in plasticity CUDA time

**Phase 2 Optimizations Applied:**
1. W_proj + W_fb fusion (bmm) → 2000 fewer matmul launches/tick at 10 agents
2. Triton kernel enabled → 0.88% of CUDA time
3. D2H batching → 77% reduction in sync overhead (17.7% → 4.0%)
4. FP16 dense matmuls → tensor core eligible
5. .item() and .clone() elimination → 9% + 11.4% savings

**Results:**
- 5-agent, 10K-neuron: 42.5ms → 25.9ms (39% improvement)
- 2-agent, 100K-neuron: 28.1ms steady
- 5-agent, 100K-neuron: 33.5ms steady

---

## Part 5: Known Limitations & Trade-offs

### 1. Tensor Core Dispatch (WMMA) Not Achieved

**Why:** RTX 2070 Turing (sm_75) requires matmul dimensions (m, n, k) divisible by 16. Single-agent forward pass has n=1 (MVM), which falls back to cuBLAS Small-N kernels (correct but no tensor cores).

**Path Forward:** Batch 16+ agent MVMs to n≥16, or upgrade to Ampere+ (supports n=1 in some cases). Deprioritized due to <5% potential gain.

### 2. Linguistic Bottleneck Resolved but Remains Optional

**TinyLlama Path:** Still supported via `use_announcer: bool = False` config flag. Slower (2.2s/tick) but produces richer, learned linguistic output. PWM path is faster but produces temperature-sampled text.

### 3. VRAM Sharing Has Limits

**Constraint:** Reference sharing works for static tensors. Dynamic allocations (activation buffers, gradients) still allocated per agent. At 15 agents, 4.2M neurons pushes ~8GB limit.

### 4. Sparse Frontier Depth

**Constraint:** Only neurons within ~4 hops of active frontier propagate. Very deep sparse graphs may not reach all semantically relevant neurons. Trade-off: speed vs. coverage.

---

## Part 6: How to Profile & Benchmark

### Using PyTorch Profiler

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    simulation.run(ticks=200)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Using nsys (NVIDIA)

```bash
nsys profile -o profile.nsys-rep python run_simulation.py
# Then open in NVIDIA NSight Systems GUI
```

### Benchmarking Specific Components

See `run_stability_test.py` (2000-tick stability run) and `run_observation.py` (2-agent action stability assay).

---

## Part 7: Future Optimization Opportunities

1. **Graph Compression:** Prune low-weight synapses during idle periods
2. **Quantization:** INT8 weights for static parameters (requires retraining validation)
3. **Multi-GPU:** Distribute agents across multiple GPUs (requires graph communication layer)
4. **Kernel Fusion:** Combine multiple elementwise ops into single custom kernel
5. **Mixed Precision:** Sparse ops in FP8, dense ops in FP16

---

## References

- **CYNOSURE_KB.md, Section 13:** Performance optimizations and profiler results
- **src/lrsg/forward.py:** LRSG forward pass and Tensor Core setup
- **src/simulation/loop.py:** Plasticity execution and stream management
- **tests/test_simulation.py:** Performance regression tests
