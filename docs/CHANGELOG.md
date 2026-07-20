# Changelog

## Unreleased (as of 20 Jul 2026)

### Embodied Pet / Battleground Track

- **Embodied loop** (`src/simulation/embodied_loop.py`): Integrates physics, LRSG, and reward shaping in a single step loop for a single embodied agent (distinct from the multi-agent mainframe arena).
- **Battleground environment** (`src/simulation/battleground_env.py`): Procedural arena with enemy spawning and chase/flee/random-walk NPC behavior.
- **Combat system**: Attack resolution with strength scaling, integrated into the embodied reward.
- **64D semantic state encoder**: Maps observations + enemies + reward into a 64-dimensional semantic state.
- **Shaped reward**: Goal-seeking navigation reward plus combat/survival terms.
- **Evaluation suite** (`src/evaluation/evaluation_suite.py`): Locomotion, navigation, and transfer evaluation.
- **Real-time web visualization** (`src/simulation/embodied_visualizer.py`, `src/simulation/static/`, `src/simulation/templates/`): Live dashboard for the embodied loop.

### Cluster-Brain Track

- **Cluster runtime** (`src/cluster/`): New `cluster.py`, `packet.py`, `scheduler.py` — a distributed/clustered execution model for the brain, driven by `src/simulation/cluster_loop.py` and `src/experiments/run_cluster_navigation.py`.
- **Bug fixes** (commit `1046407`): Fixed dead-motor-output bug and embodied-loop/cluster-brain learning bugs; validated with real runs.
- **Episode-batched Hebbian reward + exploration noise** (commit `8629e78`): Reward broadcast batched per-episode instead of per-step; added exploration noise. See `CYNOSURE_KB.md` §20 for the full experimental writeup (net result: agent moves but wanders, no directed goal-seeking emerged yet).
- Uncommitted, unvalidated: reward-baseline subtraction in `cluster_loop.py` (see `CYNOSURE_KB.md` §20.4).

## v0.3.0 (25 May 2026)

### Phase 2 Performance Optimizations

- **Double-execution bug fix** (`src/simulation/loop.py`): Stream-parallel and sequential plasticity paths were both executing every tick due to missing `else:` clause. Added `else:` to make them mutually exclusive. ~50% reduction in plasticity CUDA time; 5-agent benchmark improved from 42.5ms → 25.9ms avg tick (39%).

- **W_proj+W_fb bmm fusion** (`src/lrsg/forward.py`): Stacked W_proj and W_fb weight slices and replaced two separate `torch.matmul` calls with a single `torch.bmm`. Reduces ~2000 kernel launches per tick at 10 agents. Verified 0.0 numerical error.

- **Triton kernel enabled** (`src/lrsg/triton_kernels.py`): Was silently falling back to Python `scatter_add_` path because `gamma_t` was passed as a tensor instead of a scalar. Fixed with `gamma_val = gamma_t.item() if isinstance(gamma_t, torch.Tensor) else gamma_t`. The Triton `_propagation_kernel` now handles edge propagation with atomic add.

- **Fused clamp+nan_to_num Triton kernel** (`src/lrsg/triton_kernels.py`): Custom `_fused_clamp_nan_kernel` replaces two separate full-tensor passes (`clamp_` + `nan_to_num_`) per propagation step. 50% fewer full-tensor passes; accounts for 0.41% of CUDA time.

- **D2H batching in decode_output** (`src/lrsg/io.py`): Removed `output.float().cpu()` full-tensor copy; batched three argmax `.item()` calls into a single synchronous `torch.stack(...).cpu()`. Memcpy DtoH reduced from 17.7% → 4.0% of CUDA time (77% reduction).

- **Tensor Core optimization** (`src/lrsg/forward.py`, `src/lrsg/graph.py`, `src/config.py`): Added TF32 flags, config-driven active window (`h_window`), FP16 2D matmul via `torch.mm` with unsqueezed columns, P_start/P_end caching, bigram coherence `.item()` elimination, char surprise `.item()` elimination, context buffer clone elimination via reference swap. ~6 `.item()` GPU→CPU syncs eliminated per agent tick (9% of CUDA time); ~2 `.clone()` copies eliminated per agent tick (11.4% of CUDA time).

### New Features

- **PWM Character Prediction** (`src/lrsg/forward.py`): Added `pwm_character_prediction()` head — maps PWM state to 256-char ASCII distribution. Hebbian update via `update_char_prediction_hebbian()`. Enables agents to model linguistic output internally.

- **LPH Fatigue System** (`src/lrsg/forward.py`): Added `apply_lph_fatigue()` and `update_lph_fatigue()` — adaptive fatigue subtraction from LPH activations to prevent repetitive character generation. Fatigue decay factor: 0.85.

- **Linguistic Curriculum** (`src/lrsg/io.py`): Added `validate_curriculum()` and `encode_whisper_to_indices()` — SVO structure validation with spaCy POS tagging. Stage-based curriculum progression (noun+verb, SVO, complex structures).

- **LRSG Persistence** (`src/simulation/loop.py`, `src/lrsg/graph.py`): Added `save_structure()`/`load_structure()` for caching synaptic templates to `base_graph.pt`. Enables shared structure across agents and faster startup. Template is generated on first run.

- **Aurore Interlock Integration** (`src/simulation/loop.py`): Added optional `InterlockController` import from `aurore.interlock_controller` for physical turret safety checks.

- **Action Space Alignment** (`src/simulation/loop.py`): Fixed `ACTION_NAMES` mapping — replaced `MERGE` with `HUNT`, `DEFEND` with `COLLECT`. Aligned with `agent.py` and KB documentation.

### Infrastructure

- **CI Pipeline** (`.github/workflows/ci.yml`): CPU-only GitHub Actions workflow. Runs `pytest -k "not cuda"` with Python 3.12+. CUDA-dependent tests marked `@pytest.mark.cuda`.

- **Documentation**: Added `CYNOSURE_KB.md` (comprehensive knowledge base), `CLAUDE.md` (Claude agent instructions), expanded `AGENTS.md` and `GEMINI.md`.

### Known Issues

- **Stale test assertions**: `test_graph_creation` expects `max_out_degree=4` (asserts `(10000, 4)` shape); `test_neuro_architectures` expects `k_expected = min(10000, N) = 500` edges per node. Default `max_out_degree` is now 12. These assertions need updating in the test files.

- **Stale base_graph.pt**: If `base_graph.pt` exists with a different `num_neurons` than the test/run config, graph initialization will fail with a shape mismatch error. Delete the stale file to regenerate.

- **Pre-existing API mismatches**: `test_agent_collect_ram` references a non-existent `Agent.collect_ram()` method; `test_agent_vision` passes wrong args to `Agent._compute_vision()`. These tests need updating to match current agent API.

