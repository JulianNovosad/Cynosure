# Cynosure — Agent Guide

## KNOWLEDGE BASE

A unified knowledge base for all agents operating in this codebase is maintained at:

**`CYNOSURE_KB.md`** (project root)

Before taking any action in this codebase, search CYNOSURE_KB.md for relevant context on:
- System architecture and component roles
- Script purposes and execution entrypoints
- Data schemas and artifact locations
- Reward system logic and known issues
- Conflict/hook event classification
- Calibration and curriculum pipelines

If you discover information that contradicts or extends the knowledge base, append your findings to the relevant section and update the "Last updated" timestamp.

## Non-Negotiables

These rules are absolute. Never compromise on them for convenience, speed, or simplicity.

- **Performance is non-negotiable.** The simulation must hit its target tick rate. Do not accept slow loops, CPU fallbacks for compute-heavy paths, or unoptimized hot paths. Profile before and after every change.
- **CUDA acceleration is mandatory.** All LRSG forward passes, Hebbian updates, structural plasticity, and tensor-heavy operations must run on GPU. CPU fallback is only acceptable for tests using reduced neuron counts. Never ship a change that moves compute off the GPU without explicit approval.
- **Visual quality is non-negotiable.** The dashboard, visualizer, and ASCII map must render correctly and cleanly. Do not accept broken layouts, missing data, or degraded rendering. The frontend on localhost:8080 must remain functional during and after every change.
- **Containment safety is absolute.** The Cynosure mainframe containment boundaries must never be weakened. The `src/harness/` security layer (`SecurityAuditLog`, `AnnouncerSanitizer`, `WatchdogTimer`, `ResourceMonitor`) must remain intact. Never bypass, disable, or reduce security checks, sanitization, resource limits, or escape-detection logic. The `sanitizer.sanitize_broadcast()` call and all harness guards are mandatory.

## Quick Commands

```
pip install -r requirements.txt          # base deps (torch, numpy)
pip install transformers bitsandbytes    # needed for LLM announcer
pip install psutil networkx              # needed for simulation loop & metrics

pytest                                   # run all tests
python tests/test_agent.py               # run single test file (also works standalone)
```

## Architecture

- **Python 3.12+**, PyTorch, **CUDA required** for full runs (target: RTX 2070 8GB VRAM)
- **Scale**: Supports up to 15 autonomous agents in a 256³ grid (see `run_full.py`). Default configuration uses 10 agents in a 32³ grid for efficiency.
- **Dual-loop execution**: physics at 10 TPS, LLM announcer every 10 ticks (1 TPS)
- Each agent's "brain" is a **LRSG** (Layerless Recurrent Sparse Graph) — ~4.2M neurons, max out-degree 24, 5-step sparse forward pass per tick
- **Episodic Memory**: Recurrent autoencoder (Episodic Compressor) summarizes experience for broadcast.
- **Announcer** uses TinyLlama 1.1B (Q4 quantized via bitsandbytes) or falls back to random whispers when `use_llm_announcer=False`

## Package Layout

| Path | Purpose |
|------|---------|
| `src/config.py` | All config dataclasses: `Config`, `LRSGConfig`, `ArenaConfig`, `AgentConfig`, `SimulationConfig` |
| `src/lrsg/` | LRSG graph, forward pass, Hebbian learning, structural plasticity, I/O encoding |
| `src/arena/` | 3D environment, safe zone (contracting sphere), RAM nodes, collision detection |
| `src/agent/` | Agent class: sense, act, move, collect RAM, intrinsic reward, metabolic params |
| `src/simulation/loop.py` | Main `Simulation` class — orchestrates physics + linguistic ticks, metrics, checkpointing |
| `src/simulation/conflict.py` | Conflict resolution matrix (ATTACK/MERGE/DEFEND outcomes) |
| `src/announcer/` | LLM announcer with fallback mode |
| `src/harness/` | Security audit, telemetry, metrics validation, resource monitoring, checkpointing |
| `src/cluster/` | Cluster-brain runtime (`cluster.py`, `packet.py`, `scheduler.py`) — used by `src/simulation/cluster_loop.py` |
| `src/evaluation/` | Evaluation suite for locomotion, navigation, and transfer (`evaluation_suite.py`) |
| `src/simulation/embodied_loop.py` | Embodied pet loop: physics + LRSG + reward in a single step |
| `src/simulation/battleground_env.py` | Procedural battleground with enemy spawning/behavior and combat resolution |
| `src/simulation/embodied_visualizer.py` | Real-time web visualization for the embodied loop (`static/`, `templates/`) |
| `frontend/` | Web dashboard (served on localhost:8080 during simulation) |
| `tests/` | pytest tests (also runnable as `python tests/test_*.py`) |

## Simulation Economy

| Action | RAM Cost | Range | Description |
|--------|----------|-------|-------------|
| **ATTACK** | 2 | 3.0 | High-damage collision |
| **HUNT** | 1 | 8.0 | Stalk and steal RAM |
| **COLLECT** | 0 | 15.0 | Harvest RAM node |
| **MOVE** | 1 | N/A | Change coordinates |
| **WHISPER** | 1 | N/A | Broadcast message |
| **PROBE** | 0 | N/A | High-accuracy scan |
| **MANIPULATE** | 3 | N/A | Structural edit |

## Entry Points

| Script | Purpose |
|--------|---------|
| `run_simulation.py` | Quick 100-tick run, 15 agents, 4.2M neurons |
| `run_full.py` | Same as above with explicit config override |
| `run_experiment.py` | Exp 1: 100K-tick baseline calibration with LLM announcer enabled |
| `experiment_runner.py` | Thesis experiments: count ablation, time-series, reproducibility |
| `summary.py` | Analyze `telemetry.jsonl` output after a run |
| `analyze_run.py` | Lightweight telemetry analysis (survival, RAM, entropy) |

## Key Config Defaults (`src/config.py`)

- `num_neurons`: 4,194,304 | `max_out_degree`: 12 (default in src/config.py) | `propagation_steps`: 5
- `initial_ram`: 200 | `whisper_buffer_dim`: 48
- `num_input_neurons`: 1848 | `num_output_neurons`: 527
- `grid_size`: (32, 32, 32) | `num_agents`: 10 | `max_ticks`: 10,000
- `use_llm_announcer`: **False** by default (set True in `run_experiment.py`)
- Safe zone contracts every 100 ticks by fixed 1.0 unit (min radius 5.0)

## Test Conventions

- Tests use `sys.path.insert(0, ...)` for imports — no `__init__.py` manipulation needed
- Tests use reduced neuron counts (10,000) and small grids (50³) to run fast on CPU
- Each test file has a `if __name__ == "__main__"` block for standalone execution
- No `pytest.ini` or `pyproject.toml` — plain pytest with auto-discovery

## Gotchas

- **LLM announcer requires `transformers` + `bitsandbytes`** — not in `requirements.txt`
- Simulation starts a **dashboard HTTP server on port 8080** and a **visualizer** — both run in background threads
- VRAM-heavy: 15 agents × 4.2M neurons × 4 bytes ≈ 1GB just for edge weights
- Output files written to **repo root**: `telemetry.jsonl`, `conflicts.jsonl`, `security_audit.log`, `agent_whispers.jsonl`
- `psutil` and `networkx` are runtime imports in the simulation loop but not listed in requirements
- Tests that involve the full LRSG with 1848 input neurons require `vision_dims=1536` in config or they will fail on shape mismatch
- **Triton Windows CUDA setup:** `triton-windows` installs to user site-packages. `find_cuda_bundled()` checks `sysconfig.get_paths()["platlib"]` which may differ from the actual install path. The module `src/lrsg/triton_kernels.py` auto-sets `CUDA_PATH` env var on import to fix this. MSVC `cl.exe` is auto-detected by `find_msvc()`. Set `CC` env var if auto-detection fails.
