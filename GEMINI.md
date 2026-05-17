# GEMINI.md - Project Cynosure Master Context & Instructions

## Project Overview
Project Cynosure is a sealed multi-agent simulation where 15 autonomous rogue AIs compete for survival in a collapsing 3D cyberspace arena (256³ grid). 

### Core Architecture: LRSG (Layerless Recurrent Sparse Graph)
- **Mind-Body Entity:** No distinction between mind and body; spatial coordinates represent the localization of recurrent activity.
- **Neural Scaling:** ~4.2M neurons per agent, sparse connectivity (max out-degree 4), 5-step sparse forward pass per tick.
- **Dual-Loop Execution:**
    - **Physics Loop (10 TPS):** Movement, RAM decay, 3D collisions, and Hebbian updates.
    - **Linguistic Loop (1 TPS):** The "Announcer" (LLM) injects semantic bias and translates arena state into global "whispers".

## 🛑 Non-Negotiable Mandates
These rules are absolute. Never compromise on them.

1.  **Performance is Non-Negotiable:** The simulation must hit target tick rates. No slow loops or unoptimized hot paths. Profile before and after changes.
2.  **CUDA Acceleration is Mandatory:** All LRSG forward passes, Hebbian updates, structural plasticity, and tensor-heavy operations **must** run on GPU. CPU fallback is only for low-neuron tests.
3.  **Containment Safety is Absolute:** Never weaken containment boundaries. The `src/harness/` security layer (`SecurityAuditLog`, `AnnouncerSanitizer`, `WatchdogTimer`, `ResourceMonitor`) must remain intact. Never bypass or reduce security checks, sanitization, or resource limits.
4.  **Visual Quality is Non-Negotiable:** The dashboard (localhost:8080), visualizer, and ASCII map must remain functional and render correctly.

## Package Layout
| Path | Purpose |
|------|---------|
| `src/config.py` | Configuration dataclasses: `Config`, `LRSGConfig`, `ArenaConfig`, etc. |
| `src/lrsg/` | LRSG graph logic, forward pass, Hebbian learning, structural plasticity. |
| `src/arena/` | 3D environment, safe zone (contracting sphere), RAM nodes, collisions. |
| `src/agent/` | Agent class: sensing, acting, metabolic parameters, intrinsic rewards. |
| `src/simulation/loop.py` | Main `Simulation` class: orchestrates physics and linguistic ticks. |
| `src/simulation/conflict.py` | Conflict resolution matrix (ATTACK/MERGE/DEFEND outcomes). |
| `src/announcer/` | LLM announcer (TinyLlama 1.1B Q4) with fallback mode. |
| `src/harness/` | Security audit, telemetry, resource monitoring, checkpointing. |
| `frontend/` | Web dashboard served on localhost:8080 during simulation. |
| `tests/` | Pytest suite (runnable standalone via `python tests/test_*.py`). |

## Building and Running

### Setup & Dependencies
```bash
pip install -r requirements.txt          # Base deps: torch, numpy, waitress
pip install transformers bitsandbytes    # Required for LLM announcer
pip install psutil networkx              # Required for simulation loop & metrics
```
*Note: CUDA support is required for full runs (target: RTX 2070 8GB VRAM).*

### Entry Points
| Script | Purpose |
|--------|---------|
| `run_simulation.py` | Standard 100-tick run, 15 agents, 4.2M neurons. |
| `run_full.py` | Same as above with explicit config override. |
| `run_experiment.py` | Exp 1: 100K-tick baseline calibration (LLM announcer enabled). |
| `summary.py` | Analyze `telemetry.jsonl` output. |
| `analyze_run.py` | Lightweight telemetry analysis (survival, RAM, entropy). |

## Development & Testing Conventions

### Configuration Defaults (`src/config.py`)
- `num_neurons`: 4,194,304 | `max_out_degree`: 4 | `propagation_steps`: 5.
- `grid_size`: (256, 256, 256) | `num_agents`: 10 | `max_ticks`: 10,000.
- `use_llm_announcer`: **False** by default (must be enabled explicitly).
- Safe zone contracts every 50 ticks by 5.0 units.

### Testing
- Tests use reduced neuron counts (10,000) and small grids (50³) for CPU compatibility.
- **Requirement:** Tests involving the full LRSG with 1848 input neurons require `vision_dims=1536` in the config to avoid shape mismatches.
- Run all tests: `pytest`.

## Gotchas & Troubleshooting
- **Missing Dependencies:** `transformers`, `bitsandbytes`, `psutil`, and `networkx` are often required at runtime but might be missing from the base `requirements.txt`.
- **VRAM Usage:** 15 agents × 4.2M neurons consumes ~1GB VRAM for edge weights alone.
- **Output Files:** `telemetry.jsonl`, `conflicts.jsonl`, and `security_audit.log` are written to the repo root.
- **Background Processes:** Simulation starts a dashboard (port 8080) and a visualizer in background threads.
