# Cynosure

Robotics/agent learning research codebase. Two active tracks share the same `src/` package:

1. **Mainframe arena** — 15 autonomous LRSG agents (Layerless Recurrent Sparse Graph brains,
   ~4.2M neurons each) competing for survival in a contracting 3D cyberspace grid. See
   `AGENTS.md` / `GEMINI.md` for the package layout and quick commands.
2. **Embodied pet / cluster-brain** — a single embodied agent in a physics + battleground
   simulation, with a distributed "cluster-brain" execution model. See
   `docs/EMBODIED_PET_ARCHITECTURE.md` and `CYNOSURE_KB.md` §20 for design and current
   experimental status.

## Where to start

- **`CYNOSURE_KB.md`** — single source of truth for architecture, script purposes, data
  schemas, reward logic, and known issues. Read this before making changes.
- **`CLAUDE.md`** — directory-structure conventions and agent instructions (also apply to
  any AI coding assistant working in this repo).
- **`docs/CHANGELOG.md`** — what changed and why, in commit order.
- **`docs/PERFORMANCE_GUIDE.md`** — execution speed, bottlenecks, and GPU optimizations.

## Project layout

See the "Directory Structure" section in `CLAUDE.md` for the enforced hierarchy
(`src/`, `docs/`, `plans/`, `experiment_results/`, `state/`, `tests/`, `calibration/`).
All new code, data, and documentation must follow it.

## Setup

```bash
pip install -r requirements.txt
```

CUDA is required for full-scale runs (target: RTX 2070 8GB VRAM); reduced-size CPU runs are
used in tests.

## Running things

- Mainframe arena entry points (`run_simulation.py`, `run_full.py`, `run_experiment.py`, …)
  are documented in `AGENTS.md`.
- Embodied/cluster experiment runners live in `src/experiments/` (e.g.
  `run_embodied_battleground.py`, `run_cluster_navigation.py`).

## Tests

```bash
pytest tests/
```
