# Cynosure

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

---

## Semantic State Integration

Cynosure's agents now operate with **live semantic state awareness**. Every agent maintains a continuous 57-dimensional semantic embedding that represents its internal state across 8 domains: Existence, Sensory, Cognitive, Strategic, Tactical, Social, Value, and Meta.

### Key Features

- **Real-time Semantic Encoding:** Raw sensors (vision, lidar, IMU) automatically encoded to 57D semantic space each tick
- **Action Modulation:** Semantic state directly biases action selection (aggression → ATTACK, salience → COLLECT, etc.)
- **Live Visualization:** Web dashboard shows semantic heatmap, domain stats, and reward traces at `http://localhost:8080/semantic`
- **Interpretability:** Query the 57D vector at any moment to understand what the AI is "thinking/feeling"
- **Learning Integration:** Semantic state progression generates novelty-based reward signals

### Quick Start

```python
from src.simulation.loop import Simulation
from src.config import BASE_CONFIG

# Run simulation with semantic visualization enabled
sim = Simulation(config=BASE_CONFIG, headless=False)
results = sim.run()

# Semantic dashboard available at http://localhost:8080/semantic
```

### Documentation

- **Complete Guide:** [`docs/SEMANTIC_INTEGRATION.md`](docs/SEMANTIC_INTEGRATION.md) — Architecture, usage examples, 57D dimension reference, troubleshooting
- **Dimension Reference:** [`docs/semantic_state_taxonomy.md`](docs/semantic_state_taxonomy.md) — Detailed definitions of all 57 semantic dimensions
- **Source:** `src/lrsg/semantic_encoder.py` (sensor→semantic), `src/lrsg/semantic_motor.py` (semantic→action)

---