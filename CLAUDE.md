# CLAUDE.md

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

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This codebase appears to implement an advanced simulation and agent interaction system, possibly related to cognitive modeling or complex behavioral testing (e.g., Cynosure project). The core functionality seems distributed across modules simulating various components like agents, learning structures, and the overall environment dynamics.

## Development Commands
Since no explicit build/test command is given, I will infer standard Python development workflow commands from `requirements.txt` and test files.

### 1. Setup and Dependencies
*   **Installation:** Install dependencies listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
*   **Virtual Environment:** It is highly recommended to work within a virtual environment (e.g., using `venv` or `conda`).

### 2. Running Tests and Linting
*   **Running All Tests:** Use the standard Python testing framework, likely `pytest`, targeting the core modules:
    ```bash
    pytest tests/
    ```
*   **Testing Specific Components (Inferred):**
    *   **LRSG Component:** `pytest tests/test_lrsg.py`
    *   **Arena Component:** `pytest tests/test_arena.py`
    *   **Simulation Component:** `pytest tests/test_simulation.py`

### 3. Running Experiments (Core Logic)
The project provides several scripts in the root directory for running different types of experiments:
*   **General Experiment Runner:** This is likely the main entry point for a full run.
    ```bash
    python run_experiment.py [arguments]
    ```
*   **Curriculum Learning:** For structured, progressive testing.
    ```bash
    python run_linguistic_curriculum.py [arguments]
    ```
*   **Episodic Experimentation:** For running sequential sets of trials.
    ```bash
    python run_episodic_experiment.py [arguments]
    ```
*   **Full/Stability Test:** To run a comprehensive validation pass.
    ```bash
    python run_full.py [arguments]
    # Or for specific stability checks:
    python run_stability_test.py [arguments]
    ```

## Architecture & Core Components
The codebase is structured into several distinct packages and utility scripts, suggesting a separation of concerns between simulation logic, agent behavior, and data analysis.

### Primary Modules/Packages:
*   **`src/agent/`**: Contains the core logic for agents within the system. This module likely defines agent classes and their decision-making processes.
*   **`src/lrsg/`**: Appears to relate to a specific learning or state structure (LRSG). It handles complex state transitions or memory management critical to agent behavior.
*   **`src/arena/`**: Likely models the physical or conceptual environment where agents interact, managing entities like `ram_node.py`.
*   **`src/simulation/`**: Contains the main simulation loop logic, coordinating interactions between agents and the environment over time steps.
*   **`src/metrics/`**: Responsible for collecting, aggregating, and potentially visualizing performance metrics gathered during simulations.

### Core Interaction Flow (Big Picture):
1.  **Initialization:** An experiment runner (`run_experiment.py`, etc.) orchestrates the process, setting up initial states in the `arena`.
2.  **Simulation Loop:** The system enters a loop managed by `src/simulation/`. In each step:
    *   Agents interact with the environment defined in `src/arena/`.
    *   Agent state and learning are managed via mechanisms detailed in `src/lrsg/`.
    *   The agents execute actions, which are logged as events.
3.  **Data Capture & Analysis:** Results (metrics, logs) are periodically written to JSONL or other data files (`episodic_results.jsonl`, `telemetry.jsonl`). The system also has dedicated analysis scripts like `analyze*.py` for post-processing this raw data.

### Key Data Structures and Files:
*   **State Management:** Interactions likely involve objects from `src/arena/ram_node.py` and the logic in `src/lrsg/`.
*   **Results Storage:** Outputs are saved to directories like `experiment_results/` and files such as `detailed_metrics.jsonl`.
*   **Configuration/Logging:** Files like `conflicts.jsonl`, `security_audit.log`, and various `.md`/`.txt` files suggest checkpointing, logging, and documentation of experimental runs are integral parts of the workflow.

### File Paths Summary:
The most relevant operational code paths are:
*   **Agents/Behavior:** `src/agent/`
*   **Environment State:** `src/arena/`
*   **Simulation Core:** `src/simulation/`, `run_experiment.py`
*   **Learning Logic:** `src/lrsg/`