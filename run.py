#!/usr/bin/env python3
"""
Cynosure: Unified entry point for all experiments and simulations.

This is the canonical runtime path for all Cynosure tasks. Use this script
to run experiments, analyses, and utilities.

Usage:
    python run.py simulation       # Run standard simulation
    python run.py full             # Run full-scale simulation
    python run.py experiment       # Run baseline experiment
    python run.py calibration      # Run calibration training
    python run.py learning         # Run learning experiment
    python run.py analyze          # Analyze results
"""

import sys
import argparse
import subprocess
from pathlib import Path


def run_experiment(script_path: str) -> int:
    """Execute an experiment script as a subprocess."""
    result = subprocess.run([sys.executable, script_path], cwd=Path(__file__).parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Cynosure unified entry point")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Simulation commands
    subparsers.add_parser("simulation", help="Run standard simulation (2 agents, 200 ticks)")
    subparsers.add_parser("full", help="Run full-scale simulation (15 agents, 4.2M neurons)")
    subparsers.add_parser("experiment", help="Run baseline experiment (100K ticks)")
    subparsers.add_parser("demo", help="Run visual demonstration")
    subparsers.add_parser("observation", help="Run observation assay (2-agent stability test)")

    # Learning experiments
    subparsers.add_parser("learning", help="Run learning experiment")
    subparsers.add_parser("physics-learning", help="Run physics-based learning experiment")
    subparsers.add_parser("curriculum", help="Run linguistic curriculum")

    # Calibration
    subparsers.add_parser("calibration", help="Run PWM/language calibration")
    subparsers.add_parser("calibration-segmented", help="Run segmented calibration")

    # Analysis commands
    subparsers.add_parser("analyze", help="Analyze telemetry results")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Map commands to experiment scripts
    experiment_map = {
        "simulation": "src/experiments/run_simulation.py",
        "full": "src/experiments/run_full.py",
        "experiment": "src/experiments/run_experiment.py",
        "demo": "src/experiments/run_demo.py",
        "observation": "src/experiments/run_observation.py",
        "learning": "src/experiments/run_learning_experiment.py",
        "physics-learning": "src/experiments/run_physics_learning_experiment.py",
        "curriculum": "src/experiments/run_linguistic_curriculum.py",
        "calibration": "src/experiments/run_calibration.py",
        "calibration-segmented": "src/experiments/run_calibration_segmented.py",
        "analyze": "src/analysis/analyze_telemetry.py",
    }

    script = experiment_map.get(args.command)
    if script:
        return run_experiment(script)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
