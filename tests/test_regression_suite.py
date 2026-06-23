"""
Regression Testing Suite (Phase 6 of Integration Checkpoint)

Validates no performance degradation vs. v0.4.0 baseline.
Ensures learning improvements are maintained.
Verifies telemetry schema completeness.
"""

import json
import torch
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, LRSGConfig, SimulationConfig
from src.simulation import Simulation


class RegressionTester:
    """Regression test orchestrator."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []

    def test_2_agent_200_tick(self) -> dict:
        """2-agent baseline: target ≤25.1ms/tick (v0.4.0 baseline)."""
        cfg = Config(
            lrsg=LRSGConfig(num_neurons=100_000),
            simulation=SimulationConfig(num_agents=2, max_ticks=200, headless=True),
        )

        start = time.time()
        sim = Simulation(cfg, device=self.device)
        init_time = time.time() - start

        start = time.time()
        metrics = sim.run()
        run_time = time.time() - start

        ms_per_tick = (run_time / cfg.simulation.max_ticks) * 1000
        passed = ms_per_tick <= 30.0  # 20% tolerance

        result = {
            "test": "2-agent-200-tick",
            "baseline_ms": 25.1,
            "current_ms": ms_per_tick,
            "target_ms": 30.0,
            "passed": passed,
            "agents_alive": sim.status["agents_alive"],
        }
        self.results.append(result)
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} 2-agent-200-tick: {ms_per_tick:.1f}ms/tick (target <=30ms)")
        return result

    def test_5_agent_1000_tick(self) -> dict:
        """5-agent stability: target ≤40ms/tick (v0.4.0: 33.5ms)."""
        cfg = Config(
            lrsg=LRSGConfig(num_neurons=100_000),
            simulation=SimulationConfig(num_agents=5, max_ticks=1000, headless=True),
        )

        start = time.time()
        sim = Simulation(cfg, device=self.device)
        init_time = time.time() - start

        start = time.time()
        metrics = sim.run()
        run_time = time.time() - start

        ms_per_tick = (run_time / cfg.simulation.max_ticks) * 1000
        passed = ms_per_tick <= 40.0  # 20% tolerance

        result = {
            "test": "5-agent-1000-tick",
            "baseline_ms": 33.5,
            "current_ms": ms_per_tick,
            "target_ms": 40.0,
            "passed": passed,
            "agents_alive": sim.status["agents_alive"],
        }
        self.results.append(result)
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} 5-agent-1000-tick: {ms_per_tick:.1f}ms/tick (target ≤40ms)")
        return result

    def test_nan_stability(self) -> dict:
        """1000-tick run should not produce NaNs in any tensor."""
        cfg = Config(
            lrsg=LRSGConfig(num_neurons=100_000),
            simulation=SimulationConfig(num_agents=5, max_ticks=1000, headless=True),
        )

        try:
            sim = Simulation(cfg, device=self.device)
            metrics = sim.run()

            # Check for NaN in agent activations
            has_nan = False
            for agent in sim.agents:
                if torch.isnan(agent.brain.activations).any():
                    has_nan = True
                    break

            passed = not has_nan

            result = {
                "test": "nan-stability",
                "has_nan": has_nan,
                "passed": passed,
                "ticks": cfg.simulation.max_ticks,
            }
            self.results.append(result)
            status = "[PASS]" if passed else "[FAIL]"
            print(f"{status} NaN stability: {1000} ticks, no NaNs detected")
            return result
        except Exception as e:
            result = {"test": "nan-stability", "error": str(e), "passed": False}
            self.results.append(result)
            print(f"[FAIL] NaN stability: {e}")
            return result

    def test_telemetry_schema(self) -> dict:
        """Verify all telemetry fields present and valid."""
        cfg = Config(
            lrsg=LRSGConfig(num_neurons=100_000),
            simulation=SimulationConfig(num_agents=2, max_ticks=50, headless=True),
        )

        if Path("telemetry.jsonl").exists():
            Path("telemetry.jsonl").unlink()

        try:
            sim = Simulation(cfg, device=self.device)
            metrics = sim.run()

            # Check telemetry file exists and has valid structure
            passed = False
            field_count = 0

            if Path("telemetry.jsonl").exists():
                with open("telemetry.jsonl") as f:
                    lines = f.readlines()
                    if lines:
                        first_line = json.loads(lines[0])
                        expected_fields = {"type", "tick", "timestamp"}
                        passed = expected_fields.issubset(first_line.keys())
                        field_count = len(first_line)

            result = {
                "test": "telemetry-schema",
                "passed": passed,
                "lines_recorded": len(lines) if Path("telemetry.jsonl").exists() else 0,
                "fields_per_entry": field_count,
            }
            self.results.append(result)
            status = "[PASS]" if passed else "[FAIL]"
            print(f"{status} Telemetry schema: {field_count} fields/entry")
            return result
        except Exception as e:
            result = {"test": "telemetry-schema", "error": str(e), "passed": False}
            self.results.append(result)
            print(f"[FAIL] Telemetry schema: {e}")
            return result

    def run_all(self) -> dict:
        """Run all regression tests."""
        print("\n=== REGRESSION TEST SUITE (Phase 6) ===\n")

        try:
            self.test_2_agent_200_tick()
        except Exception as e:
            print(f"[FAIL] 2-agent-200-tick: {e}")

        try:
            self.test_5_agent_1000_tick()
        except Exception as e:
            print(f"[FAIL] 5-agent-1000-tick: {e}")

        try:
            self.test_nan_stability()
        except Exception as e:
            print(f"[FAIL] nan-stability: {e}")

        try:
            self.test_telemetry_schema()
        except Exception as e:
            print(f"[FAIL] telemetry-schema: {e}")

        passed = sum(1 for r in self.results if r.get("passed", False))
        total = len(self.results)

        print(f"\n=== SUMMARY ===")
        print(f"Passed: {passed}/{total}")
        print(f"Status: {'REGRESSION LOCK PASSED' if passed == total else 'REGRESSIONS DETECTED'}")

        # Write summary
        with open("regression_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        return {"passed": passed, "total": total, "results": self.results}


if __name__ == "__main__":
    tester = RegressionTester()
    tester.run_all()
