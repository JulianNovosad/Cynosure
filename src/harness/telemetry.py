import json
import os
import threading
import time
from pathlib import Path


import time


def touch_file(path: str):
    try:
        with open(path, "a", encoding="utf-8") as f:
            pass
    except PermissionError:
        time.sleep(0.5)
        with open(path, "a", encoding="utf-8") as f:
            pass


def _write_with_retry(path: str, record: dict, max_retries: int = 20, delay: float = 0.1):
    line = json.dumps(record) + "\n"
    for attempt in range(max_retries):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
            return
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


class TelemetryLogger:
    def __init__(self, path: str = "telemetry.jsonl", conflict_path: str = "conflicts.jsonl"):
        self.path = path
        self.conflict_path = conflict_path

    def write_metrics(self, tick: int, metrics: dict):
        _write_with_retry(self.path, {"tick": tick, "type": "metrics", "data": metrics})

    def write_conflict(self, tick: int, agent_a: int, agent_b: int, outcome: str):
        _write_with_retry(self.conflict_path, {
            "tick": tick, "type": "conflict",
            "agent_a": agent_a, "agent_b": agent_b, "outcome": outcome,
        })

    def write_perf(self, tick: int, phase: str, elapsed_ms: float):
        _write_with_retry(self.path, {
            "tick": tick, "type": "perf", "phase": phase, "elapsed_ms": elapsed_ms,
        })

    def write_checkpoint(self, tick: int, path: str):
        _write_with_retry(self.path, {"tick": tick, "type": "checkpoint", "path": path})

    def write_announcer(self, tick: int, broadcast: dict):
        _write_with_retry(self.path, {
            "tick": tick, "type": "announcer",
            "status_text": broadcast.get("summary", {}).get("status", ""),
            "whispers": broadcast.get("whispers", []),
            "raw_summary": broadcast.get("summary", {}),
        })

    def write_resource_usage(self, tick: int, vram_gb: float, ram_gb: float):
        _write_with_retry(self.path, {
            "tick": tick, "type": "resource",
            "vram_gb": round(vram_gb, 4), "ram_gb": round(ram_gb, 2),
        })

    def write_agent_brain(self, tick: int, agent_id: int, num_edges: int, num_neurons: int,
                          avg_weight: float, active_neurons: int, births: int, deaths: int):
        _write_with_retry(self.path, {
            "tick": tick, "type": "agent_brain",
            "agent_id": agent_id,
            "num_edges": num_edges,
            "num_neurons": num_neurons,
            "avg_weight": round(avg_weight, 6),
            "active_neurons": active_neurons,
            "births": births,
            "deaths": deaths,
        })

    def write_action(self, tick: int, agent_id: int, action_index: int, action_name: str, ram_after: float):
        _write_with_retry(self.path, {
            "tick": tick, "type": "action",
            "agent_id": agent_id,
            "action_index": action_index,
            "action_name": action_name,
            "ram_after": round(ram_after, 2),
        })


class Dashboard:
    def __init__(
        self,
        telemetry_path: str = "telemetry.jsonl",
        audit_path: str = "security_audit.log",
        refresh_interval: float = 1.0,
    ):
        self.telemetry_path = telemetry_path
        self.audit_path = audit_path
        self.refresh_interval = refresh_interval
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            try:
                self._render()
            except Exception:
                # Silently ignore rendering errors to keep the simulation stable
                pass
            time.sleep(self.refresh_interval)

    def _render(self):
        lines = []
        lines.append("=" * 60)
        lines.append("CYNOSURE DASHBOARD")
        lines.append("=" * 60)

        if os.path.exists(self.audit_path):
            try:
                with open(self.audit_path, "r", encoding="utf-8") as f:
                    audit_entries = f.readlines()
                escape_count = sum(
                    1 for e in audit_entries if "SimulationEscapeAttempt" in e
                )
                resource_count = sum(
                    1 for e in audit_entries if "ResourceViolation" in e
                )
                if escape_count > 0:
                    lines.append(f"!! ALERT: {escape_count} ESCAPE ATTEMPT(S) DETECTED !!")
                if resource_count > 0:
                    lines.append(f"!! ALERT: {resource_count} RESOURCE VIOLATION(S) !!")
                recent = audit_entries[-3:] if len(audit_entries) >= 3 else audit_entries
                for entry in recent:
                    lines.append(f"  AUDIT: {entry.strip()}")
            except (PermissionError, IOError):
                pass

        if os.path.exists(self.telemetry_path):
            telemetry_lines = []
            try:
                with open(self.telemetry_path, "r", encoding="utf-8") as f:
                    # Read only last few lines to avoid blocking too long
                    f.seek(0, os.SEEK_END)
                    pos = f.tell()
                    f.seek(max(0, pos - 10000), os.SEEK_SET)
                    telemetry_lines = f.readlines()
            except (PermissionError, IOError):
                pass
            
            latest_metrics = None
            latest_perf = None
            for line in reversed(telemetry_lines):
                try:
                    rec = json.loads(line)
                    if rec["type"] == "metrics" and latest_metrics is None:
                        latest_metrics = rec
                    if rec["type"] == "perf" and latest_perf is None:
                        latest_perf = rec
                except (json.JSONDecodeError, KeyError):
                    pass
                if latest_metrics is not None and latest_perf is not None:
                    break
            if latest_metrics:
                lines.append(f"  Tick: {latest_metrics['tick']}")
                for k, v in latest_metrics["data"].items():
                    lines.append(f"    {k}: {v}")
            if latest_perf:
                lines.append(f"  Perf: {latest_perf['phase']} = {latest_perf['elapsed_ms']:.1f}ms")

        print("\n".join(lines))
