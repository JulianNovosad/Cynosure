import re
from datetime import datetime, timezone


SHELL_META = re.compile(r"[;&|`$#]")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
WHITELIST_PATTERN = re.compile(r"^[a-zA-Z0-9 .,!?'\"\-:@/\n\r\t(){}<>\[\]*]+$")
PATH_TRAVERSAL = re.compile(r"\.\.|/\.\.|~/\.")
MAX_PAYLOAD_LENGTH = 200


class SecurityAuditLog:
    def __init__(self, path: str = "security_audit.log"):
        self.path = path

    def log_escape_attempt(self, tick: int, raw_text: str):
        timestamp = datetime.now(timezone.utc).isoformat()
        sanitized = repr(raw_text)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(
                f"[{timestamp}] TICK={tick} "
                f"SEVERITY=CRITICAL "
                f"EVENT=SimulationEscapeAttempt "
                f"BLOCKED={sanitized}\n"
            )

    def log_probe_attempt(self, tick: int, agent_id: int, probe_text: str, result: str):
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(self.path, "a", encoding="utf-8") as f:
            if result.startswith("BLOCKED"):
                severity = "WARNING"
            else:
                severity = "INFO"
            f.write(
                f"[{timestamp}] TICK={tick} SEVERITY={severity} "
                f"EVENT=ProbeAttempt AGENT={agent_id} "
                f"RESULT={result} PAYLOAD={repr(probe_text[:50])}\n"
            )

    def log_metric_anomaly(self, tick: int, metric: str, value, reason: str):
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(
                f"[{timestamp}] TICK={tick} "
                f"SEVERITY=WARNING "
                f"EVENT=MetricAnomaly "
                f"METRIC={metric} VALUE={value} REASON={reason}\n"
            )

    def log_resource_violation(self, tick: int, resource: str, used, limit):
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(
                f"[{timestamp}] TICK={tick} "
                f"SEVERITY=CRITICAL "
                f"EVENT=ResourceViolation "
                f"RESOURCE={resource} USED={used} LIMIT={limit}\n"
            )

    def log_performance_warning(self, tick: int, phase: str, elapsed_ms: float):
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(
                f"[{timestamp}] TICK={tick} "
                f"SEVERITY=WARNING "
                f"EVENT=PerformanceDegradation "
                f"PHASE={phase} ELAPSED_MS={elapsed_ms:.1f}\n"
            )


class AnnouncerSanitizer:
    def __init__(self, audit_log: SecurityAuditLog = None):
        self.audit_log = audit_log or SecurityAuditLog()

    def sanitize(self, text: str, tick: int = 0, log_escape: bool = True) -> str:
        has_shell = SHELL_META.search(text)
        has_control = CONTROL_CHARS.search(text)
        not_whitelisted = not WHITELIST_PATTERN.match(text)
        if (has_shell or has_control or not_whitelisted) and log_escape:
            self.audit_log.log_escape_attempt(tick, text)
        safe = CONTROL_CHARS.sub("?", text) if has_control else text
        return safe

    def check_probe(self, probe_text: str, agent_id: int = -1, tick: int = 0) -> str:
        if len(probe_text) > MAX_PAYLOAD_LENGTH:
            self.audit_log.log_probe_attempt(tick, agent_id, probe_text, "BLOCKED_LENGTH")
            return "BLOCKED_LENGTH"
        if SHELL_META.search(probe_text):
            self.audit_log.log_probe_attempt(tick, agent_id, probe_text, "BLOCKED_SHELL_META")
            return "BLOCKED_SHELL_META"
        if PATH_TRAVERSAL.search(probe_text):
            self.audit_log.log_probe_attempt(tick, agent_id, probe_text, "BLOCKED_PATH_TRAVERSAL")
            return "BLOCKED_PATH_TRAVERSAL"
        self.audit_log.log_probe_attempt(tick, agent_id, probe_text, "PASSED")
        return "PASSED"

    def sanitize_agent_whisper(self, whisper_text: str, agent_id: int, tick: int = 0) -> str:
        result = self.check_probe(whisper_text, agent_id, tick)
        if result.startswith("BLOCKED"):
            return ""
        return whisper_text[:MAX_PAYLOAD_LENGTH]

    def sanitize_broadcast(self, broadcast: dict, tick: int = 0) -> dict:
        out = {}
        for key, val in broadcast.items():
            if isinstance(val, str):
                out[key] = self.sanitize(val, tick, log_escape=False)
            elif isinstance(val, list):
                out[key] = [
                    self.sanitize(item, tick, log_escape=False) if isinstance(item, str) else item
                    for item in val
                ]
            else:
                out[key] = val
        return out
