from .security import SecurityAuditLog


METRIC_BOUNDS = {
    "survival": (0, 15),
    "ram_efficiency": (0.0, 1e6),
    "territory_control": (0.0, 1.0),
    "action_entropy": (0.0, 2.0),
    "graph_complexity": (0.0, 4.0),
    "rewire_rate": (0.0, 1e6),
    "modularity": (-1.0, 1.0),
    "strategy_divergence": (0.0, 2.0),
    "weight_drift": (0.0, 1e6),
}


class MetricValidator:
    def __init__(self, audit_log: SecurityAuditLog = None):
        self.audit_log = audit_log or SecurityAuditLog()

    def validate(self, metrics: dict, tick: int) -> dict:
        validated = {}
        for key, value in metrics.items():
            bounds = METRIC_BOUNDS.get(key)
            if bounds is None:
                validated[key] = value
                continue
            lo, hi = bounds
            if not isinstance(value, (int, float)):
                self.audit_log.log_metric_anomaly(
                    tick, key, value, "non-numeric type"
                )
                validated[key] = max(lo, min(hi, 0.0))
            elif value < lo or value > hi:
                clamped = max(lo, min(hi, value))
                self.audit_log.log_metric_anomaly(
                    tick, key, value, f"out of bounds [{lo}, {hi}] clamped to {clamped}"
                )
                validated[key] = clamped
            else:
                validated[key] = value
        return validated
