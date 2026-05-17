import time
import torch
import psutil
from .exceptions import ResourceExceededException
from .security import SecurityAuditLog


class ResourceMonitor:
    def __init__(
        self,
        vram_limit_gb: float = 7.0,
        ram_limit_gb: float = 16.0,
        audit_log: SecurityAuditLog = None,
    ):
        self.vram_limit = vram_limit_gb
        self.ram_limit = ram_limit_gb
        self.audit_log = audit_log or SecurityAuditLog()

    def check(self, tick: int):
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            if vram_used > self.vram_limit:
                self.audit_log.log_resource_violation(
                    tick, "vram", f"{vram_used:.2f}GB", f"{self.vram_limit:.2f}GB"
                )
                raise ResourceExceededException(
                    f"VRAM {vram_used:.2f}GB exceeds limit {self.vram_limit:.2f}GB"
                )

        ram_used = psutil.virtual_memory().used / 1e9
        if ram_used > self.ram_limit:
            self.audit_log.log_resource_violation(
                tick, "system_ram", f"{ram_used:.2f}GB", f"{self.ram_limit:.2f}GB"
            )
            raise ResourceExceededException(
                f"System RAM {ram_used:.2f}GB exceeds limit {self.ram_limit:.2f}GB"
            )


class WatchdogTimer:
    def __init__(self, threshold_ms: float = 200.0, audit_log: SecurityAuditLog = None):
        self.threshold = threshold_ms
        self.audit_log = audit_log or SecurityAuditLog()

    def check(self, elapsed_ms: float, tick: int, phase: str = "physics"):
        if elapsed_ms > self.threshold:
            self.audit_log.log_performance_warning(tick, phase, elapsed_ms)
            return True
        return False
