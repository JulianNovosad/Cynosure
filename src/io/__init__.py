"""
Hardware I/O abstraction layer.

Provides unified interface for simulation (Pybullet) and hardware (Raspberry Pi 5)
sensor input and motor output. Both implementations follow the IOInterface contract
to ensure identical behavior across platforms.
"""

from src.io.io_interface import IOInterface
from src.io.sim_io import SimIO
from src.io.rpi_io import RPIIO

__all__ = ['IOInterface', 'SimIO', 'RPIIO']
