"""
Raspberry Pi 5 hardware I/O implementation (skeleton).

This module provides the interface for real Raspberry Pi 5 hardware but is
not yet functional—it serves as a placeholder for future deployment.
To use real hardware, deploy a sensor/motor server on the RPi 5 and
uncomment the HTTP client code below.
"""

from src.io.io_interface import IOInterface


class RPIIO(IOInterface):
    """I/O wrapper for Raspberry Pi 5 hardware (skeleton implementation)."""

    def __init__(self, agent_id=0, rpi_host='localhost', rpi_port=5000):
        """
        Initialize RPIIO.

        Args:
            agent_id: Agent ID (for multi-robot setups)
            rpi_host: Hostname/IP of RPi 5 sensor server
            rpi_port: Port of RPi 5 sensor server

        Note:
            To use real hardware:
            1. Deploy a sensor/motor server on Raspberry Pi 5 at rpi_host:rpi_port
            2. Uncomment the HTTP client code in read_sensors() and write_motors()
            3. Ensure the server exposes GET /sensors and POST /motors endpoints
            4. The sensor response should match the SimIO dict structure exactly
        """
        self.agent_id = agent_id
        self.rpi_host = rpi_host
        self.rpi_port = rpi_port
        self.connected = False

        print("\n" + "="*70)
        print("RPIIO: Skeleton Implementation (not connected to real hardware)")
        print("="*70)
        print(f"To deploy real hardware I/O:")
        print(f"  1. Deploy RPi5 sensor server on {rpi_host}:{rpi_port}")
        print(f"  2. Server must expose:")
        print(f"     - GET  /sensors → returns dict with vision/lidar/imu/etc.")
        print(f"     - POST /motors  → accepts {{action: int}} JSON")
        print(f"  3. Uncomment HTTP client code in rpi_io.py")
        print(f"  4. Then use: io = RPIIO(); sensors = io.read_sensors()")
        print("="*70 + "\n")

    def read_sensors(self) -> dict:
        """
        Read sensor data from Raspberry Pi 5 hardware.

        Communicates with a sensor server running on the RPi via HTTP GET.

        Returns:
            dict: Sensor readings (vision, lidar, imu, motor_feedback, ram, position, time)

        Raises:
            NotImplementedError: Until real hardware is deployed and HTTP client is enabled.

        Production code (uncomment and adjust for your deployment):
            import requests
            try:
                response = requests.get(
                    f'http://{self.rpi_host}:{self.rpi_port}/sensors',
                    timeout=0.5
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to read sensors from RPi: {e}")
        """
        raise NotImplementedError(
            "RPIIO requires real Raspberry Pi 5 hardware with sensor server.\n"
            "See src/io/rpi_io.py docstring for deployment instructions."
        )

    def write_motors(self, action: int) -> None:
        """
        Send motor command to Raspberry Pi 5 hardware.

        Communicates with a motor controller running on the RPi via HTTP POST.

        Args:
            action: Discrete action in range [0, 6]

        Raises:
            NotImplementedError: Until real hardware is deployed and HTTP client is enabled.

        Production code (uncomment and adjust for your deployment):
            import requests
            if not (0 <= action <= 6):
                raise ValueError(f"Action must be in range [0, 6], got {action}")
            try:
                requests.post(
                    f'http://{self.rpi_host}:{self.rpi_port}/motors',
                    json={'action': action},
                    timeout=0.1
                )
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to write motors to RPi: {e}")
        """
        raise NotImplementedError(
            "RPIIO requires real Raspberry Pi 5 hardware with motor server.\n"
            "See src/io/rpi_io.py docstring for deployment instructions."
        )

    def close(self) -> None:
        """Cleanup and disconnect from RPi (if connected)."""
        if self.connected:
            print(f"Disconnected from RPIIO at {self.rpi_host}:{self.rpi_port}")
            self.connected = False
