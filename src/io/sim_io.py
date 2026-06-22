"""
Simulation I/O implementation using Pybullet physics engine.

Wraps the PhysicsWorld to provide IOInterface-compliant sensor/motor access.
"""

from src.io.io_interface import IOInterface
from src.simulation.physics_world import PhysicsWorld
from src.simulation.sensors import SensorSimulator


class SimIO(IOInterface):
    """I/O wrapper for Pybullet physics simulation."""

    def __init__(self, agent_id=0, use_sensor_noise=True, use_gui=False, num_agents=5):
        """
        Initialize SimIO.

        Args:
            agent_id: Which agent to read sensors from (0 to num_agents-1)
            use_sensor_noise: If True, apply realistic sensor noise
            use_gui: If True, visualize with Pybullet GUI (slow)
            num_agents: Total number of agents in the arena
        """
        self.agent_id = agent_id
        self.physics_world = PhysicsWorld(num_agents=num_agents, use_gui=use_gui)
        self.sensor_simulator = SensorSimulator() if use_sensor_noise else None
        self.last_action = 0

    def read_sensors(self) -> dict:
        """
        Read sensor data from Pybullet simulation.

        Returns:
            dict: Raw sensor readings from physics engine, optionally with noise applied
        """
        raw_sensors = self.physics_world.get_sensor_readings(self.agent_id)

        if self.sensor_simulator:
            return self.sensor_simulator.apply_noise(raw_sensors)
        return raw_sensors

    def write_motors(self, action: int) -> None:
        """
        Execute a motor command by stepping the physics simulation.

        Args:
            action: Discrete action (0-6)
        """
        if not (0 <= action <= 6):
            raise ValueError(f"Action must be in range [0, 6], got {action}")

        self.last_action = action
        # Step physics with action for this agent
        self.physics_world.step({self.agent_id: action})

    def close(self) -> None:
        """Disconnect from Pybullet and cleanup."""
        self.physics_world.close()
