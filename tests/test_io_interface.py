"""
Tests for Hardware I/O Abstraction Layer.

Verifies that SimIO and RPIIO implement IOInterface correctly,
ensuring identical sensor/motor interfaces between simulation and hardware.
"""

import pytest
from src.io.io_interface import IOInterface
from src.io.sim_io import SimIO
from src.io.rpi_io import RPIIO


class TestIOInterface:
    """Test that IOInterface is properly defined."""

    def test_io_interface_is_abstract(self):
        """Verify IOInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            IOInterface()

    def test_io_interface_has_required_methods(self):
        """Verify IOInterface defines all required methods."""
        assert hasattr(IOInterface, 'read_sensors')
        assert hasattr(IOInterface, 'write_motors')
        assert hasattr(IOInterface, 'close')


class TestSimIO:
    """Test SimIO (Pybullet simulation) implementation."""

    def test_sim_io_instantiation(self):
        """Test that SimIO can be instantiated."""
        io = SimIO(agent_id=0)
        assert io is not None
        assert isinstance(io, IOInterface)
        io.close()

    def test_sim_io_implements_interface(self):
        """Verify SimIO implements IOInterface."""
        io = SimIO(agent_id=0)
        assert isinstance(io, IOInterface)
        assert hasattr(io, 'read_sensors')
        assert hasattr(io, 'write_motors')
        assert hasattr(io, 'close')
        io.close()

    def test_sim_io_reads_sensors(self):
        """Test that SimIO produces valid sensor dictionary."""
        io = SimIO(agent_id=0, use_sensor_noise=False)
        sensors = io.read_sensors()

        # Verify all required keys are present
        assert isinstance(sensors, dict)
        assert 'vision' in sensors
        assert 'lidar' in sensors
        assert 'imu' in sensors
        assert 'motor_feedback' in sensors
        assert 'ram' in sensors
        assert 'position' in sensors
        assert 'time' in sensors

        io.close()

    def test_sim_io_sensor_types(self):
        """Test that sensor data has correct types."""
        io = SimIO(agent_id=0, use_sensor_noise=False)
        sensors = io.read_sensors()

        # vision: list of objects (x, y, z, class_id, confidence)
        assert isinstance(sensors['vision'], list)

        # lidar: list of (angle, distance) tuples
        assert isinstance(sensors['lidar'], list)

        # imu: dict with accel, gyro, mag
        assert isinstance(sensors['imu'], dict)
        assert 'accel' in sensors['imu']
        assert 'gyro' in sensors['imu']
        assert 'mag' in sensors['imu']

        # motor_feedback: dict
        assert isinstance(sensors['motor_feedback'], dict)

        # ram: float
        assert isinstance(sensors['ram'], (int, float))

        # position: tuple or list of 3 elements
        assert len(sensors['position']) == 3

        # time: int (milliseconds)
        assert isinstance(sensors['time'], int)

        io.close()

    def test_sim_io_writes_all_actions(self):
        """Test that SimIO accepts all 7 discrete actions without crashing."""
        io = SimIO(agent_id=0)

        # Test all 7 actions
        for action in range(7):
            io.write_motors(action)  # Should not raise

        io.close()

    def test_sim_io_rejects_invalid_actions(self):
        """Test that SimIO rejects actions outside [0, 6]."""
        io = SimIO(agent_id=0)

        with pytest.raises(ValueError):
            io.write_motors(-1)

        with pytest.raises(ValueError):
            io.write_motors(7)

        with pytest.raises(ValueError):
            io.write_motors(100)

        io.close()

    def test_sim_io_with_sensor_noise(self):
        """Test that SimIO can apply sensor noise."""
        io = SimIO(agent_id=0, use_sensor_noise=True)
        sensors = io.read_sensors()

        assert isinstance(sensors, dict)
        assert 'vision' in sensors
        assert 'lidar' in sensors
        assert 'imu' in sensors

        io.close()

    def test_sim_io_without_sensor_noise(self):
        """Test that SimIO can operate without sensor noise."""
        io = SimIO(agent_id=0, use_sensor_noise=False)
        sensors = io.read_sensors()

        assert isinstance(sensors, dict)
        assert 'vision' in sensors
        assert 'lidar' in sensors
        assert 'imu' in sensors

        io.close()

    def test_sim_io_close_cleanup(self):
        """Test that SimIO.close() completes without error."""
        io = SimIO(agent_id=0)
        io.close()
        # After close, the Pybullet client is disconnected.
        # Verify close() itself doesn't raise exceptions.

    def test_sim_io_multiple_agents(self):
        """Test that SimIO can handle multiple agents."""
        # Create SimIO for agent 0
        io0 = SimIO(agent_id=0, num_agents=5)
        sensors0 = io0.read_sensors()
        assert sensors0 is not None

        # Create separate instance for agent 1 would create new physics world,
        # but we can verify that different agent_ids work within same instance
        io0.agent_id = 1
        sensors1 = io0.read_sensors()
        assert sensors1 is not None

        io0.close()

    def test_sim_io_sequential_read_write(self):
        """Test that SimIO can perform sequential sensor reads and motor writes."""
        io = SimIO(agent_id=0)

        for step in range(5):
            # Read sensors
            sensors = io.read_sensors()
            assert sensors is not None

            # Write motor command
            action = step % 7
            io.write_motors(action)

            # Verify state doesn't crash on next read
            next_sensors = io.read_sensors()
            assert next_sensors is not None

        io.close()


class TestRPIIO:
    """Test RPIIO (Raspberry Pi 5) skeleton implementation."""

    def test_rpi_io_instantiation(self):
        """Test that RPIIO can be instantiated."""
        io = RPIIO(agent_id=0)
        assert io is not None
        assert isinstance(io, IOInterface)
        io.close()

    def test_rpi_io_implements_interface(self):
        """Verify RPIIO implements IOInterface."""
        io = RPIIO(agent_id=0)
        assert isinstance(io, IOInterface)
        assert hasattr(io, 'read_sensors')
        assert hasattr(io, 'write_motors')
        assert hasattr(io, 'close')
        io.close()

    def test_rpi_io_read_sensors_not_implemented(self):
        """Test that RPIIO.read_sensors() raises NotImplementedError."""
        io = RPIIO(agent_id=0)
        with pytest.raises(NotImplementedError):
            io.read_sensors()
        io.close()

    def test_rpi_io_write_motors_not_implemented(self):
        """Test that RPIIO.write_motors() raises NotImplementedError."""
        io = RPIIO(agent_id=0)
        with pytest.raises(NotImplementedError):
            io.write_motors(0)
        io.close()

    def test_rpi_io_close_no_error(self):
        """Test that RPIIO.close() completes without error."""
        io = RPIIO(agent_id=0)
        io.close()  # Should not raise

    def test_rpi_io_custom_host_port(self):
        """Test that RPIIO accepts custom host and port."""
        io = RPIIO(agent_id=0, rpi_host='192.168.1.100', rpi_port=8080)
        assert io.rpi_host == '192.168.1.100'
        assert io.rpi_port == 8080
        io.close()


class TestInterfaceConsistency:
    """Test that both implementations follow the IOInterface contract."""

    def test_interface_method_signatures(self):
        """Verify all implementations have matching method signatures."""
        from inspect import signature

        # Get method signatures from interface
        interface_read_sig = signature(IOInterface.read_sensors)
        interface_write_sig = signature(IOInterface.write_motors)
        interface_close_sig = signature(IOInterface.close)

        # Verify SimIO matches
        sim_read_sig = signature(SimIO.read_sensors)
        sim_write_sig = signature(SimIO.write_motors)
        sim_close_sig = signature(SimIO.close)

        assert str(interface_read_sig) == str(sim_read_sig)
        assert str(interface_write_sig) == str(sim_write_sig)
        assert str(interface_close_sig) == str(sim_close_sig)

        # Verify RPIIO matches
        rpi_read_sig = signature(RPIIO.read_sensors)
        rpi_write_sig = signature(RPIIO.write_motors)
        rpi_close_sig = signature(RPIIO.close)

        assert str(interface_read_sig) == str(rpi_read_sig)
        assert str(interface_write_sig) == str(rpi_write_sig)
        assert str(interface_close_sig) == str(rpi_close_sig)

    def test_both_implementations_exist(self):
        """Verify both SimIO and RPIIO implementations exist."""
        sim_io = SimIO(agent_id=0)
        assert isinstance(sim_io, IOInterface)
        sim_io.close()

        rpi_io = RPIIO(agent_id=0)
        assert isinstance(rpi_io, IOInterface)
        rpi_io.close()

    def test_sensor_dict_structure_consistency(self):
        """Test that sensor dicts have consistent structure (for SimIO at least)."""
        io = SimIO(agent_id=0)
        sensors = io.read_sensors()

        # These keys must always be present
        required_keys = [
            'vision',
            'lidar',
            'imu',
            'motor_feedback',
            'ram',
            'position',
            'time',
        ]

        for key in required_keys:
            assert key in sensors, f"Missing key '{key}' in sensor dict"

        io.close()


class TestIOIntegration:
    """Integration tests for I/O abstraction layer."""

    def test_sim_io_continuous_operation(self):
        """Test that SimIO can operate continuously for multiple steps."""
        io = SimIO(agent_id=0)

        # Run 10 continuous steps
        for i in range(10):
            sensors = io.read_sensors()
            assert sensors is not None

            # Alternate between forward and stop
            action = 1 if i % 2 == 0 else 0
            io.write_motors(action)

        io.close()

    def test_sim_io_all_action_types(self):
        """Test that all 7 action types can be executed."""
        io = SimIO(agent_id=0)

        actions = [
            (0, 'STOP'),
            (1, 'MOVE_FORWARD'),
            (2, 'MOVE_BACKWARD'),
            (3, 'TURN_LEFT'),
            (4, 'TURN_RIGHT'),
            (5, 'ATTACK'),
            (6, 'IDLE'),
        ]

        for action_code, action_name in actions:
            io.write_motors(action_code)
            sensors = io.read_sensors()
            assert sensors is not None, f"Failed to read sensors after {action_name}"

        io.close()

    def test_polymorphism_sim_vs_rpi(self):
        """Test that SimIO and RPIIO can be used polymorphically through IOInterface."""
        def run_io_operation(io: IOInterface, num_steps: int = 3):
            """Generic function that works with any IOInterface implementation."""
            try:
                for i in range(num_steps):
                    sensors = io.read_sensors()
                    assert sensors is not None
                    io.write_motors(i % 7)
                return True
            except NotImplementedError:
                # RPIIO is expected to raise NotImplementedError
                return False

        # Test with SimIO (should succeed)
        sim_io = SimIO(agent_id=0)
        success = run_io_operation(sim_io, num_steps=3)
        assert success is True
        sim_io.close()

        # Test with RPIIO (should fail gracefully with NotImplementedError)
        rpi_io = RPIIO(agent_id=0)
        success = run_io_operation(rpi_io, num_steps=3)
        assert success is False
        rpi_io.close()
