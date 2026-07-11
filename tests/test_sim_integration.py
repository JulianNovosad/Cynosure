"""
Integration tests: ClusterSimulation (cluster_loop.py) wired to ClusterScheduler.

Verifies Step 5 of plans/rewrite.md:
  "Wire scheduler to simulation environment (Cynosure sim matches AuroreMkVII
   sensor/action formats)"
"""

import numpy as np
import pytest

from src.cluster.packet import SensorPacket, ActionPacket, CAMERA_FEATURE_DIM, SERVO_CHANNELS, MOTOR_CHANNELS
from src.cluster.scheduler import ClusterScheduler, build_default_scheduler
from src.simulation.navigation import NavigationEnvironment
from src.simulation.cluster_loop import SimEncoder, ClusterSimulation, EpisodeResult


# ---------------------------------------------------------------------------
# SimEncoder
# ---------------------------------------------------------------------------

class TestSimEncoder:
    def test_output_shape(self):
        enc = SimEncoder(seed=1)
        lidar = np.ones(8, dtype=np.float32) * 5.0
        features = enc.encode((1.0, 2.0, 0.0), (8.0, 8.0, 0.0), lidar, 10.0)
        assert features.shape == (CAMERA_FEATURE_DIM,)
        assert features.dtype == np.float16

    def test_output_dtype_float16(self):
        enc = SimEncoder(seed=0)
        lidar = np.zeros(8, dtype=np.float32)
        features = enc.encode((5.0, 5.0, 0.0), (5.0, 5.0, 0.0), lidar, 10.0)
        assert features.dtype == np.float16

    def test_output_in_tanh_range(self):
        enc = SimEncoder(seed=7)
        lidar = np.random.default_rng(7).uniform(0, 10, size=8).astype(np.float32)
        features = enc.encode((3.0, 3.0, 0.0), (7.0, 7.0, 0.0), lidar, 10.0)
        assert float(np.max(np.abs(features.astype(np.float32)))) <= 1.0 + 1e-3

    def test_frozen_across_instances(self):
        enc1 = SimEncoder(seed=42)
        enc2 = SimEncoder(seed=42)
        lidar = np.ones(8, dtype=np.float32)
        f1 = enc1.encode((1.0, 1.0, 0.0), (9.0, 9.0, 0.0), lidar, 10.0)
        f2 = enc2.encode((1.0, 1.0, 0.0), (9.0, 9.0, 0.0), lidar, 10.0)
        np.testing.assert_array_equal(f1, f2)

    def test_different_positions_produce_different_features(self):
        enc = SimEncoder(seed=42)
        lidar = np.ones(8, dtype=np.float32) * 5.0
        f1 = enc.encode((1.0, 1.0, 0.0), (9.0, 9.0, 0.0), lidar, 10.0)
        f2 = enc.encode((8.0, 8.0, 0.0), (1.0, 1.0, 0.0), lidar, 10.0)
        assert not np.array_equal(f1, f2)

    def test_short_lidar_padded_safely(self):
        enc = SimEncoder(seed=42)
        lidar = np.ones(4, dtype=np.float32) * 5.0  # fewer than 8 rays
        features = enc.encode((5.0, 5.0, 0.0), (8.0, 8.0, 0.0), lidar, 10.0)
        assert features.shape == (CAMERA_FEATURE_DIM,)


# ---------------------------------------------------------------------------
# ClusterSimulation._make_sensor_packet
# ---------------------------------------------------------------------------

class TestSensorPacketConstruction:
    def setup_method(self):
        sched = build_default_scheduler(n_clusters=4, n_neurons=2000, n_edges=20000, max_active=4, rng=np.random.default_rng(0))
        self.sim = ClusterSimulation(scheduler=sched, grid_size=10.0, n_obstacles=2, seed=1)
        nav = self.sim._nav
        start, goal = nav.reset()
        self.sim._agent_pos = np.array(start, dtype=np.float64)
        self.sim._goal_pos = np.array(goal, dtype=np.float64)
        self.sim._tick = 0

    def test_packet_is_sensor_packet(self):
        pkt = self.sim._make_sensor_packet()
        assert isinstance(pkt, SensorPacket)

    def test_camera_features_shape_and_dtype(self):
        pkt = self.sim._make_sensor_packet()
        assert pkt.camera_features.shape == (CAMERA_FEATURE_DIM,)
        assert pkt.camera_features.dtype == np.float16

    def test_rangefinder_is_scalar_float32(self):
        pkt = self.sim._make_sensor_packet()
        assert isinstance(float(pkt.rangefinder), float)
        assert pkt.rangefinder.dtype == np.float32

    def test_servo_positions_shape(self):
        pkt = self.sim._make_sensor_packet()
        assert pkt.servo_positions.shape == (SERVO_CHANNELS,)
        assert pkt.servo_positions.dtype == np.float16

    def test_tick_monotonic(self):
        self.sim._tick = 5
        pkt = self.sim._make_sensor_packet()
        assert int(pkt.tick) == 5

    def test_packet_activation_vector_valid(self):
        pkt = self.sim._make_sensor_packet()
        vec = pkt.to_activation_vector()
        assert vec.dtype == np.float16
        assert len(vec) >= CAMERA_FEATURE_DIM


# ---------------------------------------------------------------------------
# ClusterSimulation._apply_action
# ---------------------------------------------------------------------------

class TestActionApplication:
    def setup_method(self):
        sched = build_default_scheduler(n_clusters=4, n_neurons=2000, n_edges=20000, max_active=4, rng=np.random.default_rng(0))
        self.sim = ClusterSimulation(scheduler=sched, grid_size=10.0, n_obstacles=2, seed=2)
        self.sim._nav.reset()
        self.sim._agent_pos = np.array([5.0, 5.0, 0.1], dtype=np.float64)
        self.sim._goal_pos = np.array([8.0, 8.0, 0.1], dtype=np.float64)

    def test_stop_action_holds_position(self):
        pos_before = self.sim._agent_pos.copy()
        self.sim._apply_action(ActionPacket.stop())
        # With zero motor vector, vx=vy=0 → position unchanged
        np.testing.assert_array_almost_equal(self.sim._agent_pos[:2], pos_before[:2], decimal=4)

    def test_full_forward_moves_agent(self):
        mv = np.ones(MOTOR_CHANNELS, dtype=np.float16)
        action = ActionPacket(motor_vector=mv)
        pos_before = self.sim._agent_pos[:2].copy()
        self.sim._apply_action(action)
        # Agent should have moved
        delta = np.linalg.norm(self.sim._agent_pos[:2] - pos_before)
        assert delta > 0.0

    def test_position_stays_within_grid(self):
        # Drive into a corner
        self.sim._agent_pos = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        mv = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float16)
        action = ActionPacket(motor_vector=mv)
        for _ in range(50):
            self.sim._apply_action(action)
        ax, ay = self.sim._agent_pos[0], self.sim._agent_pos[1]
        assert 0.0 <= ax <= 10.0
        assert 0.0 <= ay <= 10.0


# ---------------------------------------------------------------------------
# ClusterSimulation.run_episode
# ---------------------------------------------------------------------------

class TestEpisodeRun:
    def setup_method(self):
        rng = np.random.default_rng(42)
        sched = build_default_scheduler(n_clusters=4, n_neurons=2000, n_edges=20000, max_active=4, rng=rng)
        self.sim = ClusterSimulation(scheduler=sched, grid_size=10.0, n_obstacles=2, seed=42)

    def test_episode_result_type(self):
        result = self.sim.run_episode()
        assert isinstance(result, EpisodeResult)

    def test_episode_increments(self):
        r1 = self.sim.run_episode()
        r2 = self.sim.run_episode()
        assert r2.episode == r1.episode + 1

    def test_steps_within_bounds(self):
        result = self.sim.run_episode()
        assert 1 <= result.steps <= ClusterSimulation.MAX_STEPS

    def test_reached_goal_consistent(self):
        result = self.sim.run_episode()
        if result.reached_goal:
            assert result.final_distance < ClusterSimulation.GOAL_RADIUS + 0.1  # small tolerance
        else:
            assert result.steps == ClusterSimulation.MAX_STEPS

    def test_total_reward_is_float(self):
        result = self.sim.run_episode()
        assert isinstance(result.total_reward, float)


# ---------------------------------------------------------------------------
# ClusterSimulation.run (batch)
# ---------------------------------------------------------------------------

class TestBatchRun:
    def setup_method(self):
        rng = np.random.default_rng(0)
        sched = build_default_scheduler(n_clusters=4, n_neurons=2000, n_edges=20000, max_active=4, rng=rng)
        self.sim = ClusterSimulation(scheduler=sched, grid_size=10.0, n_obstacles=2, seed=0)

    def test_run_returns_correct_count(self):
        results = self.sim.run(5)
        assert len(results) == 5

    def test_all_results_are_episode_results(self):
        results = self.sim.run(3)
        for r in results:
            assert isinstance(r, EpisodeResult)

    def test_episode_numbers_sequential(self):
        results = self.sim.run(4)
        for i, r in enumerate(results):
            assert r.episode == i + 1


# ---------------------------------------------------------------------------
# Full scheduler integration: step() produces ActionPacket from SensorPacket
# ---------------------------------------------------------------------------

class TestSchedulerIntegration:
    def test_step_returns_action_packet(self):
        sched = build_default_scheduler(n_clusters=4, n_neurons=2000, n_edges=20000, max_active=4)
        sim = ClusterSimulation(scheduler=sched, seed=1)
        nav = sim._nav
        start, goal = nav.reset()
        sim._agent_pos = np.array(start, dtype=np.float64)
        sim._goal_pos = np.array(goal, dtype=np.float64)
        sim._tick = 0

        pkt = sim._make_sensor_packet()
        action = sched.step(pkt, reward=0.0)
        assert isinstance(action, ActionPacket)
        assert action.motor_vector.shape == (MOTOR_CHANNELS,)
        assert action.motor_vector.dtype == np.float16

    def test_motor_vector_in_valid_range(self):
        sched = build_default_scheduler(n_clusters=4, n_neurons=2000, n_edges=20000, max_active=4)
        sim = ClusterSimulation(scheduler=sched, seed=2)
        nav = sim._nav
        start, goal = nav.reset()
        sim._agent_pos = np.array(start, dtype=np.float64)
        sim._goal_pos = np.array(goal, dtype=np.float64)
        sim._tick = 0

        for _ in range(10):
            sim._tick += 1
            pkt = sim._make_sensor_packet()
            action = sched.step(pkt, reward=0.1)
            assert np.all(action.motor_vector >= -1.0)
            assert np.all(action.motor_vector <= 1.0)

    def test_reward_modulates_weights(self):
        """Hebbian update changes cluster weights when reward != 0."""
        sched = build_default_scheduler(n_clusters=4, n_neurons=2000, n_edges=20000, max_active=4, rng=np.random.default_rng(99))
        sim = ClusterSimulation(scheduler=sched, seed=3)
        nav = sim._nav
        start, goal = nav.reset()
        sim._agent_pos = np.array(start, dtype=np.float64)
        sim._goal_pos = np.array(goal, dtype=np.float64)
        sim._tick = 0

        # Capture weight snapshot before
        first_cluster = list(sched._cluster_store.values())[0]
        weights_before = first_cluster.weight.copy()

        # Run a tick with positive reward to trigger Hebbian update
        sim._tick += 1
        pkt = sim._make_sensor_packet()
        sched.step(pkt, reward=1.0)

        # Run another tick to ensure the first cluster was activated
        for _ in range(20):
            sim._tick += 1
            pkt = sim._make_sensor_packet()
            sched.step(pkt, reward=1.0)

        weights_after = first_cluster.weight.copy()
        # Weights must have changed (Hebbian update applied)
        assert not np.array_equal(weights_before, weights_after), \
            "Hebbian update should change weights when reward != 0"
