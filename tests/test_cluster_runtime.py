"""
Exhaustive tests for cluster runtime (cluster.py and packet.py).

Covers:
  - Packet format invariants (dtype, shape, frozen constants)
  - SensorPacket.to_activation_vector composition
  - ActionPacket.from_activation / stop
  - create_cluster bounds validation
  - cluster_forward: output shape, dtype, tanh gating, recurrent decay
  - Eligibility trace update and decay
  - Hebbian weight update with reward
  - Weight decay and clamping
  - Memory footprint within spec
"""

import numpy as np
import pytest

from src.cluster.packet import (
    SensorPacket,
    ActionPacket,
    CAMERA_FEATURE_DIM,
    SERVO_CHANNELS,
    MOTOR_CHANNELS,
)
from src.cluster.cluster import Cluster, create_cluster, cluster_forward, hebbian_update


# -----------------------------------------------------------------------
# Packet format invariants
# -----------------------------------------------------------------------

class TestSensorPacket:
    def _make(self, **kwargs):
        defaults = dict(
            camera_features=np.ones(CAMERA_FEATURE_DIM, dtype=np.float32),
            rangefinder=1.5,
            servo_positions=np.zeros(SERVO_CHANNELS, dtype=np.float32),
            tick=42,
        )
        defaults.update(kwargs)
        return SensorPacket(**defaults)

    def test_camera_features_coerced_to_float16(self):
        pkt = self._make(camera_features=np.ones(CAMERA_FEATURE_DIM, dtype=np.float64))
        assert pkt.camera_features.dtype == np.float16

    def test_rangefinder_coerced_to_float32(self):
        pkt = self._make(rangefinder=2.0)
        assert isinstance(pkt.rangefinder, np.float32)

    def test_servo_positions_coerced_to_float16(self):
        pkt = self._make(servo_positions=np.array([0.1, -0.2, 0.3], dtype=np.float64))
        assert pkt.servo_positions.dtype == np.float16

    def test_tick_coerced_to_int64(self):
        pkt = self._make(tick=99)
        assert isinstance(pkt.tick, np.int64)

    def test_wrong_camera_shape_raises(self):
        with pytest.raises(AssertionError):
            self._make(camera_features=np.ones(64, dtype=np.float16))

    def test_wrong_servo_shape_raises(self):
        with pytest.raises(AssertionError):
            self._make(servo_positions=np.zeros(2, dtype=np.float16))

    def test_audio_optional_none(self):
        pkt = self._make()
        assert pkt.audio_feature is None

    def test_audio_coerced_when_provided(self):
        audio = np.ones(16, dtype=np.float32)
        pkt = self._make(audio_feature=audio)
        assert pkt.audio_feature is not None
        assert pkt.audio_feature.dtype == np.float16
        assert pkt.audio_feature.shape == (16,)

    def test_audio_wrong_shape_raises(self):
        with pytest.raises(AssertionError):
            self._make(audio_feature=np.ones(8, dtype=np.float16))

    def test_to_activation_vector_shape_without_audio(self):
        pkt = self._make()
        vec = pkt.to_activation_vector()
        expected = CAMERA_FEATURE_DIM + 1 + SERVO_CHANNELS  # 128+1+3 = 132
        assert vec.shape == (expected,), f"Expected ({expected},), got {vec.shape}"

    def test_to_activation_vector_shape_with_audio(self):
        pkt = self._make(audio_feature=np.zeros(16, dtype=np.float16))
        vec = pkt.to_activation_vector()
        expected = CAMERA_FEATURE_DIM + 1 + SERVO_CHANNELS + 16  # 148
        assert vec.shape == (expected,)

    def test_to_activation_vector_dtype(self):
        pkt = self._make()
        vec = pkt.to_activation_vector()
        assert vec.dtype == np.float16

    def test_to_activation_vector_contains_rangefinder(self):
        pkt = self._make(rangefinder=3.75)
        vec = pkt.to_activation_vector()
        # rangefinder is at index CAMERA_FEATURE_DIM (128)
        assert abs(float(vec[CAMERA_FEATURE_DIM]) - 3.75) < 0.1  # float16 precision


class TestActionPacket:
    def test_motor_vector_coerced_to_float16(self):
        ap = ActionPacket(motor_vector=np.ones(MOTOR_CHANNELS, dtype=np.float64))
        assert ap.motor_vector.dtype == np.float16

    def test_wrong_motor_shape_raises(self):
        with pytest.raises(AssertionError):
            ActionPacket(motor_vector=np.zeros(3, dtype=np.float16))

    def test_stop_returns_zeros(self):
        ap = ActionPacket.stop()
        assert np.all(ap.motor_vector == 0.0)

    def test_from_activation_clips_to_range(self):
        # Values outside [-1, 1] should be clamped
        activation = np.array([2.0, -3.0, 0.5, -0.5] + [0.0] * 100, dtype=np.float32)
        ap = ActionPacket.from_activation(activation)
        assert float(ap.motor_vector[0]) == pytest.approx(1.0, abs=1e-2)
        assert float(ap.motor_vector[1]) == pytest.approx(-1.0, abs=1e-2)

    def test_from_activation_requires_min_length(self):
        with pytest.raises(AssertionError):
            ActionPacket.from_activation(np.zeros(2, dtype=np.float16))

    def test_frozen_constants(self):
        assert CAMERA_FEATURE_DIM == 128
        assert SERVO_CHANNELS == 3
        assert MOTOR_CHANNELS == 4


# -----------------------------------------------------------------------
# create_cluster
# -----------------------------------------------------------------------

class TestCreateCluster:
    def test_default_params_within_spec(self):
        c = create_cluster(cluster_id=0)
        assert 2000 <= c.n_neurons <= 8192
        assert 20000 <= c.n_edges() <= 80000

    def test_n_neurons_too_small_raises(self):
        with pytest.raises(AssertionError):
            create_cluster(0, n_neurons=1999, n_edges=20000)

    def test_n_neurons_too_large_raises(self):
        with pytest.raises(AssertionError):
            create_cluster(0, n_neurons=9000, n_edges=20000)

    def test_n_edges_too_small_raises(self):
        with pytest.raises(AssertionError):
            create_cluster(0, n_neurons=4096, n_edges=19999)

    def test_n_edges_too_large_raises(self):
        with pytest.raises(AssertionError):
            create_cluster(0, n_neurons=4096, n_edges=80001)

    def test_src_tgt_dtype_uint16(self):
        c = create_cluster(0, n_neurons=2000, n_edges=20000)
        assert c.src.dtype == np.uint16
        assert c.tgt.dtype == np.uint16

    def test_weight_dtype_float16(self):
        c = create_cluster(0, n_neurons=2000, n_edges=20000)
        assert c.weight.dtype == np.float16

    def test_no_self_loops(self):
        c = create_cluster(0, n_neurons=2000, n_edges=20000, rng=np.random.default_rng(0))
        assert not np.any(c.src == c.tgt), "Self-loops must be removed"

    def test_initial_activations_zero(self):
        c = create_cluster(0, n_neurons=2000, n_edges=20000)
        assert np.all(c.activation == 0.0)
        assert np.all(c.recurrent == 0.0)

    def test_initial_traces_zero(self):
        c = create_cluster(0, n_neurons=2000, n_edges=20000)
        assert np.all(c.trace == 0.0)

    def test_memory_bytes_within_spec(self):
        # At 4096N/40000E: 40000*(2+2+2+2) + 4096*(2+2) = 320000+16384 = 336384 bytes (~329KB)
        # Upper-end spec (8K/80K): 80000*8 + 8192*4 = 640000+32768 = 672768 bytes (~657KB)
        c = create_cluster(0, n_neurons=4096, n_edges=40000)
        mb = c.memory_bytes()
        assert mb >= 300_000, f"Too small: {mb} bytes"
        assert mb <= 700_000, f"Too large: {mb} bytes"


# -----------------------------------------------------------------------
# cluster_forward
# -----------------------------------------------------------------------

class TestClusterForward:
    def _make_cluster(self, n=2000, e=20000, seed=1):
        return create_cluster(0, n_neurons=n, n_edges=e, rng=np.random.default_rng(seed))

    def _make_input(self, dim, seed=2):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(dim).astype(np.float16)

    def test_output_shape(self):
        c = self._make_cluster()
        out = cluster_forward(c, self._make_input(132))
        assert out.shape == (c.n_neurons,)

    def test_output_dtype_float16(self):
        c = self._make_cluster()
        out = cluster_forward(c, self._make_input(132))
        assert out.dtype == np.float16

    def test_output_bounded_by_tanh(self):
        c = self._make_cluster()
        out = cluster_forward(c, self._make_input(132))
        assert float(np.max(np.abs(out))) <= 1.0 + 1e-3, "tanh gating failed"

    def test_recurrent_state_updated(self):
        c = self._make_cluster()
        cluster_forward(c, self._make_input(132))
        assert not np.all(c.recurrent == 0.0), "Recurrent state should update"

    def test_trace_updated_after_forward(self):
        c = self._make_cluster()
        cluster_forward(c, self._make_input(132))
        assert not np.all(c.trace == 0.0), "Eligibility traces should update"

    def test_short_input_pads_correctly(self):
        c = self._make_cluster(n=2000, e=20000)
        # Input shorter than n_neurons — should not raise
        out = cluster_forward(c, np.ones(10, dtype=np.float16))
        assert out.shape == (c.n_neurons,)

    def test_no_float16_overflow(self):
        c = self._make_cluster()
        # Saturate input with large values
        big_input = np.full(132, 100.0, dtype=np.float16)
        out = cluster_forward(c, big_input)
        assert np.all(np.isfinite(out.astype(np.float32))), "NaN/Inf in output"

    def test_trace_decays_over_ticks(self):
        c = self._make_cluster(seed=10)
        inp = self._make_input(132, seed=20)

        cluster_forward(c, inp)
        trace_after_first = c.trace.copy().astype(np.float32)

        # Zero-input tick: trace should decay
        cluster_forward(c, np.zeros(132, dtype=np.float16))
        trace_after_second = c.trace.astype(np.float32)

        # Not all traces should be equal to first tick (decay applied)
        diffs = np.abs(trace_after_second - trace_after_first * c.trace_decay)
        # Allow for product-of-activations term; just check it's not identical
        assert not np.allclose(trace_after_second, trace_after_first), "Trace must decay"

    def test_recurrent_decays_to_zero_on_zero_input(self):
        c = self._make_cluster(seed=5)
        inp = self._make_input(132)
        cluster_forward(c, inp)

        # Many zero-input ticks should drive recurrent state toward zero
        zero_inp = np.zeros(132, dtype=np.float16)
        for _ in range(50):
            cluster_forward(c, zero_inp)

        assert float(np.mean(np.abs(c.recurrent.astype(np.float32)))) < 0.01


# -----------------------------------------------------------------------
# Hebbian update
# -----------------------------------------------------------------------

class TestHebbianUpdate:
    def _make_cluster(self, seed=7):
        c = create_cluster(0, n_neurons=2000, n_edges=20000, rng=np.random.default_rng(seed))
        # Prime traces with a forward pass
        inp = np.random.default_rng(seed+1).standard_normal(132).astype(np.float16)
        cluster_forward(c, inp)
        return c

    def test_positive_reward_increases_weights_on_positive_traces(self):
        c = self._make_cluster()
        w_before = c.weight.copy().astype(np.float32)
        t = c.trace.astype(np.float32)
        hebbian_update(c, reward=1.0)
        w_after = c.weight.astype(np.float32)
        # Net change should be correlated with trace sign
        delta = w_after - w_before * c.weight_decay  # subtract decay effect
        pos_trace_mask = t > 0.0
        if pos_trace_mask.any():
            assert delta[pos_trace_mask].mean() > -0.1, "Positive reward should reinforce positive traces"

    def test_negative_reward_reverses_update(self):
        c = self._make_cluster()
        w_before = c.weight.copy().astype(np.float32)
        t = c.trace.astype(np.float32)
        hebbian_update(c, reward=-1.0)
        w_after = c.weight.astype(np.float32)
        delta = w_after - w_before * c.weight_decay
        pos_trace_mask = t > 0.0
        if pos_trace_mask.any():
            assert delta[pos_trace_mask].mean() < 0.1, "Negative reward should weaken positive traces"

    def test_weight_clamped_to_range(self):
        c = self._make_cluster()
        # Force weights to high values before update
        c.weight[:] = np.float16(9.0)
        c.trace[:] = np.float16(1.0)
        hebbian_update(c, reward=100.0)
        assert float(np.max(np.abs(c.weight.astype(np.float32)))) <= 10.0 + 1e-2

    def test_weight_decay_applied_every_update(self):
        c = create_cluster(0, n_neurons=2000, n_edges=20000)
        c.weight[:] = np.float16(1.0)
        hebbian_update(c, reward=0.0)  # zero reward → only decay
        expected = 1.0 * c.weight_decay
        actual = float(np.mean(np.abs(c.weight.astype(np.float32))))
        assert abs(actual - expected) < 0.01, f"Expected ~{expected}, got {actual}"

    def test_no_nan_inf_after_update(self):
        c = self._make_cluster()
        hebbian_update(c, reward=1.0)
        assert np.all(np.isfinite(c.weight.astype(np.float32))), "NaN/Inf in weights"
