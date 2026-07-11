"""
Frozen packet formats for the cluster brain interface.

These formats are the boundary between AuroreMkVII (robot side) and Cynosure (server side).
Field names, dtypes, and order are FROZEN — do not change once deployed to hardware.

SensorPacket: robot → server (state packet at 30–60 Hz)
ActionPacket: server → robot (motor command per tick)
"""

from dataclasses import dataclass
import numpy as np


# Frozen constants — wire format depends on these
CAMERA_FEATURE_DIM = 128   # encoder output dimension (frozen)
SERVO_CHANNELS = 3         # elevation, azimuth, interlock
MOTOR_CHANNELS = 4         # 4-channel Fusion HAT+ motor driver


@dataclass
class SensorPacket:
    """
    State packet transmitted from robot at 30–60 Hz.

    All fields are frozen: names, dtypes, and order must match AuroreMkVII exactly.
    """
    camera_features: np.ndarray   # float16, shape (CAMERA_FEATURE_DIM,) — frozen encoder output
    rangefinder: np.float32       # float32, scalar distance in meters
    servo_positions: np.ndarray   # float16, shape (SERVO_CHANNELS,), normalized [−1, 1]
    tick: np.int64                # monotonic tick counter (packet sequence number)
    audio_feature: np.ndarray | None = None  # float16, shape (16,), optional ambient audio

    def __post_init__(self) -> None:
        self.camera_features = np.asarray(self.camera_features, dtype=np.float16)
        self.rangefinder = np.float32(self.rangefinder)
        self.servo_positions = np.asarray(self.servo_positions, dtype=np.float16)
        self.tick = np.int64(self.tick)
        if self.audio_feature is not None:
            self.audio_feature = np.asarray(self.audio_feature, dtype=np.float16)

        assert self.camera_features.shape == (CAMERA_FEATURE_DIM,), (
            f"camera_features must be ({CAMERA_FEATURE_DIM},), got {self.camera_features.shape}"
        )
        assert self.servo_positions.shape == (SERVO_CHANNELS,), (
            f"servo_positions must be ({SERVO_CHANNELS},), got {self.servo_positions.shape}"
        )
        if self.audio_feature is not None:
            assert self.audio_feature.shape == (16,), (
                f"audio_feature must be (16,), got {self.audio_feature.shape}"
            )

    def to_activation_vector(self) -> np.ndarray:
        """Flatten all sensor fields into a single float16 activation vector for cluster input."""
        parts = [
            self.camera_features,                               # (128,)
            np.array([self.rangefinder], dtype=np.float16),     # (1,)
            self.servo_positions,                               # (3,)
        ]
        if self.audio_feature is not None:
            parts.append(self.audio_feature)                   # (16,)
        return np.concatenate(parts).astype(np.float16)

    @property
    def activation_dim(self) -> int:
        base = CAMERA_FEATURE_DIM + 1 + SERVO_CHANNELS
        return base + (16 if self.audio_feature is not None else 0)


@dataclass
class ActionPacket:
    """
    Motor command transmitted from server to robot per tick.

    All fields are frozen: names, dtypes, and order must match AuroreMkVII exactly.
    motor_vector: float16[4], range [−1, 1], one value per motor channel.
      index 0: front-left, 1: front-right, 2: rear-left, 3: rear-right
      +1.0 = full forward, −1.0 = full reverse, 0.0 = stop
    """
    motor_vector: np.ndarray   # float16, shape (MOTOR_CHANNELS,), range [−1, 1]

    def __post_init__(self) -> None:
        self.motor_vector = np.asarray(self.motor_vector, dtype=np.float16)
        assert self.motor_vector.shape == (MOTOR_CHANNELS,), (
            f"motor_vector must be ({MOTOR_CHANNELS},), got {self.motor_vector.shape}"
        )

    # Cluster weights are scaled conservatively (see create_cluster) so the
    # recurrent loop stays stable (decays on zero input instead of saturating).
    # That keeps raw per-tick activations small (~0.01-0.1), too small to move
    # a motor meaningfully. This gain recovers usable motor range without
    # touching the internal recurrent dynamics' stability.
    OUTPUT_GAIN = 10.0

    @classmethod
    def stop(cls) -> "ActionPacket":
        return cls(motor_vector=np.zeros(MOTOR_CHANNELS, dtype=np.float16))

    @classmethod
    def from_activation(cls, activation: np.ndarray) -> "ActionPacket":
        """Extract motor command from cluster output activation vector (first MOTOR_CHANNELS dims)."""
        assert len(activation) >= MOTOR_CHANNELS
        raw = activation[:MOTOR_CHANNELS].astype(np.float32) * cls.OUTPUT_GAIN
        # Clamp to [−1, 1]
        return cls(motor_vector=np.clip(raw, -1.0, 1.0).astype(np.float16))
