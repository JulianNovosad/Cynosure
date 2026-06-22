import hmac
import hashlib
import os
import numpy as np
from typing import Tuple, Optional

_HMAC_KEY: Optional[bytes] = None

def _load_hmac_key() -> Optional[bytes]:
    key_b64 = os.environ.get("CYNOSURE_HMAC_KEY")
    if key_b64:
        try:
            import base64
            key = base64.b64decode(key_b64)
            if len(key) >= 16:
                return key
        except Exception:
            pass
    key_path = os.environ.get("CYNOSURE_HMAC_KEY_FILE", "")
    if key_path:
        try:
            with open(key_path, "rb") as f:
                key = f.read().strip()
            if len(key) >= 16:
                return key
        except (FileNotFoundError, PermissionError):
            pass
    return None

def get_hmac_key() -> bytes:
    global _HMAC_KEY
    if _HMAC_KEY is None:
        _HMAC_KEY = _load_hmac_key()
    if _HMAC_KEY is None:
        raise RuntimeError(
            "HMAC key not configured. Set CYNOSURE_HMAC_KEY env var "
            "(base64-encoded, min 16 bytes) or CYNOSURE_HMAC_KEY_FILE pointing to a key file."
        )
    return _HMAC_KEY

def verify_hmac_signature(data: bytes, signature: bytes) -> bool:
    key = get_hmac_key()
    expected = hmac.new(key, data, hashlib.sha256).digest()
    return hmac.compare_digest(expected, signature)


class MotorController:
    def __init__(self, pwm_channels: int = 12):
        self.pwm_channels = pwm_channels
        print("MotorController stub initialized.")

    def send_action(self, action_vector: np.ndarray, hmac_signature: Optional[bytes] = None) -> bool:
        if not self._authenticate_command(action_vector, hmac_signature):
            print("MotorController: HMAC verification FAILED — command rejected")
            return False
        print("MotorController stub: Sending action command...")
        if action_vector.size >= self.pwm_channels:
             return True
        else:
             return False

    def _authenticate_command(self, action_vector: np.ndarray, signature: Optional[bytes]) -> bool:
        if signature is None:
            return False
        key = get_hmac_key()
        data = action_vector.tobytes()
        expected = hmac.new(key, data, hashlib.sha256).digest()
        return hmac.compare_digest(expected, signature)

import struct

def map_action_to_motors(action_vector: np.ndarray) -> bytes:
    """
    Refactored for edge abstraction: Converts high-level action indices
    into a packed byte structure for hardware transmission (SPI/UART).
    Format: [Action:B, Direction:B, Intent:B, Magnitude:f] (Total 7 bytes)
    """
    if action_vector.size < 3:
        return struct.pack("!BBBf", 0, 0, 0, 0.0)
    
    # Extract integer components for hardware-level dispatch
    action_idx = int(action_vector[0])
    dir_idx = int(action_vector[1])
    intent_idx = int(action_vector[2])
    magnitude = float(action_vector[3]) if action_vector.size > 3 else 1.0
    
    return struct.pack("!BBBf", action_idx, dir_idx, intent_idx, magnitude)

if __name__ == "__main__":
    controller = MotorController()
    test_actions = np.random.rand(7).astype(np.float32)
    command = map_action_to_motors(test_actions)
    print(f"Generated motor command vector shape: {command.shape}")
    success = controller.send_action(command)
    print(f"Command sent successfully stub: {success}")
