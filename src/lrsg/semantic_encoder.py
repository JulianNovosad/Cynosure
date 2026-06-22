import torch
import torch.nn as nn

class SemanticEncoder(nn.Module):
    """
    Maps raw sensor inputs to 57-dimensional semantic state.

    Input: dict with keys:
        - 'vision': list of detected objects [(x, y, z, class_id, confidence), ...]
        - 'lidar': list of (angle, distance) pairs
        - 'imu': dict {'accel': (x,y,z), 'gyro': (x,y,z), 'mag': (x,y,z)}
        - 'motor_feedback': dict {'left_speed': float, 'right_speed': float}
        - 'ram': float (current resource level, 0-100)
        - 'position': (x, y, z) tuple
        - 'time': int (tick number)

    Output: torch.Tensor of shape (57,), dtype float32, values in [-1, 1]

    The 57 dimensions are:
    [0-6]:     Existence (Integrity, Continuity, Vulnerability, ResourceMargin, Stability, Recovery, TerminationRisk)
    [7]:       PersistenceDrive
    [8-14]:    Sensory (Awareness, Salience, Noise, Focus, Novelty, Predictability, Orientation)
    [15]:      Saturation
    [16-23]:   Cognitive (Comprehension, Uncertainty, Abstraction, MemoryPressure, Consistency, UpdateDrive, SimulationDepth, Surprise)
    [24-31]:   Strategic (GoalClarity, GoalConflict, Optionality, Commitment, Leverage, Timing, RiskPosture, LongHorizon)
    [32-39]:   Tactical (Exploration, Exploitation, Defense, Aggression, Stealth, Efficiency, Adaptation, Execution)
    [40-47]:   Social (AgencyDetection, IntentInference, Trust, Dominance, Alignment, DeceptionAwareness, Reciprocity, Reputation)
    [48-52]:   Value (UtilitySignal, ConstraintPressure, NormInternalization, MoralTension, [unused])
    [53-56]:   Meta (SelfModel, MonitorLoad, Drift, ControlBalance)
    """

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        # Learned projections (minimal for MVP)
        self.vision_fc = nn.Linear(512, 16)  # Object detection → sensory + cognitive
        self.lidar_fc = nn.Linear(360, 16)   # 360 lidar rays → spatial awareness
        self.imu_fc = nn.Linear(9, 8)        # 9 IMU values (accel 3 + gyro 3 + mag 3) → tactical
        self.state_fc = nn.Linear(7, 17)     # ram, position(3), time, [motor feedback](2) → existence + value + meta

    def forward(self, sensor_dict):
        """
        Args:
            sensor_dict: dict with keys as above

        Returns:
            torch.Tensor of shape (57,) in range [-1, 1]
        """
        device = self.device if hasattr(self, 'device') else next(self.parameters()).device

        # Process vision: detect objects, compute salience + novelty
        vision_data = sensor_dict.get('vision', [])
        if vision_data:
            # Flatten detected objects: [x, y, z, class, conf, ...] → single vector
            vision_features = torch.zeros(512, device=device)
            for i, obj in enumerate(vision_data[:10]):  # Max 10 objects
                x, y, z, cls_id, conf = obj
                idx = i * 5
                vision_features[idx:idx+5] = torch.tensor([x/100, y/100, z/100, cls_id/10, conf], device=device, dtype=torch.float32)
            vision_embed = torch.tanh(self.vision_fc(vision_features.unsqueeze(0))).squeeze(0)
        else:
            vision_embed = torch.zeros(16, device=device)

        # Process lidar: compute proximity + awareness
        lidar_data = sensor_dict.get('lidar', [])
        if lidar_data:
            lidar_distances = torch.zeros(360, device=device)
            for i, (angle, dist) in enumerate(lidar_data[:360]):
                lidar_distances[i] = min(dist / 50.0, 1.0)  # Normalize to max range 50m
            lidar_embed = torch.tanh(self.lidar_fc(lidar_distances.unsqueeze(0))).squeeze(0)
        else:
            lidar_embed = torch.zeros(16, device=device)

        # Process IMU: acceleration, rotation → tactical state
        imu_data = sensor_dict.get('imu', {})
        accel = torch.tensor(imu_data.get('accel', (0,0,0)), dtype=torch.float32, device=device) / 10.0  # Normalize
        gyro = torch.tensor(imu_data.get('gyro', (0,0,0)), dtype=torch.float32, device=device) / 1.0
        mag = torch.tensor(imu_data.get('mag', (0,0,0)), dtype=torch.float32, device=device) / 2.0
        imu_features = torch.cat([accel, gyro, mag])
        imu_embed = torch.tanh(self.imu_fc(imu_features.unsqueeze(0))).squeeze(0)

        # Process state: RAM, position, motor feedback → existence + value
        ram = float(sensor_dict.get('ram', 50.0)) / 100.0  # Normalize [0, 1]
        pos = sensor_dict.get('position', (0, 0, 0))
        motor_feedback = sensor_dict.get('motor_feedback', {'left_speed': 0, 'right_speed': 0})
        time_tick = float(sensor_dict.get('time', 0)) / 1000.0  # Normalize for long runs

        state_features = torch.tensor([
            ram,
            pos[0] / 100.0,
            pos[1] / 100.0,
            pos[2] / 100.0,
            time_tick,
            motor_feedback.get('left_speed', 0) / 2.0,
            motor_feedback.get('right_speed', 0) / 2.0,
        ], dtype=torch.float32, device=device)
        state_embed = torch.tanh(self.state_fc(state_features.unsqueeze(0))).squeeze(0)

        # Concatenate all embeddings: 16 + 16 + 8 + 17 = 57
        semantic_state = torch.cat([vision_embed, lidar_embed, imu_embed, state_embed])

        return semantic_state  # shape (57,), range [-1, 1]
