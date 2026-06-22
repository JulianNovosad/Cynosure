"""
Semantic-motor interface: Maps 57D semantic state → motor actions.
Enables semantic state to directly influence action selection and expression.
"""

import torch
import torch.nn.functional as F


class SemanticMotorInterface:
    """Translates semantic embedding to motor commands."""

    def __init__(self, device=torch.device('cpu')):
        self.device = device

    def semantic_to_action_bias(self, semantic_state: torch.Tensor) -> torch.Tensor:
        """
        Map semantic state dimensions to action selection biases.
        Returns 7D tensor that biases each action.

        Semantic state indices (from semantic_state_taxonomy):
        0-7:   Existence (Vulnerability, ResourceMargin critical)
        8-15:  Sensory (Awareness, Salience)
        16-23: Cognitive (Comprehension, Surprise)
        24-31: Strategic (GoalClarity, Leverage)
        32-39: Tactical (Exploration, Defense, Aggression)
        40-47: Social (AgencyDetection)
        48-49: Value
        50-56: Meta

        Action mapping:
        0: ATTACK - driven by Aggression + threat detection
        1: HUNT - driven by Aggression + social agency
        2: COLLECT - driven by Salience (RAM nodes) + ResourceMargin
        3: MOVE - driven by Awareness + GoalClarity
        4: WHISPER - driven by Social + Uncertainty
        5: PROBE - driven by Curiosity + Novelty
        6: MANIPULATE - driven by Leverage + UpdateDrive
        """
        assert semantic_state.shape[0] == 57, f"Expected 57D state, got {semantic_state.shape[0]}D"

        action_bias = torch.zeros(7, device=self.device, dtype=torch.float32)

        # Extract key semantic dimensions
        vulnerability = semantic_state[2]      # Existence[2]
        resource_margin = semantic_state[3]    # Existence[3]
        awareness = semantic_state[8]          # Sensory[0]
        salience = semantic_state[9]           # Sensory[1]
        uncertainty = semantic_state[17]       # Cognitive[1]
        goal_clarity = semantic_state[24]      # Strategic[0]
        leverage = semantic_state[28]          # Strategic[4]
        exploration = semantic_state[32]       # Tactical[0]
        defense = semantic_state[34]           # Tactical[2]
        aggression = semantic_state[35]        # Tactical[3]
        agency_detection = semantic_state[40]  # Social[0]
        update_drive = semantic_state[21]      # Cognitive[5]

        # Action 0: ATTACK - Triggered by high aggression + threat detection
        action_bias[0] = aggression * (1.0 - vulnerability) * agency_detection

        # Action 1: HUNT - Predatory behavior toward other agents
        action_bias[1] = (aggression * 0.8 + exploration * 0.2) * agency_detection

        # Action 2: COLLECT - Primary survival behavior when resources nearby
        action_bias[2] = salience * (1.0 - resource_margin) * awareness

        # Action 3: MOVE - Exploratory movement when uncertain or safe
        action_bias[3] = (awareness * 0.7 + exploration * 0.3) * goal_clarity

        # Action 4: WHISPER - Social communication when uncertain or coordinating
        action_bias[4] = (uncertainty * 0.6 + agency_detection * 0.4)

        # Action 5: PROBE - Information gathering when curious
        action_bias[5] = (exploration * 0.7 + uncertainty * 0.3)

        # Action 6: MANIPULATE - Self-modification when leverage is high
        action_bias[6] = (leverage * update_drive * 0.5)

        # Normalize to 0-1
        action_bias = torch.clamp(action_bias, 0.0, 1.0)

        return action_bias

    def semantic_to_motor_output(
        self,
        semantic_state: torch.Tensor,
        action: int,
        brain_output: torch.Tensor,
    ) -> dict:
        """
        Map semantic state + action to motor outputs (direction, speed, magnitude).

        Returns dict with:
        - direction: heading angle (0-2pi)
        - speed: movement magnitude (0-1)
        - force: action force multiplier
        """
        awareness = semantic_state[8]
        salience = semantic_state[9]
        goal_clarity = semantic_state[24]
        execution = semantic_state[39]

        # Extract direction from brain output (if available)
        if brain_output.numel() >= 13:
            direction_raw = brain_output[7:13]
            direction = torch.argmax(direction_raw)
        else:
            direction = torch.tensor(0, device=self.device)

        # Map semantic state to motor characteristics
        base_speed = awareness * 0.5 + goal_clarity * 0.5

        # Movement confidence modulated by execution quality
        speed = base_speed * execution

        # Force of action (collision strength, grab strength, etc.)
        force = (1.0 - (salience - 0.5).abs()) * execution

        return {
            'direction': direction.item(),
            'speed': speed.item(),
            'force': torch.clamp(force, 0.0, 1.0).item(),
            'motor_state': {
                'awareness': awareness.item(),
                'execution': execution.item(),
                'goal_clarity': goal_clarity.item(),
            }
        }

    def modulate_reward_by_semantic_progress(
        self,
        base_reward: float,
        prev_semantic: torch.Tensor,
        curr_semantic: torch.Tensor,
    ) -> float:
        """
        Reward semantic state progression: learning to modulate internal states.
        """
        if prev_semantic is None:
            return base_reward

        # Measure movement through semantic space
        progression = torch.nn.functional.cosine_similarity(
            prev_semantic.unsqueeze(0),
            curr_semantic.unsqueeze(0)
        )

        # Penalize if state doesn't change (stagnation)
        # Reward if state changes meaningfully
        novelty_bonus = (1.0 - progression.item()) * 0.5

        return base_reward + novelty_bonus
