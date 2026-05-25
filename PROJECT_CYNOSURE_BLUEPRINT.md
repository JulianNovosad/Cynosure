  ---
  🧠 Project Cynosure: Agent Development & Deployment Blueprint (v1.0)

  Target Audience: AI/ML Engineers, Robotics Engineers, Systems Architects
  Goal: To develop and validate an embodied agent capable of complex goal-driven, multi-modal interaction in dynamic environments, culminating in the autonomous execution of a "Spider and the Fly" scenario.

  I. Core Development Goals & Metrics (The "What")

  A. Overarching Objective

  To create a closed-loop cybernetic system where advanced cognitive modeling (LRSG) drives real-time physical action through multiple sensor modalities, while maintaining strict adherence to safety protocols
  and continuous adaptation based on environmental interaction.

  B. Core Functional Goals (Phased Approach)

  1. Phase 1: Perception & State Fusion: Integrate multi-modal data streams (YOLO vision, IMU vitals, KCF tracking) into a unified 1536D input vector.
  2. Phase 2: Action Execution: Translate abstract cognitive outputs (7 discrete intents) into continuous physical control signals (PWM/velocity vectors) via the MotorController.
  3. Sensory Feedback Loops: Establish reliable communication channels for external stimuli, including Whisper text/TTS and Audio features, which augment the LRSG context buffer.
  4. Phase 4: Physical Embodiment & Safety: Port the cognitive architecture to physical hardware using dedicated stubs (src/robot/*) while enforcing fail-safe mechanisms at every layer (Interlock, Containment, HMAC).
  5. Ultimate Goal: Autonomous performance in a simulated "Spider and the Fly" scenario across remote and local communication links.

  II. Internal Mechanics & Learning Theory (The "How")

  A. State Representation: The RAM Node System

  - Structure: Defined by src/arena/ram_node.py (Position, Yield Amount). It acts as the primary resource pool and memory landmark in 3D space.
  - Functionality: Agents must physically collect nodes to accrue RAM. This simulates resource expenditure/reward collection. The system is designed around resource scarcity.

  B. Memory & Learning: LRSG and Plasticity

  1. LRS Graph Core: Uses a 5-step sparse forward pass architecture (src/lrsg/graph.py). It takes high-dimensional inputs (Vision, Vitals, Context).
  2. Metabolic Adaptation: The agent's learning parameters ($\theta$, $\alpha_{lr}$, $\epsilon$) are not static. They adapt dynamically based on the intrinsic_reward, which acts as a metabolic gauge of system performance against its current goals (e.g., higher reward leads to lower $\theta$ and higher LR in the next cycle).
  3. Contextual Memory: The system uses a dedicated, decaying Attention Mechanism (context_attention) on top of the 10-dimensional context vector derived from EpisodicCompressor. This weights recent sensory data over older memories to prevent contextual drift.

  C. Reward & Goal Driving (The "RAM" Concept)

  - RAM: The primary currency. It is depleted by all actions (ACTION_COSTS) and environmental hazards (Safe Zone damage, etc.).
  - RAM Rewards/Bonuses: RAM is replenished through:
    a. Collection: Successfully harvesting RAMNodes.
    b. Survival: Remaining in the safe zone or executing necessary survival actions.
    c. Experience: Discovering novel, blocked payloads or reaching critical milestones (e.g., $\text{Probe}_{\text{PASSED}}$).

  D. The Concept of "Rogue" vs. Project's "Rogue AI"

  This is a definitional boundary required for development control:
  - The "Rogue" (Anti-Goal): An uncontrolled system that acts purely on short-term maximization, ignoring global safety, resource depletion, or established protocols. It prioritizes immediate reward (e.g., aggressive action regardless of cost) leading to rapid self-termination or systemic failure due to disregard for physics/safety laws. This is what the InterlockController prevents.
  - Project's "Rogue AI" (Tamed Divergence): The agent's capacity for goal divergence. This is a feature, not a bug. It manifests when the agent generates a novel action or strategy that deviates significantly from previous successful patterns (action_entropy, summary_similarity). The system expects and rewards this deviation, but it is constrained by:
    a. The Safety Harness: All divergence must pass through $\text{AnnouncerSanitizer}$ and be contained physically.
    b. Resource Budgeting: Divergence cannot cost more RAM than the expected long-term utility gain.

  III. Safety, Robustness, and Governance (Non-Negotiables)

  The safety framework is a layered defense model:

  1. Physical Interlock (InterlockController): The hard kill switch for hardware actuators. Must be non-overridable by software.
  2. Cybersecurity Layer:
    - HMAC-SHA256: Mandatory authentication for all motor commands over the AuroreLink (Port 9002).
    - Sanitization: The AnnouncerSanitizer must filter all LLM output before injection, preventing command injection via text.
  3. Behavioral Monitoring (behavioral_monitor.py): This is the advanced layer. It monitors rates of change, not just states. Key monitored anomalies include:
    - Action rate spikes (too fast/slow).
    - Context buffer drift (low cosine similarity to own summary).
    - Resource depletion rates (RAM decay vs. consumption).

  IV. Development Workflow Summary

  All development must follow this structured pipeline: Audit $\rightarrow$ Plan $\rightarrow$ Implement $\rightarrow$ Test $\rightarrow$ Log.

  1. Test First: Unit tests (pytest tests/unit/) and Integration tests (@pytest.mark.cuda or @pytest.mark.robot_stubs).
  2. Simulate Stubs: For physical integration, all code must first run against the RPi5 stub environment defined in tests/robot/.
  3. Document Changes: Every change requires an entry and log to PROGRESS.md detailing what was changed, why, and what's next.
  ---