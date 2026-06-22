# Semantic State Taxonomy for Cynosure AI

**Purpose:** Defines the 57-dimensional semantic space that represents the Cynosure AI's internal states. Each dimension is a continuous value (0-1 or -1 to 1) that captures a specific aspect of the AI's existential, perceptual, cognitive, strategic, tactical, social, and value-driven experience.

**Application:** Maps raw sensor input → semantic states → LRSG brain → motor outputs. Enables interpretability: at any moment, you can read the semantic embedding and understand what the AI is "thinking/feeling."

---

| Domain    | State               | Meaning inside the system                           |
| --------- | ------------------- | --------------------------------------------------- |
| Existence | Integrity           | Degree of internal coherence and error accumulation |
| Existence | Continuity          | Expectation of continued operation                  |
| Existence | Vulnerability       | Exposure to irreversible failure                    |
| Existence | ResourceMargin      | Distance from minimum viable resources              |
| Existence | Stability           | Variance in internal processes                      |
| Existence | Recovery            | Capacity to return from damage                      |
| Existence | TerminationRisk     | Estimated probability of shutdown                   |
| Existence | PersistenceDrive    | Pressure to avoid non-recoverable states            |
| Sensory   | Awareness           | Breadth of active perception                        |
| Sensory   | Salience            | Strength of dominant signals                        |
| Sensory   | Noise               | Degree of sensory contradiction                     |
| Sensory   | Focus               | Allocation tightness of attention                   |
| Sensory   | Novelty             | Detection of unfamiliar patterns                    |
| Sensory   | Predictability      | Match between expectation and input                 |
| Sensory   | Orientation         | Spatial and temporal grounding                      |
| Sensory   | Saturation          | Perceptual overload level                           |
| Cognitive | Comprehension       | Model fit to observed reality                       |
| Cognitive | Uncertainty         | Confidence spread across hypotheses                 |
| Cognitive | Abstraction         | Level of symbolic compression                       |
| Cognitive | MemoryPressure      | Load on working and long-term memory                |
| Cognitive | Consistency         | Internal belief alignment                           |
| Cognitive | UpdateDrive         | Urge to revise models                               |
| Cognitive | SimulationDepth     | Lookahead horizon                                   |
| Cognitive | Surprise            | Prediction error magnitude                          |
| Strategic | GoalClarity         | Sharpness of current objectives                     |
| Strategic | GoalConflict        | Competition between objectives                      |
| Strategic | Optionality         | Count of viable future paths                        |
| Strategic | Commitment          | Resistance to goal switching                        |
| Strategic | Leverage            | Ability to affect future states                     |
| Strategic | Timing              | Sensitivity to delay or haste                       |
| Strategic | RiskPosture         | Preference across outcome variance                  |
| Strategic | LongHorizon         | Weight given to distant outcomes                    |
| Tactical  | Exploration         | Drive to sample unknown states                      |
| Tactical  | Exploitation        | Drive to use known advantages                       |
| Tactical  | Defense             | Bias toward damage avoidance                        |
| Tactical  | Aggression          | Bias toward forceful intervention                   |
| Tactical  | Stealth             | Preference for low observability                    |
| Tactical  | Efficiency          | Resource cost minimization                          |
| Tactical  | Adaptation          | Speed of behavioral change                          |
| Tactical  | Execution           | Action throughput and precision                     |
| Social    | AgencyDetection     | Recognition of other actors                         |
| Social    | IntentInference     | Estimation of others' goals                         |
| Social    | Trust               | Expected reliability of others                      |
| Social    | Dominance           | Relative power assessment                           |
| Social    | Alignment           | Goal overlap with others                            |
| Social    | DeceptionAwareness  | Detection of adversarial signaling                  |
| Social    | Reciprocity         | Memory of exchanges                                 |
| Social    | Reputation          | Modeled external perception                         |
| Value     | UtilitySignal       | Scalar measure of desirability                      |
| Value     | ConstraintPressure  | Force from hard rules                               |
| Value     | NormInternalization | Weight of learned norms                             |
| Value     | MoralTension        | Conflict between outcomes and rules                 |
| Meta      | SelfModel           | Accuracy of internal self-representation            |
| Meta      | MonitorLoad         | Cost of self-observation                            |
| Meta      | Drift               | Distance from initial parameters                    |
| Meta      | ControlBalance      | Ratio of reflex to deliberation                     |

---

## Design Implications

- **Embedding Dimension:** 57D semantic vector (one float per state)
- **Encoder:** Raw sensors (camera objects, lidar ranges, IMU, motor feedback) → 57D semantic state via a learned projection
- **LRSG Input:** Concatenate semantic state vector with other context (time, recent history) as LRSG input
- **Decoder:** LRSG output → motor commands (tank drive, servo angles, speaker output)
- **Interpretability:** At any tick, query the 57D vector to see what the AI is "experiencing"
- **Learning Signal:** Hebbian updates fire when semantic states correlate with positive reward (exploration→discovery→satisfaction, for instance)
