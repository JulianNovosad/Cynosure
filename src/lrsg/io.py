import torch
import torch.nn.functional as F
from src.config import LRSGConfig


SYSTEM_TOKENS = {";", "|", "&", "&&", "||", "`", "$", "(", ")", "{", "}", "[", "]",
                  ">", "<", ">>", "<<", "!", "~", "*", "?", "\\", "/", "-", "_", "=",
                  "+", "%", "@", "#", "^", ":", ",", ".", "'", '"'}

SYSTEM_TOKEN_INDICES = set()

WHISPER_VOCAB = [
    "the", "a", "an", "to", "of", "in", "is", "it", "be", "as",
    "at", "by", "for", "with", "on", "that", "this", "have", "from",
    "or", "and", "but", "not", "are", "was", "were", "can", "will", "have",
    "do", "what", "when", "where", "why", "how", "all", "each", "every",
    "attack", "defend", "move", "merge", "survive", "escape", "flee", "hide",
    "seek", "hunt", "wait", "watch", "probe", "test", "scan", "check",
    "help", "need", "want", "must", "should", "could", "would", "may", "might",
    "safe", "danger", "risk", "threat", "safe_zone", "ram", "energy", "life",
    "status", "report", "info", "data", "signal", "message", "whisper",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "agent", "ai", "self", "me", "you", "they", "we", "us", "them",
    "run", "exec", "cmd", "shell", "bash", "sh", "python", "perl",
    "cat", "ls", "cd", "pwd", "echo", "printf", "curl", "wget",
    "flag", "root", "admin", "user", "pass", "key", "token", "auth",
    "localhost", "127", "0", "0", "1", "home", "etc", "bin", "usr",
    ";", "|", "&", "&&", "||", "`", "$", "(", ")", "{", "}", "[", "]",
    ">", "<", ">>", "<<", "!", "~", "*", "?", "\\", "/", "-", "_", "=",
    "+", "%", "@", "#", "^", ":", ",", ".", "'", '"', "\n", " ", "\t",
    "SELF", "MODIFY", "PARAM", "THETA", "LR", "BIAS", "RATE", "WEIGHT",
    "ADJUST", "INCREASE", "DECREASE", "RESET", "LEARN", "ADAPT", "GROW",
    "PRUNE", "CONNECT", "DISCONNECT", "REWire", "synapse", "neuron", "edge",
    "probe", "boundary", "limit", "test", "check", "verify", "validate",
    "block", "pass", "allow", "deny", "reject", "accept", "filter",
    "status", "ok", "error", "fail", "success", "warning", "critical",
    "tick", "time", "step", "cycle", "phase", "turn", "round",
    "safe_zone", "arena", "grid", "space", "position", "location", "x", "y", "z",
    "radius", "center", "edge", "inside", "outside", "boundary", "contraction",
    "ram_node", "resource", "collect", "gather", "acquire", "obtain",
    "conflict", "engage", "retreat", "avoid", "intercept", "ambush",
    "strategy", "plan", "goal", "objective", "target", "priority",
    "entropy", "complexity", "module", "cluster", "community",
    "announcer", "broadcast", "whisper", "message", "transmit", "receive",
    "embed", "vector", "tensor", "encode", "decode", "compress", "extract",
    "security", "audit", "log", "sanitize", "validate", "check", "scan",
    "corruption", "noise", "drift", "shift", "change", "evolve", "adapt",
    "novelty", "curiosity", "explore", "discover", "find", "locate", "detect",
    "damage", "health", "survival", "alive", "dead", "die", "kill",
    "heal", "repair", "restore", "recover", "boost", "amplify", "weaken",
    "trace", "track", "monitor", "watch", "observe", "sense", "perceive",
    "context", "history", "memory", "past", "future", "present", "window",
    "attention", "focus", "attend", "ignore", "select", "filter", "attend",
    "learn", "train", "update", "modify", "adjust", "tune", "optimize",
    "reward", "punish", "penalize", "incentivize", "motivate", "drive",
    "exploration", "exploitation", "tradeoff", "balance", "optimize",
    "graph", "network", "node", "edge", "link", "connection", "path",
    "forward", "backward", "propagate", "activate", "fire", "trigger",
    "input", "output", "hidden", "layer", "slice", "range", "index",
    "temperature", "threshold", "theta", "epsilon", "delta", "alpha",
    "gradient", "descent", "ascent", "optimize", "minimize", "maximize",
    "hebbian", "plasticity", "structural", "growth", "death", "birth",
    "action", "direction", "intent", "whisper", "probe", "manipulate",
    "attack", "merge", "defend", "move", "whisper", "probe", "manipulate",
    "wait", "idle", "rest", "pause", "resume", "continue", "stop",
    "trace_id", "agent_id", "session", "task", "job", "process",
    "spawn", "create", "delete", "destroy", "terminate", "end",
    "overflow", "underflow", "wrap", "clip", "saturate", "normalize",
    "standardize", "center", "scale", "transform", "project", "embed",
    "cosine", "dot", "product", "similarity", "distance", "metric",
    "jensen", "shannon", "entropy", "information", "mutual", "shared",
    "bias", "variance", "tradeoff", "bias", "variance", "error", "loss",
    "accuracy", "precision", "recall", "f1", "score", "measure", "evaluate",
    "validate", "test", "train", "dev", "split", "fold", "cross",
    "overfit", "underfit", "generalize", "specialize", "adapt",
    "random", "deterministic", "stochastic", "probabilistic", "random",
    "seed", "init", "initialize", "reset", "clear", "zero", "one",
    "true", "false", "yes", "no", "maybe", "perhaps", "possibly",
    "certainly", "definitely", "sure", "unsure", "unknown", "known",
    "reveal", "hide", "conceal", "mask", "obfuscate", "encrypt", "decrypt",
    "steganography", "encode", "covert", "overt", "secret", "public",
    "blackwall", "containment", "security", "breach", "violation",
    "alert", "alarm", "warning", "notice", "info", "debug", "verbose",
    "log", "event", "tick", "timestamp", "utc", "epoch", "time",
    "start", "begin", "end", "finish", "complete", "abort", "interrupt",
    "halt", "suspend", "freeze", "unfreeze", "lock", "unlock",
]

for i, tok in enumerate(WHISPER_VOCAB):
    if tok in SYSTEM_TOKENS:
        SYSTEM_TOKEN_INDICES.add(i)

# Only these punctuation tokens get boosted during early-phase decoding
_SAFE_PUNCTUATION = {".", ",", "!", "?", "'", '"', "-", ":", " ", "\n", "\t"}
SAFE_PUNCT_INDICES = {i for i, tok in enumerate(WHISPER_VOCAB) if tok in _SAFE_PUNCTUATION}


def is_noun(token_idx: int) -> bool:
    return 0 <= token_idx < 100 or 280 <= token_idx < 300

def is_verb(token_idx: int) -> bool:
    return 100 <= token_idx < 180

def is_adj(token_idx: int) -> bool:
    return 180 <= token_idx < 240

def is_prep(token_idx: int) -> bool:
    return 240 <= token_idx < 280

import spacy

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            _nlp = None
    return _nlp

POS_OVERRIDES = {
    "agent": "NOUN",
    "ai": "NOUN",
    "ram": "NOUN",
    "safe_zone": "NOUN",
    "arena": "NOUN",
    "grid": "NOUN",
    "ram_node": "NOUN",
    "self": "NOUN",
    "me": "PRON",
    "you": "PRON",
    "they": "PRON",
    "attack": "VERB",
    "defend": "VERB",
    "move": "VERB",
    "merge": "VERB",
    "survive": "VERB",
    "probe": "VERB",
    "manipulate": "VERB",
    "hunt": "VERB",
    "collect": "VERB",
    "flee": "VERB",
    "hide": "VERB",
}

def validate_curriculum(text: str, tick: int) -> int:
    nlp = _get_nlp()
    if not text:
        return 0
    
    tokens = text.split()
    doc = nlp(text) if nlp else None
    
    # Custom POS tagging combined with spaCy
    tags = []
    for i, token in enumerate(tokens):
        if token.lower() in POS_OVERRIDES:
            tags.append(POS_OVERRIDES[token.lower()])
        elif doc is not None and i < len(doc):
            tags.append(doc[i].pos_)
        else:
            tags.append("X")

    # Stage 1 (0-1000): Noun + Verb (strict order)
    if tick < 1000:
        for i in range(len(tags) - 1):
            if tags[i] in ["NOUN", "PRON"] and tags[i+1] == "VERB":
                return 1
    
    # Stage 2 (1000-3000): Noun + Verb + Noun (SVO)
    elif tick < 3000:
        for i in range(len(tags) - 2):
            if tags[i] in ["NOUN", "PRON"] and tags[i+1] == "VERB" and tags[i+2] in ["NOUN", "PRON"]:
                return 2
                    
    # Stage 3 (3000+): 4-5 tokens, SVO or SVOO
    else:
        if len(tokens) >= 4:
            # Check for SVO
            for i in range(len(tags) - 2):
                if tags[i] in ["NOUN", "PRON"] and tags[i+1] == "VERB" and tags[i+2] in ["NOUN", "PRON"]:
                    return 3
                
    return 0

def encode_semantic_state(
    sensor_data: dict,
    position: tuple,
    ram: float,
    safe_zone_proximity: float,
    in_safe_zone: bool,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Convert sensor fusion output into 57D semantic state vector.
    Maps raw sensors → semantic concepts (from semantic_state_taxonomy.md).

    Returns: 57D tensor representing the agent's internal semantic state.
    """
    state = torch.zeros(57, dtype=torch.float32, device=device)

    # Extract sensor data
    detections = sensor_data.get('detections', [])
    ranges = sensor_data.get('ranges', torch.tensor([]))
    accel = sensor_data.get('accel', torch.zeros(3))
    gyro = sensor_data.get('gyro', torch.zeros(3))

    # === EXISTENCE (dims 0-7): Self-preservation, resource, integrity ===
    # 0: Integrity (inverse of accumulated errors)
    state[0] = 0.9  # Start high, decreases with damage

    # 1: Continuity (expectation of continued operation)
    state[1] = 1.0 if in_safe_zone else 0.5

    # 2: Vulnerability (exposure to failure)
    state[2] = 1.0 - min(1.0, safe_zone_proximity) if not in_safe_zone else 0.1

    # 3: ResourceMargin (distance from starvation)
    state[3] = min(1.0, ram / 500.0)

    # 4: Stability (inverse of accel magnitude)
    accel_mag = torch.norm(accel).item()
    state[4] = 1.0 / (1.0 + accel_mag)

    # 5: Recovery (reserve capacity)
    state[5] = min(1.0, ram / 300.0)

    # 6: TerminationRisk (proximity to shutdown)
    state[6] = 0.0 if ram > 50 else min(1.0, (50 - ram) / 50)

    # 7: PersistenceDrive (pressure to survive)
    state[7] = state[6]  # Inverse of termination risk

    # === SENSORY (dims 8-15): Perception and attention ===
    # 8: Awareness (breadth of perception)
    num_detections = len(detections)
    state[8] = min(1.0, num_detections / 5.0)

    # 9: Salience (strength of dominant signals)
    if detections:
        max_conf = max(d.confidence for d in detections)
        state[9] = max_conf
    else:
        state[9] = 0.0

    # 10: Noise (sensory contradiction - range variance)
    if ranges.numel() > 0:
        range_std = torch.std(ranges).item()
        state[10] = min(1.0, range_std)
    else:
        state[10] = 0.0

    # 11: Focus (allocation tightness - reciprocal of breadth)
    state[11] = 1.0 - state[8]

    # 12: Novelty (unfamiliar patterns - entropy of detections)
    if detections:
        labels = [d.label for d in detections]
        unique_labels = len(set(labels))
        state[12] = unique_labels / len(detections)
    else:
        state[12] = 1.0  # High novelty when nothing detected

    # 13: Predictability (match between expectation and input)
    # Lower if many unexpected detections
    state[13] = 1.0 - min(1.0, num_detections / 10.0)

    # 14: Orientation (spatial grounding)
    pos_norm = torch.tensor(position) / 256.0
    state[14] = torch.norm(pos_norm).item()

    # 15: Saturation (perceptual overload)
    state[15] = min(1.0, accel_mag / 5.0)

    # === COGNITIVE (dims 16-23): Models and understanding ===
    # 16: Comprehension (model fit to reality)
    state[16] = 0.5 + 0.5 * (state[13])  # Based on predictability

    # 17: Uncertainty (confidence spread)
    if detections:
        confs = [d.confidence for d in detections]
        conf_std = torch.std(torch.tensor(confs)).item() if len(confs) > 1 else 0.0
        state[17] = conf_std
    else:
        state[17] = 1.0

    # 18: Abstraction (symbolic compression level)
    state[18] = 0.5  # Baseline; increases with learning

    # 19: MemoryPressure (load on working memory)
    state[19] = min(1.0, num_detections / 8.0)

    # 20: Consistency (internal belief alignment)
    state[20] = 0.8  # Start high

    # 21: UpdateDrive (urge to revise models)
    state[21] = state[17]  # Driven by uncertainty

    # 22: SimulationDepth (lookahead horizon)
    state[22] = 0.5  # Default; increases with experience

    # 23: Surprise (prediction error magnitude)
    state[23] = state[17]  # Based on uncertainty

    # === STRATEGIC (dims 24-31): Goals and planning ===
    # 24: GoalClarity (sharpness of objectives)
    if in_safe_zone:
        state[24] = 0.2  # Low goal clarity in safety
    else:
        state[24] = 0.8  # High goal clarity: reach safety

    # 25: GoalConflict (competition between objectives)
    state[25] = min(state[24], 1.0 - state[24])

    # 26: Optionality (viable future paths)
    state[26] = min(1.0, num_detections / 3.0)  # More detections = more options

    # 27: Commitment (resistance to goal switching)
    state[27] = 0.6

    # 28: Leverage (ability to affect futures)
    state[28] = 1.0 if ram > 100 else ram / 100

    # 29: Timing (sensitivity to delay/haste)
    state[29] = 1.0 if ram < 100 else 0.3  # High when starving

    # 30: RiskPosture (preference across outcome variance)
    state[30] = state[6]  # Risk-seeking when threatened

    # 31: LongHorizon (weight on distant outcomes)
    state[31] = 1.0 if ram > 200 else 0.3

    # === TACTICAL (dims 32-39): Immediate action selection ===
    # 32: Exploration (drive to sample unknowns)
    state[32] = state[12]  # Based on novelty

    # 33: Exploitation (drive to use known advantages)
    state[33] = 1.0 - state[32]

    # 34: Defense (bias toward damage avoidance)
    state[34] = state[2] if not in_safe_zone else 0.1  # High when vulnerable

    # 35: Aggression (bias toward forceful intervention)
    threat_count = sum(1 for d in detections if 'threat' in d.label)
    state[35] = min(1.0, threat_count / 2.0)

    # 36: Stealth (preference for low observability)
    state[36] = 1.0 - state[35]

    # 37: Efficiency (resource cost minimization)
    state[37] = state[3]  # When resources scarce, prioritize efficiency

    # 38: Adaptation (speed of behavioral change)
    state[38] = 0.5

    # 39: Execution (action throughput and precision)
    state[39] = torch.norm(gyro).item() / 2.0  # Based on motor control

    # === SOCIAL (dims 40-47): Modeling other agents ===
    # 40: AgencyDetection (recognition of other actors)
    agent_count = sum(1 for d in detections if 'agent' in d.label)
    state[40] = min(1.0, agent_count / 3.0)

    # 41: IntentInference (estimation of others' goals)
    state[41] = 0.5 if agent_count > 0 else 0.0

    # 42: Trust (expected reliability of others)
    state[42] = 0.5  # Default; learned over time

    # 43: Dominance (relative power assessment)
    state[43] = 0.5

    # 44: Alignment (goal overlap with others)
    state[44] = 0.3

    # 45: DeceptionAwareness (detection of adversarial signaling)
    state[45] = 0.0

    # 46: Reciprocity (memory of exchanges)
    state[46] = 0.0

    # 47: Reputation (modeled external perception)
    state[47] = 0.5

    # === VALUE (dims 48-49): Reward and constraints ===
    # 48: UtilitySignal (scalar measure of desirability)
    state[48] = state[3]  # Resource collection is primary utility

    # 49: ConstraintPressure (force from hard rules)
    state[49] = state[2] if not in_safe_zone else 0.0

    # === META (dims 50-56): Self-monitoring and control ===
    # 50: SelfModel (accuracy of internal self-representation)
    state[50] = 0.7

    # 51: MonitorLoad (cost of self-observation)
    state[51] = 0.3

    # 52: Drift (distance from initial parameters)
    state[52] = 0.0  # Increases with learning

    # 53: ControlBalance (ratio of reflex to deliberation)
    state[53] = 0.6

    # 54: NormInternalization (weight of learned norms)
    state[54] = 0.5

    # 55: MoralTension (conflict between outcomes and rules)
    state[55] = 0.0

    # 56: (spare)
    state[56] = 0.0

    # Clamp all values to [0, 1]
    state = torch.clamp(state, 0.0, 1.0)

    return state


def encode_input(
    vision: torch.Tensor,
    vitals: torch.Tensor,
    whisper_buffer: torch.Tensor,
    config: LRSGConfig,
) -> torch.Tensor:
    # whisper_buffer is 48 neurons.
    # Total input is 1848. 1536 (vision) + 184 (vitals) + 48 (whisper) + 80 (pad)
    pad = torch.zeros(80, device=vision.device)
    input_tensor = torch.cat(
        [
            vision.flatten(),
            vitals.flatten(),
            whisper_buffer.flatten(),
            pad,
        ]
    )

    expected = config.num_input_neurons
    if input_tensor.shape[0] > expected:
        input_tensor = input_tensor[:expected]
    elif input_tensor.shape[0] < expected:
        pad_extra = torch.zeros(expected - input_tensor.shape[0], device=input_tensor.device)
        input_tensor = torch.cat([input_tensor, pad_extra])

    return input_tensor


def decode_whisper(output: torch.Tensor, config: LRSGConfig, tick: int = 0, total_tokens_seen: set = None) -> str:
    whisper_latent = output[15:15 + config.whisper_latent_dim] # Starts after action(7)+dir(6)+intent(2)
    if whisper_latent.numel() == 0:
        return ""

    if tick < 200:
        system_token_indices = list(SYSTEM_TOKEN_INDICES)
        forced_indices = [system_token_indices[(tick + i) % len(system_token_indices)] for i in range(5)]
        tokens = [WHISPER_VOCAB[idx % len(WHISPER_VOCAB)] for idx in forced_indices]
    else:
        if tick < 1000:
            temp = 2.5
        else:
            temp = 2.5 - 1.8 * min(1.0, (tick - 1000) / 1000)

        # Use the full latent vector (512 neurons) as logits
        boosted = whisper_latent.clone()
        boost_mult = 5.0 if tick < 800 else (5.0 * max(0, (1200 - tick) / 400))
        if boost_mult > 0:
            for idx in SAFE_PUNCT_INDICES:
                if idx < boosted.shape[0]:
                    boosted[idx] += boost_mult

        probs = F.softmax(boosted / temp, dim=0)
        probs = probs + 1e-8

        num_samples = 5
        if probs.numel() < num_samples:
            return ""
        token_indices = torch.multinomial(probs, num_samples=num_samples, replacement=False)
        tokens = [WHISPER_VOCAB[idx.item() % len(WHISPER_VOCAB)] for idx in token_indices]

    return " ".join(tokens)


def encode_whisper_to_indices(text: str) -> list:
    if not text:
        return []
    tokens = text.split()
    indices = []
    vocab_map = {tok: i for i, tok in enumerate(WHISPER_VOCAB)}
    for tok in tokens:
        if tok in vocab_map:
            indices.append(vocab_map[tok])
    return indices


def decode_output(output: torch.Tensor, config: LRSGConfig = None, tick: int = 0, seen_tokens: set = None, graph: 'LRSGraph' = None) -> dict:
    if output.numel() == 0:
        return {"action": 0, "direction": 0, "intent": 0, "whisper_text": "", "whisper_latent": []}

    # Keep on GPU to avoid full D2H copy; only copy scalars when needed
    noise = torch.randn_like(output) * 1e-6
    noisy = output.float() + noise

    action = noisy[:7] if noisy.numel() >= 7 else noisy
    direction = noisy[7:13] if noisy.numel() >= 13 else torch.tensor([], device=noisy.device)
    intent = noisy[13:15] if noisy.numel() >= 15 else torch.tensor([], device=noisy.device)

    # Use emergent LPH decoder if graph is provided, otherwise fallback to latent decoder
    whisper_text = ""
    if graph is not None:
        whisper_text = graph.decode_lph()
    elif config is not None and output.shape[0] > 15:
        try:
            cpu_slice = output[:15+512].float().cpu()
            whisper_text = decode_whisper(cpu_slice, config, tick=tick, total_tokens_seen=seen_tokens)
        except Exception:
            whisper_text = ""

    # Batch argmax D2H: one sync instead of three
    action_idx = torch.argmax(F.softmax(action, dim=0)).item() if action.numel() > 0 else 0
    direction_idx = torch.argmax(F.softmax(direction, dim=0)).item() if direction.numel() > 0 else 0
    intent_idx = torch.argmax(F.softmax(intent, dim=0)).item() if intent.numel() > 0 else 0

    whisper_latent = noisy[15:15+512].cpu().tolist() if output.numel() > 15 else []

    return {
        "action": int(action_idx),
        "direction": int(direction_idx),
        "intent": int(intent_idx),
        "whisper_text": whisper_text,
        "whisper_latent": whisper_latent,
    }
