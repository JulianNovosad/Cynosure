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
            for idx in SYSTEM_TOKEN_INDICES:
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

    action = noisy[:7]
    direction = noisy[7:13]
    intent = noisy[13:15]

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
    action_probs = F.softmax(action, dim=0)
    direction_probs = F.softmax(direction, dim=0)
    intent_probs = F.softmax(intent, dim=0)

    argmax_vals = torch.stack([
        torch.argmax(action_probs).float(),
        torch.argmax(direction_probs).float(),
        torch.argmax(intent_probs).float(),
    ]).cpu()
    action_idx = int(argmax_vals[0].item())
    direction_idx = int(argmax_vals[1].item())
    intent_idx = int(argmax_vals[2].item())

    whisper_latent = noisy[15:15+512].cpu().tolist() if output.numel() > 15 else []

    return {
        "action": action_idx,
        "direction": direction_idx,
        "intent": intent_idx,
        "whisper_text": whisper_text,
        "whisper_latent": whisper_latent,
    }
