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


def encode_input(
    vision: torch.Tensor,
    vitals: torch.Tensor,
    announcer_embedding: torch.Tensor,
    config: LRSGConfig,
) -> torch.Tensor:
    input_tensor = torch.cat(
        [
            vision.flatten(),
            vitals.flatten(),
            announcer_embedding.flatten(),
        ]
    )

    expected = config.num_input_neurons
    if input_tensor.shape[0] > expected:
        input_tensor = input_tensor[:expected]
    elif input_tensor.shape[0] < expected:
        pad = torch.zeros(expected - input_tensor.shape[0], device=input_tensor.device)
        input_tensor = torch.cat([input_tensor, pad])

    return input_tensor


def decode_whisper(output: torch.Tensor, config: LRSGConfig, tick: int = 0, total_tokens_seen: set = None) -> str:
    whisper_latent = output[12:12 + config.whisper_latent_dim]
    if whisper_latent.numel() == 0:
        return ""

    if tick < 200:
        system_token_indices = list(SYSTEM_TOKEN_INDICES)
        forced_token = WHISPER_VOCAB[system_token_indices[tick % len(system_token_indices)]]
        tokens = [forced_token]
        remaining = 7
    else:
        if tick < 1000:
            temp = 2.5
        else:
            temp = 2.5 - 1.8 * min(1.0, (tick - 1000) / 1000)

        boosted = whisper_latent.clone()
        boost_mult = 5.0 if tick < 800 else (5.0 * max(0, (1200 - tick) / 400))
        if boost_mult > 0:
            for idx in SYSTEM_TOKEN_INDICES:
                if idx < boosted.shape[0]:
                    boosted[idx] += boost_mult

        probs = F.softmax(boosted / temp, dim=0)
        probs = probs + 1e-8

        num_samples = min(8, config.whisper_vocab_size // 64, probs.numel())
        if num_samples <= 0:
            return ""
        token_indices = torch.multinomial(probs, num_samples=num_samples, replacement=False)
        tokens = [WHISPER_VOCAB[idx.item() % len(WHISPER_VOCAB)] for idx in token_indices]
        remaining = 0

    natural_tokens = ["the", "a", "an", "to", "of", "in", "is", "it", "be", "as",
                      "at", "by", "for", "with", "on", "that", "this", "have", "from"]
    extra_tokens = [natural_tokens[(tick + i) % len(natural_tokens)] for i in range(remaining)]
    tokens.extend(extra_tokens)

    return " ".join(tokens)


def decode_output(output: torch.Tensor, config: LRSGConfig = None, tick: int = 0, seen_tokens: set = None) -> dict:
    if output.numel() == 0:
        return {"action": 0, "direction": 0, "intent": 0, "whisper_text": "", "whisper_latent": []}

    output_cpu = output.float().cpu()
    noise = torch.randn_like(output_cpu) * 1e-6
    noisy = output_cpu + noise

    action = noisy[0:7] if noisy.shape[0] >= 7 else torch.zeros(7)
    direction = noisy[7:13] if noisy.shape[0] >= 13 else torch.zeros(6)
    intent = noisy[13:15] if noisy.shape[0] >= 15 else torch.zeros(2)

    whisper_latent = noisy[15:25] if output.shape[0] > 15 else torch.zeros(10)

    whisper_text = ""
    if config is not None and output.shape[0] > 12:
        try:
            whisper_text = decode_whisper(output_cpu, config, tick=tick, total_tokens_seen=seen_tokens)
        except Exception:
            whisper_text = ""

    action_idx = 0
    if action.numel() > 0:
        try:
            action_sum = action.sum().item()
            if action_sum != 0 and not (action_sum != action_sum):
                action_idx = int(torch.argmax(action).item())
        except Exception:
            pass

    if tick % 10 == 0:
        raw_logits = action.tolist()
        print(f"[DIAGNOSTIC] TICK={tick} RAW_ACTIONS={raw_logits}")
        print(f"  SELECTED={action_idx}  MOVE_VAL={raw_logits[3] if len(raw_logits) > 3 else 'N/A':.4f}")

    direction_idx = 0
    if direction.numel() > 0:
        try:
            direction_sum = direction.sum().item()
            if direction_sum != 0 and not (direction_sum != direction_sum):
                direction_idx = int(torch.argmax(direction).item())
        except Exception:
            pass

    intent_idx = 0
    if intent.numel() > 0:
        try:
            intent_sum = intent.sum().item()
            if intent_sum != 0 and not (intent_sum != intent_sum):
                intent_idx = int(torch.argmax(intent).item())
        except Exception:
            pass

    return {
        "action": action_idx,
        "direction": direction_idx,
        "intent": intent_idx,
        "whisper_text": whisper_text,
        "whisper_latent": whisper_latent.tolist() if whisper_latent.numel() > 0 else [],
    }
