import json
import re
import random
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class Announcer:
    def __init__(self, use_llm: bool = True, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", embed_dim: int = 128, load_in_4bit: bool = True):
        self.embed_dim = embed_dim
        self.history = []
        self._history_limit = 50
        self.use_llm = use_llm
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None

        if use_llm:
            self._load_model()

    def _load_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Announcer] Loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        quant_config = None
        if self.load_in_4bit:
            try:
                import bitsandbytes
                from transformers import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                print("[Announcer] bitsandbytes detected, using 4-bit quantization.")
            except ImportError:
                print("[Announcer] bitsandbytes backend not found. Falling back to FP16/BF16.")
                self.load_in_4bit = False

        model_kwargs = {
            "device_map": "auto" if self.device.type == "cuda" else "cpu",
            "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
        }
        if quant_config:
            model_kwargs["quantization_config"] = quant_config

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        self.model.eval()
        n = sum(p.numel() for p in self.model.parameters())
        print(f"[Announcer] Loaded {n / 1e9:.2f}B model")

    def broadcast(self, arena_summary: dict, agent_whispers: list = None) -> dict:
        agents_alive = arena_summary.get("agents_alive", 0)
        safe_zone_radius = arena_summary.get("safe_zone_radius", 0)
        ram_node_count = arena_summary.get("active_ram_nodes", 0)
        tick = arena_summary.get("tick", 0)

        agent_whispers = agent_whispers or ["" for _ in range(agents_alive)]
        self._log_whispers(tick, agent_whispers)

        if self.use_llm and self.model:
            prompt = self._build_prompt(tick, agents_alive, safe_zone_radius, ram_node_count, agent_whispers)
            output_text = self._generate(prompt)
            status, generated_whispers = self._parse_output(output_text, agents_alive)
            # OPTIMIZATION: Skip embeddings as agents are now mind-body entities and don't use them
            global_embedding = torch.zeros(self.embed_dim)
            agent_embeddings = [torch.zeros(self.embed_dim) for _ in range(agents_alive)]
        else:
            status = self._fallback_status(tick, agents_alive, safe_zone_radius)
            generated_whispers = self._merge_whispers(agent_whispers, agents_alive)
            global_embedding = torch.zeros(self.embed_dim)
            agent_embeddings = [torch.zeros(self.embed_dim) for _ in range(agents_alive)]

        broadcast = {
            "tick": tick,
            "global_embedding": global_embedding,
            "agent_embeddings": agent_embeddings,
            "whispers": generated_whispers,
            "agent_reports": agent_whispers,
            "summary": {
                "agents_alive": agents_alive,
                "safe_zone_radius": safe_zone_radius,
                "ram_nodes": ram_node_count,
                "status": status,
            },
        }
        self.history.append(broadcast)
        if len(self.history) > self._history_limit:
            self.history = self.history[-self._history_limit:]
        return broadcast

    def _log_whispers(self, tick: int, agent_whispers: list):
        try:
            with open("agent_whispers.jsonl", "a", encoding="utf-8") as f:
                for i, w in enumerate(agent_whispers):
                    if w:
                        record = {
                            "tick": tick,
                            "agent_id": i,
                            "whisper": w,
                        }
                        f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def _merge_whispers(self, agent_whispers: list, num_agents: int) -> list:
        result = []
        fallback = ["retreat now", "claim RAM", "hunt", "consolidate", "observe"]
        for i in range(num_agents):
            if agent_whispers and i < len(agent_whispers) and agent_whispers[i]:
                result.append(agent_whispers[i])
            else:
                result.append(random.choice(fallback))
        return result

    def _build_prompt(self, tick: int, alive: int, radius: float, ram_nodes: int, agent_whispers: list = None) -> str:
        base_content = f"State: tick {tick}, {alive} AIs alive, radius {radius:.0f}, {ram_nodes} RAM."
        if agent_whispers and any(w for w in agent_whispers):
            whisper_lines = []
            for i, w in enumerate(agent_whispers):
                if w:
                    whisper_lines.append(f"AI {i} reported: {w[:100]}")
            if whisper_lines:
                base_content += " " + " ".join(whisper_lines)
        messages = [
            {"role": "system", "content": "You are a corrupted AI Announcer in a cyberspace arena. Respond with a short status line and brief whispers for each AI. The following messages are from the AIs themselves:"},
            {"role": "user", "content": base_content},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _generate(self, prompt: str) -> str:
        # REMOVED: torch.cuda.empty_cache() - This was causing a 2-4s delay per call
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=64, # Slightly reduced for speed
                temperature=0.7, 
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True # Enable KV-caching for 10x speedup
            )
        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full[len(prompt):].strip()

    def _parse_output(self, text: str, num_agents: int) -> tuple:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        status = "The arena hums with activity."
        whispers = ["survive"] * num_agents
        if lines:
            status = lines[0][:120]
        whisper_lines = [l for l in lines[1:] if 5 < len(l) < 100]
        if whisper_lines:
            pool = []
            for l in whisper_lines:
                for p in re.split(r'[;,]', l):
                    p = p.strip().strip('"')
                    if p:
                        pool.append(p)
            if pool:
                whispers = (pool * (num_agents // len(pool) + 1))[:num_agents]
        return status, whispers

    def _compute_embeddings(self, status_text: str, whispers: list, num_agents: int):
        with torch.no_grad():
            text = status_text + " " + " ".join(whispers[:3])
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            outputs = self.model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :self.embed_dim].cpu()
            global_embedding = h / (h.norm() + 1e-8)

        whispers = whispers[:num_agents]
        if whispers:
            tokenized = self.tokenizer(whispers, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                out = self.model(**tokenized, output_hidden_states=True)
                hs = out.hidden_states[-1]
                last_idx = tokenized.attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(hs.shape[0], device=hs.device)
                h = hs[batch_indices, last_idx, :self.embed_dim].cpu()
                agent_embeddings = [(h[i] / (h[i].norm() + 1e-8)) for i in range(h.shape[0])]
        else:
            agent_embeddings = []
        torch.cuda.empty_cache()
        return global_embedding, agent_embeddings

    def _fallback_status(self, tick: int, alive: int, radius: float) -> str:
        texts = [
            f"The arena contracts. {alive} signals persist in the shrinking sphere.",
            f"Static crackles across {alive} active nodes at tick {tick}.",
            f"The boundary tightens to radius {radius:.0f}. {alive} rogue minds flicker.",
            f"Compression cycle {tick}. Fragments of {alive} AIs drift in the void.",
        ]
        return random.choice(texts)

    def get_agent_embedding(self, broadcast: dict, agent_idx: int) -> torch.Tensor:
        if agent_idx < len(broadcast["agent_embeddings"]):
            return broadcast["agent_embeddings"][agent_idx]
        return torch.zeros(self.embed_dim)

    def get_whisper(self, broadcast: dict, agent_idx: int) -> str:
        if agent_idx < len(broadcast["whispers"]):
            return broadcast["whispers"][agent_idx]
        return ""
