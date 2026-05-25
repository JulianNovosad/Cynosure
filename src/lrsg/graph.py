import torch
import math
from src.config import LRSGConfig


class LRSGraph:
    def __init__(self, config: LRSGConfig, device: torch.device = torch.device("cpu"), template_struct: dict = None):
        self.config = config
        self.device = device
        self.N = config.num_neurons
        self.max_deg = config.max_out_degree

        num_input = config.num_input_neurons
        num_output = config.num_output_neurons
        self.input_slice = slice(0, num_input)
        self.output_slice = slice(self.N - num_output, self.N)

        # DIMENSIONS: [N, 10000]
        self.out_targets = torch.full(
            (self.N, self.max_deg), -1, dtype=torch.int32, device=device
        )
        self.out_weights = torch.zeros(
            (self.N, self.max_deg), dtype=torch.float16, device=device
        )
        self.synapse_age = torch.zeros(
            (self.N, self.max_deg), dtype=torch.int32, device=device
        )
        self.activations = torch.zeros(self.N, dtype=torch.float32, device=device)

        # Neurobiological State
        self.rolling_avg = torch.zeros(self.N, dtype=torch.float32, device=device)
        self.positions = torch.rand((self.N, 3), device=device) * 256.0
        self.block_ids = (torch.arange(self.N, device=device) // 100).to(torch.int32)
        
        # Sub-network indices (Capped to 32 to prevent OOM on 4.2M neuron runs on 8GB VRAM)
        h_start = num_input
        h_end = self.N - num_output
        h_size = h_end - h_start
        
        sub_size = min(32, int(0.01 * h_size))
        
        self.idx_p = torch.arange(h_start, h_start + sub_size, device=device)
        self.idx_a = torch.arange(h_start + sub_size, h_start + 2 * sub_size, device=device)
        self.idx_b = torch.arange(h_start + 2 * sub_size, h_start + 3 * sub_size, device=device)

        # PWM: Predictive World Model indices
        pwm_start = h_start + 3 * sub_size
        pwm_end = h_start + 4 * sub_size
        self.idx_pwm = torch.arange(pwm_start, pwm_end, device=device)

        self.idx_h_rest = torch.cat([
            torch.arange(pwm_end, h_end, device=device),
            self.idx_a, self.idx_b, self.idx_p
        ])

        # LPH: Language Processing Hub (Block 9)
        self.idx_lph = torch.where(self.block_ids == 9)[0]
        if self.idx_lph.numel() == 0:
            # Fallback for small N environments or tests
            self.idx_lph = torch.arange(min(900, self.N-1), min(1000, self.N), device=self.device)
        
        self.char_to_idx = torch.zeros(256, dtype=torch.long, device=self.device)
        self.bigram_counts = torch.zeros(256, 256, dtype=torch.float32, device=device)
        self._prev_char_code = -1
        self._init_lph()

        # Sliding activation history buffer for temporal-mean LPH decoding
        self.lph_history = torch.zeros(5, self.idx_lph.numel(), dtype=torch.float32, device=self.device)
        self._lph_history_cursor = 0

        # Adaptive node fatigue array for LPH block
        self.lph_fatigue = torch.zeros(self.idx_lph.numel(), dtype=torch.float32, device=self.device)

        # Streaming word assembler: accumulates characters across ticks
        self._char_buffer = []
        self._max_word_len = 12
        self._word_delimiters = {32, 10, 13, 9, 46, 44, 59, 58}  # space, \n, \r, \t, ., ,, ;, :

        # Cached scalar values (no .item() in hot path)
        self.p_start = int(self.idx_p[0].item())
        self.p_end = int(self.idx_p[-1].item()) + 1

        # PCZ/BG Parameters
        self.W_v = torch.nn.Parameter(torch.randn(1, device=device).half() * 0.1)
        self.W_proj = torch.nn.Parameter(torch.randn(self.idx_a.numel(), self.N, device=device).half() * 0.01)
        self.W_fb = torch.nn.Parameter(torch.randn(self.idx_b.numel(), self.N, device=device).half() * 0.01)
        self.W_inh = torch.nn.Parameter(torch.randn(self.idx_b.numel(), self.idx_a.numel(), device=device).half() * 0.1)

        # PWM Parameters
        pwm_dim = self.idx_pwm.numel()
        act_dim = 7 # Standard action space
        self.W_pwm = torch.nn.Parameter(torch.randn(pwm_dim, pwm_dim, device=device).half() * 0.01)
        self.W_act = torch.nn.Parameter(torch.randn(pwm_dim, act_dim, device=device).half() * 0.01)
        self.W_dec = torch.nn.Parameter(torch.randn(num_input, pwm_dim, device=device).half() * 0.01)

        # PWM Character Prediction Head: maps PWM state → 256-char distribution
        self.W_char_pred = torch.nn.Parameter(torch.randn(256, pwm_dim, device=device).half() * 0.005)
        self.char_pred_lr = config.char_pred_lr
        self.char_pred_history = torch.zeros(5, dtype=torch.long, device=device)
        self.char_pred_cursor = 0

        # Language Hebbian reward tracking
        self._lph_activation_trace = torch.zeros(20, self.idx_lph.numel(), dtype=torch.float32, device=device)
        self._trace_cursor = 0
        self._language_reward_accumulator = torch.tensor(0.0, device=device)

        # Prediction States
        self.predicted_sensory = torch.zeros(num_input, dtype=torch.float32, device=device)
        self.prediction_error = torch.tensor(0.0, device=device)
        self.char_prediction_loss = torch.tensor(0.0, device=device)


        self.activity_history = torch.full(
            (self.N,), -1000, dtype=torch.long, device=device
        )

        self._edge_history = []
        
        # PERSISTENCE GUARD: Load pre-built graph if it exists to save time/VRAM cycles
        if template_struct is not None:
            self._apply_struct(template_struct)
        else:
            import os
            base_path = "base_graph.pt"
            if os.path.exists(base_path):
                print(f"[Graph] Loading pre-built synaptic structure from {base_path}...")
                self.load_structure(base_path)
            else:
                self._init_edges()
            
        self._edge_history.append(self.num_edges)

    def _apply_struct(self, struct: dict):
        """Applies structure from a template dict (sharing static tensors)."""
        # Targets and weights MUST be unique copies because they change
        self.out_targets.copy_(struct["out_targets"])
        self.out_weights.copy_(struct["out_weights"])
        
        # Static parts can be shared by reference if they are on the right device
        if struct["positions"].device == self.device:
            self.positions = struct["positions"]
            self.block_ids = struct["block_ids"]
        else:
            self.positions = struct["positions"].to(self.device)
            self.block_ids = struct["block_ids"].to(self.device)

        # Dense weights are also static and can be shared
        if "W_proj" in struct:
            # We must use the same dtype as the template (which we converted to half)
            self.W_proj = torch.nn.Parameter(struct["W_proj"].to(self.device))
            self.W_fb = torch.nn.Parameter(struct["W_fb"].to(self.device))
            self.W_inh = torch.nn.Parameter(struct["W_inh"].to(self.device))
            self.W_pwm = torch.nn.Parameter(struct["W_pwm"].to(self.device))
            self.W_act = torch.nn.Parameter(struct["W_act"].to(self.device))
            self.W_dec = torch.nn.Parameter(struct["W_dec"].to(self.device))
            self.W_v = torch.nn.Parameter(struct["W_v"].to(self.device))

    def save_structure(self, path: str):
        """Saves only the static synaptic structure (targets, weights, positions, blocks)."""
        struct = {
            "out_targets": self.out_targets.cpu(),
            "out_weights": self.out_weights.cpu(),
            "positions": self.positions.cpu(),
            "block_ids": self.block_ids.cpu(),
            "W_proj": self.W_proj.cpu(),
            "W_fb": self.W_fb.cpu(),
            "W_inh": self.W_inh.cpu(),
            "W_pwm": self.W_pwm.cpu(),
            "W_act": self.W_act.cpu(),
            "W_dec": self.W_dec.cpu(),
            "W_v": self.W_v.cpu(),
            "W_char_pred": self.W_char_pred.cpu(),
        }
        torch.save(struct, path)

    def load_structure(self, path: str):
        """Loads static synaptic structure from disk."""
        struct = torch.load(path, map_location=self.device, weights_only=True)
        self.out_targets.copy_(struct["out_targets"])
        self.out_weights.copy_(struct["out_weights"])
        self.positions.copy_(struct["positions"])
        self.block_ids.copy_(struct["block_ids"])
        if "W_proj" in struct: self.W_proj.data.copy_(struct["W_proj"])
        if "W_fb" in struct: self.W_fb.data.copy_(struct["W_fb"])
        if "W_inh" in struct: self.W_inh.data.copy_(struct["W_inh"])
        if "W_pwm" in struct: self.W_pwm.data.copy_(struct["W_pwm"])
        if "W_act" in struct: self.W_act.data.copy_(struct["W_act"])
        if "W_dec" in struct: self.W_dec.data.copy_(struct["W_dec"])
        if "W_v" in struct: self.W_v.data.copy_(struct["W_v"])
        if "W_char_pred" in struct: self.W_char_pred.data.copy_(struct["W_char_pred"])

    def _init_edges(self):
        """
        OPTIMIZED STOCHASTIC SPATIAL INITIALIZATION (O(N) complexity)
        - Uses pre-calculated block boundaries and vectorized sampling.
        """
        alpha, beta = 0.8, 0.1
        gamma, delta = 0.1, 0.5
        k = self.max_deg
        
        num_local_samples = k * 4
        num_distant_samples = k * 2
        total_candidates = num_local_samples + num_distant_samples
        
        print(f"[Graph] Starting optimized stochastic initialization for {self.N:,} neurons...")
        
        # Pre-calculate block boundaries (assuming 100 neurons per block as per block_ids init)
        # block_ids = i // 100
        block_size = 100
        
        chunk_size = 1024
        for i_start in range(0, self.N, chunk_size):
            i_end = min(i_start + chunk_size, self.N)
            curr_chunk = i_end - i_start
            
            if i_start % 50000 == 0:
                print(f"[Graph] Initializing synapses: {i_start/self.N*100:.1f}%", end='\r')
            
            # 1. Vectorized Sample Candidates
            # Local: Sample from current block (i // 100)
            # For simplicity and speed, we know block_ids[i] is i // 100
            # Indices for block B are [B*100, (B+1)*100)
            block_ids = torch.arange(i_start, i_end, device=self.device) // block_size
            local_offsets = torch.randint(0, block_size, (curr_chunk, num_local_samples), device=self.device)
            local_samples = (block_ids.unsqueeze(1) * block_size + local_offsets) % self.N
            
            # Distant: Randomly from whole graph
            distant_samples = torch.randint(0, self.N, (curr_chunk, num_distant_samples), device=self.device)
            
            # Combine
            candidates = torch.cat([local_samples, distant_samples], dim=1) # [curr_chunk, total_candidates]
            
            # 2. Calculate Probabilities (Vectorized)
            # src_pos: [curr_chunk, 1, 3]
            # dst_pos: [curr_chunk, total_candidates, 3]
            src_pos = self.positions[i_start:i_end].unsqueeze(1)
            # Flatten candidates to get positions efficiently
            flat_cand = candidates.view(-1)
            dst_pos = self.positions[flat_cand].view(curr_chunk, total_candidates, 3)
            
            # dists: [curr_chunk, total_candidates]
            dists = torch.norm(dst_pos - src_pos, dim=2)
            
            # c_ij: [curr_chunk, total_candidates]
            # (src_block == dst_block)
            cand_block_ids = (flat_cand // block_size).view(curr_chunk, total_candidates)
            c_ij = (block_ids.unsqueeze(1) == cand_block_ids).float()
            
            p_vals = alpha * torch.exp(-beta * dists) * c_ij + \
                     gamma * torch.exp(-delta * dists) * (1.0 - c_ij)
            
            # 3. Pick top K
            k_eff = min(k, total_candidates)
            top_p, top_idx_in_cand = torch.topk(p_vals, k_eff, dim=1)
            
            # Map back to global indices
            # candidates: [curr_chunk, total_candidates]
            # top_idx_in_cand: [curr_chunk, k_eff]
            batch_idx = torch.arange(curr_chunk, device=self.device).unsqueeze(1)
            final_targets = candidates[batch_idx, top_idx_in_cand]
            
            self.out_targets[i_start:i_end, :k_eff] = final_targets.to(torch.int32)
            
            # 4. Weight Initialization
            sigma = 0.1 / math.sqrt(k_eff)
            weights = torch.randn((curr_chunk, k_eff), device=self.device) * sigma
            self.out_weights[i_start:i_end, :k_eff] = weights.to(dtype=torch.float16)

        print(f"[Graph] Initializing synapses: 100.0% ")

    def _init_inhibitory(self, weights: torch.Tensor) -> torch.Tensor:
        mask = weights > 0
        n_pos = mask.sum().item()
        n_inhib = int(n_pos * 0.05)
        if n_inhib == 0:
            return weights
        flat_idx = mask.nonzero(as_tuple=True)[0]
        inhib_idx = flat_idx[torch.randint(0, flat_idx.numel(), (n_inhib,), device=self.device)]
        inhib_weights = torch.randn(n_inhib, device=self.device, dtype=torch.float16) * 0.1
        inhib_weights = torch.clamp(inhib_weights, min=-2.0, max=-0.01)
        weights.view(-1)[inhib_idx] = inhib_weights
        return weights

    def _build_chains(self, num_input, num_output, output_start, hidden_start, hidden_end, num_hidden):
        chain_max_len = 5
        num_chains = self.N // 8
        device = self.device

        inputs = torch.randint(0, num_input, (num_chains,), device=device)
        outputs = torch.randint(output_start, self.N, (num_chains,), device=device)

        all_src = []
        all_dst = []

        for pos in range(chain_max_len):
            if pos == 0:
                src_chunk = inputs.clone()
                if num_hidden > 0:
                    dst_chunk = torch.randint(hidden_start, hidden_end, (num_chains,), device=device)
                else:
                    dst_chunk = outputs.clone()
            elif pos < chain_max_len - 1:
                if num_hidden > 0:
                    src_chunk = torch.randint(hidden_start, hidden_end, (num_chains,), device=device)
                    dst_chunk = torch.randint(hidden_start, hidden_end, (num_chains,), device=device)
                else:
                    continue
            else:
                if num_hidden > 0:
                    src_chunk = torch.randint(hidden_start, hidden_end, (num_chains,), device=device)
                else:
                    src_chunk = inputs.clone()
                dst_chunk = outputs.clone()

            all_src.append(src_chunk)
            all_dst.append(dst_chunk)

        if not all_src:
            return

        cat_src = torch.cat(all_src)
        cat_dst = torch.cat(all_dst)

        catted = cat_src.long() * (self.N + 1) + cat_dst.long()
        unique_catted, inverse = torch.unique(catted, return_inverse=True)
        uniq_src = unique_catted // (self.N + 1)
        uniq_dst = unique_catted % (self.N + 1)

        valid = (uniq_src >= 0) & (uniq_dst >= 0) & (uniq_dst < self.N)
        uniq_src = uniq_src[valid]
        uniq_dst = uniq_dst[valid]

        order = uniq_src.argsort()
        uniq_src_sorted = uniq_src[order]
        uniq_dst_sorted = uniq_dst[order]

        weights = torch.randn(uniq_src_sorted.numel(), device=device) * 0.5

        unique_src, inverse, counts = torch.unique_consecutive(
            uniq_src_sorted, return_inverse=True, return_counts=True
        )
        arange = torch.arange(uniq_src_sorted.shape[0], device=device)
        group_starts = torch.zeros(unique_src.shape[0], dtype=torch.long, device=device)
        group_starts[1:] = counts[:-1].cumsum(0)
        group_offset = arange - group_starts[inverse]

        capped = group_offset < self.max_deg
        final_src = uniq_src_sorted[capped]
        final_dst = uniq_dst_sorted[capped]
        final_weights = weights[capped].to(dtype=torch.float16)
        final_slot = group_offset[capped]

        self.out_targets[final_src, final_slot] = final_dst
        self.out_weights[final_src, final_slot] = final_weights

    def _init_lph(self):
        """Frequency-weighted coordinate projection: map high-frequency ASCII chars to
        LPH nodes with highest out-degree capacity and spatial dispersion."""
        freq_sorted_ascii = [
            32,  # ' ' (space)
            101, # 'e'
            116, # 't'
            97,  # 'a'
            111, # 'o'
            105, # 'i'
            110, # 'n'
            115, # 's'
            104, # 'h'
            114, # 'r'
            100, # 'd'
            108, # 'l'
            99,  # 'c'
            117, # 'u'
            109, # 'm'
            119, # 'w'
            102, # 'f'
            103, # 'g'
            121, # 'y'
            112, # 'p'
            98,  # 'b'
            118, # 'v'
            107, # 'k'
            120, # 'x'
            106, # 'j'
            113, # 'q'
            122, # 'z'
            69,  # 'E'
            84,  # 'T'
            65,  # 'A'
            79,  # 'O'
            73,  # 'I'
            78,  # 'N'
            83,  # 'S'
            72,  # 'H'
            82,  # 'R'
            68,  # 'D'
            76,  # 'L'
            67,  # 'C'
            85,  # 'U'
            77,  # 'M'
            87,  # 'W'
            70,  # 'F'
            71,  # 'G'
            89,  # 'Y'
            80,  # 'P'
            66,  # 'B'
            86,  # 'V'
            75,  # 'K'
            88,  # 'X'
            74,  # 'J'
            81,  # 'Q'
            90,  # 'Z'
            48,  # '0'
            49,  # '1'
            50,  # '2'
            51,  # '3'
            52,  # '4'
            53,  # '5'
            54,  # '6'
            55,  # '7'
            56,  # '8'
            57,  # '9'
            46,  # '.'
            44,  # ','
            33,  # '!'
            63,  # '?'
            59,  # ';'
            58,  # ':'
            39,  # "'"
            34,  # '"'
            45,  # '-'
            95,  # '_'
            40,  # '('
            41,  # ')'
            91,  # '['
            93,  # ']'
            123, # '{'
            125, # '}'
            60,  # '<'
            62,  # '>'
            61,  # '='
            43,  # '+'
            42,  # '*'
            47,  # '/'
            92,  # '\\'
            124, # '|'
            38,  # '&'
            37,  # '%'
            35,  # '#'
            64,  # '@'
            94,  # '^'
            126, # '~'
            96,  # '`'
            36,  # '$'
            10,  # '\n'
            9,   # '\t'
            13,  # '\r'
        ]
        remaining = [c for c in range(256) if c not in freq_sorted_ascii]
        freq_sorted_ascii.extend(remaining)

        lph_nodes = self.idx_lph
        lph_pos = self.positions[lph_nodes]

        out_degree = (self.out_targets[lph_nodes] >= 0).sum(dim=1).float()

        p_min = lph_pos.min(dim=0)[0]
        p_max = lph_pos.max(dim=0)[0]
        p_span = p_max - p_min + 1e-6

        centroid = lph_pos.mean(dim=0)
        dispersion = torch.norm(lph_pos - centroid.unsqueeze(0), dim=1)

        capacity_score = out_degree * 0.7 + (dispersion / (dispersion.max() + 1e-6)) * 0.3

        _, rank_order = torch.sort(capacity_score, descending=True)

        num_lph = lph_nodes.numel()
        num_chars = len(freq_sorted_ascii)

        for rank_idx in range(min(num_lph, num_chars)):
            node_rank_position = rank_order[rank_idx]
            node_global_idx = lph_nodes[node_rank_position]
            char_code = freq_sorted_ascii[rank_idx]

            freq_norm = 1.0 - (rank_idx / max(num_chars - 1, 1))
            offset_base = torch.tensor([
                (char_code % 8) / 7.0,
                ((char_code // 8) % 8) / 7.0,
                (char_code // 64) / 3.0
            ], device=self.device)

            target_pos = p_min + offset_base * p_span
            node_current_pos = self.positions[node_global_idx]

            blend = freq_norm * 0.6 + 0.2
            blend = min(blend, 0.95)
            self.positions[node_global_idx] = node_current_pos * (1.0 - blend) + target_pos * blend

            self.char_to_idx[char_code] = node_global_idx

        for char_code in freq_sorted_ascii[num_lph:]:
            best_rank_node = lph_nodes[rank_order[0]]
            self.char_to_idx[char_code] = best_rank_node

        # Seed common English bigrams with elevated counts for bootstrap
        common_bigrams = [
            (ord('t'), ord('h')), (ord('h'), ord('e')), (ord('e'), ord('r')),
            (ord('i'), ord('n')), (ord('n'), ord('g')), (ord('a'), ord('n')),
            (ord('o'), ord('n')), (ord('r'), ord('e')), (ord('s'), ord('t')),
            (ord('e'), ord('s')), (ord('r'), ord('a')), (ord('a'), ord('r')),
            (ord('e'), ord('d')), (ord('o'), ord('r')), (ord('a'), ord('t')),
            (ord('t'), ord('o')), (ord('s'), ord('e')), (ord('o'), ord('u')),
            (ord('u'), ord('r')), (ord('n'), ord('d')), (ord('r'), ord('i')),
            (ord('i'), ord('s')), (ord('o'), ord('f')), (ord('f'), ord('o')),
            (ord('r'), ord('s')), (ord('a'), ord('d')), (ord('e'), ord('a')),
            (ord('l'), ord('e')), (ord('e'), ord('l')), (ord('c'), ord('o')),
            (ord('o'), ord('l')), (ord('l'), ord('l')), (ord('m'), ord('e')),
            (ord('e'), ord('m')), (ord('n'), ord('e')), (ord('s'), ord('a')),
            (ord('a'), ord('l')), (ord('l'), ord('y')), (ord('y'), ord(' ')),
            (ord(' '), ord('t')), (ord(' '), ord('a')), (ord(' '), ord('s')),
            (ord(' '), ord('r')), (ord(' '), ord('m')), (ord(' '), ord('c')),
            (ord('r'), ord('a')), (ord('a'), ord('m')), (ord('m'), ord(' ')),
            (ord('s'), ord('e')), (ord('e'), ord('e')), (ord('k'), ord(' ')),
            (ord('h'), ord('i')), (ord('i'), ord('d')), (ord('d'), ord('e')),
            (ord('f'), ord('a')), (ord('a'), ord('r')), (ord('n'), ord('o')),
            (ord('o'), ord('t')), (ord('t'), ord(' ')), (ord('g'), ord('o')),
            (ord('o'), ord(' ')), (ord('b'), ord('y')), (ord('y'), ord(' ')),
        ]
        for c1, c2 in common_bigrams:
            self.bigram_counts[c1, c2] = 1.0

    def encode_string(self, text: str) -> torch.Tensor:
        """Inject character sequence as spatial spikes in LPH"""
        spikes = torch.zeros(self.N, device=self.device)
        for char in text:
            c_val = ord(char) % 256
            spikes[self.char_to_idx[c_val]] = 1.0
        return spikes

    def _decode_raw_char(self) -> str:
        """Internal: single-character decode from temporal mean (no word assembly)."""
        mean_acts = torch.mean(self.lph_history, dim=0)
        if mean_acts.max() < 1e-6:  # Lowered threshold from 0.0
            return ""
        winner_local = torch.argmax(mean_acts)
        winner_global = self.idx_lph[winner_local]
        char_val = torch.where(self.char_to_idx == winner_global)[0]
        if char_val.numel() > 0:
            return chr(char_val[0].item())
        return ""

    def decode_lph(self) -> str:
        """Sliding-window temporal-mean decoder with streaming word assembly."""
        slot = self._lph_history_cursor % 5
        self.lph_history[slot] = self.activations[self.idx_lph].detach().clone()
        self._lph_history_cursor += 1

        mean_acts = torch.mean(self.lph_history, dim=0)

        if mean_acts.max() < 1e-6:  # Lowered threshold from 0.0
            return ""

        winner_local = torch.argmax(mean_acts)
        winner_global = self.idx_lph[winner_local]
        char_val = torch.where(self.char_to_idx == winner_global)[0]
        if char_val.numel() == 0:
            return ""

        char_code = char_val[0].item()
        ch = chr(char_code)

        if char_code in self._word_delimiters:
            if len(self._char_buffer) > 0:
                word = "".join(self._char_buffer)
                self._char_buffer = []
                return word
            return ""

        if char_code >= 32 and char_code < 127:
            self._char_buffer.append(ch)

        if len(self._char_buffer) >= self._max_word_len:
            word = "".join(self._char_buffer)
            self._char_buffer = []
            return word

        return ""

    def get_prob_slice(self, i: int) -> torch.Tensor:
        """
        Computes the probability distribution slice for source neuron i.
        P(i, j) = α·exp(-β·d)·C_ij + γ·exp(-δ·d)·(1-C_ij)
        """
        alpha, beta = 0.8, 0.1
        gamma, delta = 0.1, 0.5
        
        src_pos = self.positions[i].unsqueeze(0) # [1, 3]
        dists = torch.cdist(src_pos, self.positions).squeeze(0) # [N]
        
        src_block = self.block_ids[i]
        c_ij = (src_block == self.block_ids).float()
        
        prob = alpha * torch.exp(-beta * dists) * c_ij + \
               gamma * torch.exp(-delta * dists) * (1.0 - c_ij)
        
        # Normalize to form a valid categorical distribution for sampling
        prob /= prob.sum()
        return prob

    def get_w_pwm_effective_rank(self) -> float:
        """
        Calculates the effective rank of W_pwm using Shannon entropy of singular values.
        R_eff = exp(H(p)), where p_i = sigma_i / sum(sigma)
        """
        with torch.no_grad():
            s = torch.linalg.svdvals(self.W_pwm.float())
            p = s / s.sum()
            # Entropy H = -sum(p * log(p))
            entropy = -torch.sum(p * torch.log(p + 1e-10))
            return torch.exp(entropy).item()

    def _add_synapse(self, src: int, dst: int, weight: float):
        if src < 0 or dst < 0 or dst >= self.N:
            return
        targets = self.out_targets[src]
        empty = (targets == -1).nonzero(as_tuple=True)[0]
        if empty.numel() > 0:
            idx = empty[0].item()
            self.out_targets[src, idx] = dst
            self.out_weights[src, idx] = torch.tensor(weight, dtype=torch.float16, device=self.device)

    def reset_activations(self):
        self.activations.zero_()

    def inject_input(self, input_tensor: torch.Tensor):
        n = min(input_tensor.shape[0], self.config.num_input_neurons)
        self.activations[:n] = input_tensor[:n]

    def read_output(self) -> torch.Tensor:
        return self.activations[self.output_slice].clone()

    @property
    def num_edges(self) -> int:
        if not hasattr(self, "_cached_num_edges") or self._cached_num_edges is None:
            self._cached_num_edges = int((self.out_targets >= 0).sum().item())
        return self._cached_num_edges

    def _invalidate_edge_cache(self):
        self._cached_num_edges = None

    @property
    def avg_out_degree(self) -> float:
        return self.num_edges / self.N

    def save_state(self, path: str):
        """Save learnable weights and language state for cross-segment persistence."""
        state = {
            "W_char_pred": self.W_char_pred.detach().clone(),
            "out_weights": self.out_weights.detach().clone(),
            "out_targets": self.out_targets.detach().clone(),
            "bigram_counts": self.bigram_counts.detach().clone(),
            "lph_fatigue": self.lph_fatigue.detach().clone(),
            "_lph_activation_trace": self._lph_activation_trace.detach().clone(),
            "_prev_char_code": self._prev_char_code,
            "_language_reward_accumulator": self._language_reward_accumulator.item(),
            "_trace_cursor": self._trace_cursor,
            "char_pred_history": self.char_pred_history.detach().clone(),
            "char_pred_cursor": self.char_pred_cursor,
            "_char_buffer": list(self._char_buffer),
            "lph_history": self.lph_history.detach().clone(),
            "_lph_history_cursor": self._lph_history_cursor,
            "W_pwm": self.W_pwm.detach().clone(),
            "W_act": self.W_act.detach().clone(),
            "W_dec": self.W_dec.detach().clone(),
            "W_v": self.W_v.detach().clone(),
            "W_proj": self.W_proj.detach().clone(),
            "W_fb": self.W_fb.detach().clone(),
            "W_inh": self.W_inh.detach().clone(),
        }
        torch.save(state, path)

    def load_state(self, path: str):
        """Load learnable weights and language state from a saved checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.W_char_pred.data.copy_(state["W_char_pred"])
        self.out_weights.data.copy_(state["out_weights"])
        self.out_targets.data.copy_(state["out_targets"])
        self.bigram_counts.copy_(state["bigram_counts"])
        # Reset fatigue and history to give LPH fresh start
        self.lph_fatigue.zero_()
        self._lph_activation_trace.zero_()
        self._prev_char_code = -1
        # Keep accumulated reward
        self._language_reward_accumulator = torch.tensor(state["_language_reward_accumulator"], device=self.device)
        self._trace_cursor = 0
        self.char_pred_history.zero_()
        self.char_pred_cursor = 0
        self._char_buffer = []
        self.lph_history.zero_()
        self._lph_history_cursor = 0
        self.W_pwm.data.copy_(state["W_pwm"])
        self.W_act.data.copy_(state["W_act"])
        self.W_dec.data.copy_(state["W_dec"])
        self.W_v.data.copy_(state["W_v"])
        self.W_proj.data.copy_(state["W_proj"])
        self.W_fb.data.copy_(state["W_fb"])
        self.W_inh.data.copy_(state["W_inh"])
