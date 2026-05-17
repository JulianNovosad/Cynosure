import torch
from src.config import LRSGConfig


class LRSGraph:
    def __init__(self, config: LRSGConfig, device: torch.device = torch.device("cpu")):
        self.config = config
        self.device = device
        self.N = config.num_neurons
        self.max_deg = config.max_out_degree

        num_input = config.num_input_neurons
        num_output = config.num_output_neurons
        self.input_slice = slice(0, num_input)
        self.output_slice = slice(self.N - num_output, self.N)

        self.out_targets = torch.full(
            (self.N, self.max_deg), -1, dtype=torch.long, device=device
        )
        self.out_weights = torch.zeros(
            (self.N, self.max_deg), dtype=torch.float16, device=device
        )
        self.activations = torch.zeros(self.N, dtype=torch.float32, device=device)

        self.activity_history = torch.full(
            (self.N,), -1000, dtype=torch.long, device=device
        )

        self._edge_history = []

        self._init_edges()

        self._edge_history.append(self.num_edges)

    def _init_edges(self):
        num_input = self.config.num_input_neurons
        num_output = self.config.num_output_neurons
        output_start = self.N - num_output
        hidden_start = num_input
        hidden_end = output_start
        num_hidden = max(0, hidden_end - hidden_start)

        self._build_chains(num_input, num_output, output_start, hidden_start, hidden_end, num_hidden)

        for slot in range(self.max_deg):
            empty = self.out_targets[:, slot] == -1
            n = empty.sum().item()
            if n > 0:
                self.out_targets[empty, slot] = torch.randint(0, self.N, (n,), device=self.device)
                self.out_weights[empty, slot] = torch.randn(n, device=self.device, dtype=torch.float16) * 0.5

        self.out_targets[output_start:] = -1
        self.out_weights[output_start:] = torch.zeros(self.out_weights[output_start:].shape, dtype=torch.float16, device=self.device)

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
        return (self.out_targets >= 0).sum().item()

    @property
    def avg_out_degree(self) -> float:
        return self.num_edges / self.N
