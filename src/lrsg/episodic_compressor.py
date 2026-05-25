import math
import torch
import torch.nn as nn


class EpisodicCompressor(nn.Module):
    def __init__(self, input_dim: int = 10, latent_dim: int = 32, hidden_dim: int = 50, device=None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        if device is None:
            device = torch.device("cpu")
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, device=device),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim, device=device),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, device=device),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim, device=device),
            nn.Tanh(),
        )

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

    def train_step(self, x: torch.Tensor) -> tuple[float, torch.Tensor]:
        x = x.to(self.device)
        z, x_recon = self(x)
        loss = self.loss_fn(x_recon, x)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return loss.item(), z.detach()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.decoder(z)


class EpisodicMemoryBuffer:
    def __init__(self, max_size: int = 100, input_dim: int = 10, device=None):
        self.max_size = max_size
        self.input_dim = input_dim
        if device is None:
            device = torch.device("cpu")
        self.device = device

        self.buffer = torch.zeros(max_size, input_dim, device=device)
        self.timestamps = [0] * max_size
        self.compression_mask = torch.zeros(max_size, device=device)
        self.head = 0
        self.count = 0

    def push(self, experience_vec: torch.Tensor, tick: int):
        vec = experience_vec.detach().clone().to(self.device)
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)
        if vec.shape[1] != self.input_dim:
            vec = vec[:, :self.input_dim]
            if vec.shape[1] < self.input_dim:
                pad = torch.zeros(vec.shape[0], self.input_dim - vec.shape[1], device=self.device)
                vec = torch.cat([vec, pad], dim=1)

        idx = self.head % self.max_size
        self.buffer[idx] = vec.squeeze(0)
        self.timestamps[idx] = tick
        self.head += 1
        self.count = min(self.count + 1, self.max_size)

    def get_recent(self, n: int = 10) -> torch.Tensor:
        n = min(n, self.count)
        if n == 0:
            return torch.zeros(n, self.input_dim, device=self.device)

        indices = []
        for i in range(n):
            idx = (self.head - 1 - i) % self.max_size
            indices.append(idx)
        indices = indices[::-1]
        return self.buffer[indices].to(self.device)

    def get_all(self) -> torch.Tensor:
        if self.count == 0:
            return torch.zeros(1, self.input_dim, device=self.device)
        first_valid = (self.head - self.count) % self.max_size
        result = []
        for i in range(self.count):
            idx = (first_valid + i) % self.max_size
            result.append(self.buffer[idx])
        return torch.stack(result)

    def age_weights(self, current_tick: int, half_life: int = 50) -> torch.Tensor:
        weights = torch.ones(self.count, device=self.device)
        for i in range(self.count):
            age = current_tick - self.timestamps[i]
            weights[i] = math.exp(-age / half_life)
        weights = weights / weights.sum()
        return weights

    def update_compression_mask(self, compressed: torch.Tensor, threshold: float = 0.5):
        recon = torch.zeros_like(compressed)
        with torch.no_grad():
            pass
        error = torch.abs(compressed - recon).mean()
        self.compression_mask[:self.count] = (error < threshold).float()

    def size(self) -> int:
        return self.count