import math
import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def main() -> None:
    torch.manual_seed(0)
    batch, heads, seq, dim = 2, 4, 6, 8
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)

    rope = RotaryEmbedding(dim)
    cos, sin = rope(seq, q.device)
    q_rot, k_rot = apply_rotary(q, k, cos, sin)
    print("q_rot shape:", q_rot.shape)
    print("k_rot first token sum:", k_rot[0, 0, 0].sum().item())


if __name__ == "__main__":
    main()
