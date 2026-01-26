import math
import torch
from torch import nn


class GQAAttention(nn.Module):
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        if d_model % num_q_heads != 0:
            raise ValueError("d_model must be divisible by num_q_heads")
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim * num_kv_heads, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim * num_kv_heads, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq, _ = x.shape
        q = self.q_proj(x).view(bsz, seq, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        repeat = self.num_q_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq, -1)
        return self.out_proj(out)


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 5, 32)
    attn = GQAAttention(d_model=32, num_q_heads=4, num_kv_heads=2)
    causal = torch.tril(torch.ones(5, 5)).unsqueeze(0).unsqueeze(0)
    y = attn(x, causal)
    print("gqa output shape:", y.shape)


if __name__ == "__main__":
    main()
