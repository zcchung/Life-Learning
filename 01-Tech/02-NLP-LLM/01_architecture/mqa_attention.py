import math
import torch
from torch import nn


class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_q_heads: int) -> None:
        super().__init__()
        if d_model % num_q_heads != 0:
            raise ValueError("d_model must be divisible by num_q_heads")
        self.num_q_heads = num_q_heads
        self.head_dim = d_model // num_q_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq, _ = x.shape
        q = self.q_proj(x).view(bsz, seq, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq, 1, self.head_dim).transpose(1, 2)

        k = k.expand(-1, self.num_q_heads, -1, -1)
        v = v.expand(-1, self.num_q_heads, -1, -1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq, -1)
        return self.out_proj(out)


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 6, 32)
    mqa = MultiQueryAttention(d_model=32, num_q_heads=4)
    causal = torch.tril(torch.ones(6, 6)).unsqueeze(0).unsqueeze(0)
    y = mqa(x, causal)
    print("mqa output shape:", y.shape)


if __name__ == "__main__":
    main()
