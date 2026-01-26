import math
import torch
from torch import nn


class CachedSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cache: dict | None = None) -> tuple[torch.Tensor, dict]:
        bsz, seq, dim = x.shape
        qkv = self.qkv(x).view(bsz, seq, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if cache is not None and "k" in cache:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        new_cache = {"k": k, "v": v}

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.tril(torch.ones(scores.size(-2), scores.size(-1), device=scores.device))
        scores = scores.masked_fill(causal == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq, dim)
        return self.proj(out), new_cache


def main() -> None:
    torch.manual_seed(0)
    attn = CachedSelfAttention(d_model=16, num_heads=4)
    cache = None
    outputs = []
    for _ in range(3):
        token = torch.randn(1, 1, 16)
        out, cache = attn(token, cache)
        outputs.append(out)
    print("decoded steps:", len(outputs))
    print("cache seq len:", cache["k"].size(2))


if __name__ == "__main__":
    main()
