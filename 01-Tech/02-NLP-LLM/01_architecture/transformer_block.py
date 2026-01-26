import math
import torch
from torch import nn


class PreNormTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, d_model)
        )
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = nn.Dropout(dropout)

    def _attn(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq, dim = x.shape
        qkv = self.qkv(x).view(bsz, seq, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq, dim)
        return self.proj(out)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(self._attn(self.ln1(x), attn_mask))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 6, 32)
    causal = torch.tril(torch.ones(6, 6)).unsqueeze(0).unsqueeze(0)
    block = PreNormTransformerBlock(d_model=32, num_heads=4, mlp_hidden=64)
    y = block(x, causal)
    print("block output shape:", y.shape)


if __name__ == "__main__":
    main()
