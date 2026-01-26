import math
import torch


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def main() -> None:
    torch.manual_seed(0)
    batch, heads, seq, dim = 2, 2, 5, 4
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)

    causal = torch.tril(torch.ones(seq, seq)).unsqueeze(0).unsqueeze(0)
    out = scaled_dot_product_attention(q, k, v, causal)
    print("attention output shape:", out.shape)
    print("last token head0 sum:", out[0, 0, -1].sum().item())


if __name__ == "__main__":
    main()
