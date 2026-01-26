import math
import torch


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def flash_attention_blockwise(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int = 4) -> torch.Tensor:
    bsz, heads, seq, dim = q.shape
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(dim)
    for b in range(bsz):
        for h in range(heads):
            for i in range(seq):
                qi = q[b, h, i]
                m_i = torch.tensor(float("-inf"), device=q.device)
                l_i = torch.tensor(0.0, device=q.device)
                acc = torch.zeros(dim, device=q.device)
                for start in range(0, seq, block_size):
                    kb = k[b, h, start : start + block_size]
                    vb = v[b, h, start : start + block_size]
                    scores = (qi @ kb.transpose(0, 1)) * scale
                    m_b = scores.max()
                    p = torch.exp(scores - m_b)
                    l_b = p.sum()
                    acc_b = p @ vb

                    m_new = torch.maximum(m_i, m_b)
                    l_i = torch.exp(m_i - m_new) * l_i + torch.exp(m_b - m_new) * l_b
                    acc = torch.exp(m_i - m_new) * acc + torch.exp(m_b - m_new) * acc_b
                    m_i = m_new
                out[b, h, i] = acc / l_i
    return out


def main() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 2, 6, 8)
    k = torch.randn(1, 2, 6, 8)
    v = torch.randn(1, 2, 6, 8)

    ref = naive_attention(q, k, v)
    approx = flash_attention_blockwise(q, k, v, block_size=3)
    print("max diff:", (ref - approx).abs().max().item())


if __name__ == "__main__":
    main()
