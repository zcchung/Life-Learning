import torch


def _get_alibi_slopes(num_heads: int) -> torch.Tensor:
    def get_pow_2_slopes(n: int) -> torch.Tensor:
        start = 2.0 ** (-2.0 ** -(math.log2(n) - 3))
        ratio = start
        return torch.tensor([start * (ratio ** i) for i in range(n)])

    import math

    if math.log2(num_heads).is_integer():
        return get_pow_2_slopes(num_heads)
    closest_pow2 = 2 ** math.floor(math.log2(num_heads))
    slopes = get_pow_2_slopes(closest_pow2)
    extra = get_pow_2_slopes(2 * closest_pow2)[0::2][: num_heads - closest_pow2]
    return torch.cat([slopes, extra])


def alibi_bias(num_heads: int, seq_len: int) -> torch.Tensor:
    slopes = _get_alibi_slopes(num_heads).view(num_heads, 1, 1)
    pos = torch.arange(seq_len)
    rel = pos.view(1, -1) - pos.view(-1, 1)
    rel = rel.abs().float().unsqueeze(0)
    return -slopes * rel


def main() -> None:
    num_heads, seq_len = 4, 6
    bias = alibi_bias(num_heads, seq_len)
    print("alibi bias shape:", bias.shape)
    print("head0 bias row0:", bias[0, 0].tolist())


if __name__ == "__main__":
    main()
