import torch


def relative_position_bucket(relative_position: torch.Tensor, num_buckets: int = 32, max_distance: int = 128) -> torch.Tensor:
    num_buckets //= 2
    sign = (relative_position > 0).to(torch.long)
    n = relative_position.abs()

    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_if_large = max_exact + (
        (torch.log(n.float() / max_exact + 1e-6) / torch.log(torch.tensor(max_distance / max_exact)))
        * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    bucket = torch.where(is_small, n.to(torch.long), val_if_large)
    return bucket + sign * num_buckets


def relative_position_bias(seq_len: int, num_buckets: int = 32) -> torch.Tensor:
    pos = torch.arange(seq_len)
    rel = pos.view(1, -1) - pos.view(-1, 1)
    return relative_position_bucket(rel, num_buckets=num_buckets)


def main() -> None:
    seq_len = 6
    buckets = relative_position_bias(seq_len, num_buckets=32)
    print("bucket shape:", buckets.shape)
    print("bucket row0:", buckets[0].tolist())


if __name__ == "__main__":
    main()
