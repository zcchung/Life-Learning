import torch


def pad_batch(seqs: list[list[int]], pad_id: int = 0) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s)
    return out


def causal_mask(seq_len: int) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len))


def main() -> None:
    seqs = [[1, 2, 3], [4, 5], [6]]
    batch = pad_batch(seqs, pad_id=0)
    mask = causal_mask(batch.size(1))
    print("batch:")
    print(batch)
    print("causal mask:")
    print(mask)


if __name__ == "__main__":
    main()
