import torch
from torch import nn


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, pad_id: int = -100) -> torch.Tensor:
    vocab = logits.size(-1)
    shift_logits = logits[:, :-1].contiguous().view(-1, vocab)
    shift_labels = labels[:, 1:].contiguous().view(-1)
    return nn.CrossEntropyLoss(ignore_index=pad_id)(shift_logits, shift_labels)


def main() -> None:
    torch.manual_seed(0)
    batch, seq, vocab = 2, 5, 10
    logits = torch.randn(batch, seq, vocab)
    labels = torch.randint(0, vocab, (batch, seq))
    loss = causal_lm_loss(logits, labels)
    print("loss:", loss.item())


if __name__ == "__main__":
    main()
