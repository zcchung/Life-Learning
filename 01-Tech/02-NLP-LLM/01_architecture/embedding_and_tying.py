import torch
from torch import nn


class TiedEmbeddingLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.proj.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        return self.proj(x)


def main() -> None:
    torch.manual_seed(0)
    model = TiedEmbeddingLM(vocab_size=20, d_model=16)
    ids = torch.randint(0, 20, (2, 5))
    logits = model(ids)
    print("logits shape:", logits.shape)
    print("tied weights:", model.proj.weight.data_ptr() == model.embed.weight.data_ptr())


if __name__ == "__main__":
    main()
