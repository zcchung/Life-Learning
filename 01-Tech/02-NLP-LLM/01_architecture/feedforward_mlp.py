import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class GEGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, hidden * 2)
        self.out = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.fc(x).chunk(2, dim=-1)
        return self.out(x * torch.nn.functional.gelu(gate))


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16)
    ff = FeedForward(16, 32)
    geglu = GEGLU(16, 32)
    print("ff output shape:", ff(x).shape)
    print("geglu output shape:", geglu(x).shape)


if __name__ == "__main__":
    main()
