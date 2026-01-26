import torch
from torch import nn


class Expert(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(), nn.Linear(hidden, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Top1MoE(nn.Module):
    def __init__(self, d_model: int, hidden: int, num_experts: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([Expert(d_model, hidden) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, dim = x.shape
        scores = self.gate(x)
        top_idx = scores.argmax(dim=-1)
        out = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            mask = top_idx == expert_id
            if mask.any():
                out[mask] = expert(x[mask])
        return out


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    moe = Top1MoE(d_model=16, hidden=32, num_experts=4)
    y = moe(x)
    print("moe output shape:", y.shape)


if __name__ == "__main__":
    main()
