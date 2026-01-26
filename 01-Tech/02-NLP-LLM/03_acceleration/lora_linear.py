import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.weight.requires_grad = False
        self.r = r
        self.alpha = alpha
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.weight.t()
        lora = (x @ self.A.t()) @ self.B.t() * self.scaling
        return base + lora


def main() -> None:
    torch.manual_seed(0)
    layer = LoRALinear(8, 4, r=2, alpha=4.0)
    opt = torch.optim.AdamW([layer.A, layer.B], lr=1e-2)

    x = torch.randn(3, 8)
    target = torch.randn(3, 4)
    loss = nn.MSELoss()(layer(x), target)
    loss.backward()
    opt.step()
    print("loss:", loss.item())
    print("base weight grad is None:", layer.weight.grad is None)


if __name__ == "__main__":
    main()
