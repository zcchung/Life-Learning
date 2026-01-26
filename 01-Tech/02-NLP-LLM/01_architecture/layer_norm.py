import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8)

    ln_ref = nn.LayerNorm(8)
    ln = LayerNorm(8)
    ln.weight.data.copy_(ln_ref.weight)
    ln.bias.data.copy_(ln_ref.bias)

    out_ref = ln_ref(x)
    out = ln(x)
    print("LayerNorm max diff:", (out - out_ref).abs().max().item())

    rms = RMSNorm(8)
    out_rms = rms(x)
    print("RMSNorm output mean:", out_rms.mean().item())


if __name__ == "__main__":
    main()
