import torch
import torch.nn.functional as F


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def swiglu(x: torch.Tensor) -> torch.Tensor:
    x, gate = x.chunk(2, dim=-1)
    return x * F.silu(gate)


def geglu(x: torch.Tensor) -> torch.Tensor:
    x, gate = x.chunk(2, dim=-1)
    return x * F.gelu(gate)


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    x2 = torch.randn(2, 8)
    print("gelu mean:", gelu(x).mean().item())
    print("swish mean:", swish(x).mean().item())
    print("swiglu shape:", swiglu(x2).shape)
    print("geglu shape:", geglu(x2).shape)


if __name__ == "__main__":
    main()
