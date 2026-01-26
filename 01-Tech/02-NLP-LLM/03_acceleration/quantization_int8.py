import torch


def quantize_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = x.abs().max().clamp(min=1e-8)
    scale = max_abs / 127.0
    q = torch.clamp((x / scale).round(), -128, 127).to(torch.int8)
    return q, scale


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 4) * 3
    q, scale = quantize_int8(x)
    x_hat = dequantize_int8(q, scale)
    print("max error:", (x - x_hat).abs().max().item())
    print("quantized dtype:", q.dtype)


if __name__ == "__main__":
    main()
