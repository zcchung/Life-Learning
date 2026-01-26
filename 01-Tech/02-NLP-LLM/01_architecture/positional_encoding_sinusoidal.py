import math
import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


def main() -> None:
    batch, seq, d_model = 2, 6, 16
    x = torch.zeros(batch, seq, d_model)
    pe = SinusoidalPositionalEncoding(d_model)
    out = pe(x)
    print("positional encodings shape:", out.shape)
    print("first token, first 6 dims:", out[0, 0, :6].tolist())


if __name__ == "__main__":
    main()
