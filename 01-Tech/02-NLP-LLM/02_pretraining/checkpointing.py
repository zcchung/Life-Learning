import os
import torch
from torch import nn


def save_checkpoint(path: str, model: nn.Module, opt: torch.optim.Optimizer, step: int) -> None:
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, path)


def load_checkpoint(path: str, model: nn.Module, opt: torch.optim.Optimizer) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    return ckpt["step"]


def main() -> None:
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    path = "tmp_checkpoint.pt"

    save_checkpoint(path, model, opt, step=5)
    step = load_checkpoint(path, model, opt)
    print("restored step:", step)

    if os.path.exists(path):
        os.remove(path)


if __name__ == "__main__":
    main()
