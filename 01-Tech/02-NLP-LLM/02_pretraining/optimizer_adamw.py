import torch
from torch import nn


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith("bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def main() -> None:
    model = nn.Sequential(nn.Linear(16, 32), nn.LayerNorm(32), nn.Linear(32, 16))
    groups = build_param_groups(model, weight_decay=0.1)
    opt = torch.optim.AdamW(groups, lr=1e-3)
    print("param groups:", len(opt.param_groups))
    print("decay params:", len(opt.param_groups[0]["params"]))
    print("no_decay params:", len(opt.param_groups[1]["params"]))


if __name__ == "__main__":
    main()
