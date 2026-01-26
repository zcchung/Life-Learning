import torch
from torch import nn


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(16, 8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    x = torch.randn(4, 16, device=device)
    y = torch.randn(4, 8, device=device)

    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        loss = nn.MSELoss()(model(x), y)

    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    print("loss:", loss.item())


if __name__ == "__main__":
    main()
