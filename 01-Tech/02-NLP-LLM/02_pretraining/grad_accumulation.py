import torch
from torch import nn


def main() -> None:
    torch.manual_seed(0)
    model = nn.Linear(8, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    accum_steps = 4
    for step in range(8):
        x = torch.randn(2, 8)
        y = torch.randn(2, 4)
        loss = loss_fn(model(x), y) / accum_steps
        loss.backward()
        if (step + 1) % accum_steps == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
            print("optimizer step at", step + 1)


if __name__ == "__main__":
    main()
