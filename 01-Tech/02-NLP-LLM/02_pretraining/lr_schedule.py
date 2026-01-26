import math


def warmup_cosine_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    total = 20
    warmup = 4
    base_lr = 3e-4
    lrs = [warmup_cosine_lr(s, warmup, total, base_lr) for s in range(total)]
    print("first 6 lrs:", [round(x, 6) for x in lrs[:6]])
    print("last 3 lrs:", [round(x, 6) for x in lrs[-3:]])


if __name__ == "__main__":
    main()
