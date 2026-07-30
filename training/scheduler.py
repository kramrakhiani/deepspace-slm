import math
from typing import Optional


class CosineWarmupScheduler:
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: Optional[float] = None,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr if min_lr is not None else max_lr * 0.1
        self.current_step = 0

    def get_lr(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self.current_step

        if step < self.warmup_steps:
            return self.max_lr * (step / max(1, self.warmup_steps))
        elif step >= self.max_steps:
            return self.min_lr
        else:
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_step += 1
        return lr

    def state_dict(self) -> dict:
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state: dict):
        self.current_step = state["current_step"]
        self.warmup_steps = state["warmup_steps"]
        self.max_steps = state["max_steps"]
        self.max_lr = state["max_lr"]
        self.min_lr = state["min_lr"]
