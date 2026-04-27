import torch

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        super(WarmupScheduler, self).__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            return [base_lr * (self.max_steps - self.last_epoch) / (self.max_steps - self.warmup_steps) for base_lr in self.base_lrs]