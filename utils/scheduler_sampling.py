import math

import torch
import torch.nn as nn

class LinearSchedulerSampling:
    def __init__(self, teacher_forcing_ratio: float = 1.0, max_steps: int = 10000, use: bool = False):
        self.initial_teacher_forcing_ratio = teacher_forcing_ratio
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_steps = max_steps
        self.step_count = 0
        self.use = use

    def should_sample(self) -> bool:
        if not self.use:
            return True
        return torch.rand(1).item() <= self.teacher_forcing_ratio
    
    def step(self):
        if self.use:
            progress = self.step_count / self.max_steps
            self.teacher_forcing_ratio = max(0.0, self.initial_teacher_forcing_ratio * (1 - progress))
        self.step_count += 1

    def get_ratio(self) -> float:
        return self.teacher_forcing_ratio
    

class SigmoidSchedulerSampling:
    def __init__(self, teacher_forcing_ratio: float = 1.0, max_steps: int = 10000, sigma: int = 10000, use: bool = False):
        self.initial_teacher_forcing_ratio = teacher_forcing_ratio
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.center = max_steps / 2
        self.max_steps = max_steps
        self.sigma = sigma
        self.step_count = 0
        self.use = use

    def should_sample(self) -> bool:
        if not self.use:
            return True
        use = torch.rand(1).item() <= self.teacher_forcing_ratio
        # print(f"Step: {self.step_count}, Teacher Forcing Ratio: {self.teacher_forcing_ratio:.4f}, Use Teacher Forcing: {use}")
        return use
    
    def step(self):
        if self.use:
            self.teacher_forcing_ratio = 1.0 / (1.0 + math.exp((self.step_count - self.center) / self.sigma))
        self.step_count += 1

    def get_ratio(self) -> float:
        return self.teacher_forcing_ratio