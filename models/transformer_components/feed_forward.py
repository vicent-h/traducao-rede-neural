import torch
import torch.nn as nn
from torch.nn import functional as F

from logging import getLogger

logger = getLogger(__name__)


class FeedForward(nn.Module):
    def __init__(self, model_dim, n_factor=4, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(model_dim, model_dim * n_factor)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(model_dim * n_factor, model_dim)
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x