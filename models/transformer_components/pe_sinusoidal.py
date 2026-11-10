import torch
import torch.nn as nn
from torch.nn import functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.model_dim = model_dim

        # Create a long enough P matrix
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, model_dim, 2).float() / model_dim)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, model_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_length, model_dim)
        seq_length = x.size(1)
        return self.pe[:, :seq_length, :].to(x.device)  # Return positional encoding for the input sequence length
    