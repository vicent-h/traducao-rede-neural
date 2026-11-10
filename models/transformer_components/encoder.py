import torch
import torch.nn as nn
from torch.nn import functional as F
from models.transformer_components.mha import MultiHeadAttention
from models.transformer_components.attention import CachedMultiHeadAttention
from models.transformer_components.pe_sinusoidal import SinusoidalPositionalEncoding
from models.transformer_components.feed_forward import FeedForward

from logging import getLogger

logger = getLogger(__name__)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        # Disable use_cache for encoder self-attention to avoid cache persisting across
        # different batches (which may have different batch sizes).
        self.self_attn = MultiHeadAttention(model_dim, num_heads, use_cache=False)
        self.ff = FeedForward(model_dim, n_factor=4, dropout=dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # src shape: (seq_length, batch_size, model_dim)
        src2 = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.ff(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoderLayerPreNorm(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(TransformerEncoderLayerPreNorm, self).__init__()
        # Disable use_cache for encoder self-attention to avoid cache persisting across
        # different batches (which may have different batch sizes).
        self.self_attn = CachedMultiHeadAttention(model_dim, num_heads)
        self.ff = FeedForward(model_dim, n_factor=4, dropout=dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def reset_cache(self):
        self.self_attn.reset_use_cache()

    def forward(self, src):
        # src shape: (seq_length, batch_size, model_dim)
        src_norm = self.norm1(src)
        src2 = self.self_attn(src_norm, src_norm, src_norm)
        src = src + self.dropout1(src2)
        src_norm = self.norm2(src)
        src2 = self.ff(src_norm)
        src = src + self.dropout2(src2)
        return src

class TransformerEncoder(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            model_dim, 
            num_heads, 
            num_layers, 
            dropout=0.1
        ):
        super(TransformerEncoder, self).__init__()
        self.model_dim = model_dim
        self.transformer_encoder = nn.ModuleList(
            [TransformerEncoderLayer(model_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        logger.debug(f"Input shape after embedding and positional encoding: {x.shape}")
        # x = x.permute(1, 0, 2)  # (seq_length, batch_size, model_dim) for transformer
        output = x
        for i, layer in enumerate(self.transformer_encoder):
            logger.debug(f"Layer do encoder {i}:")
            output = layer(output)
        logger.debug(f"Output shape after encoder layers: {output.shape}")
        # output = output.permute(1, 0, 2)  # (batch_size, seq_length, model_dim)
        return output

class TransformerEncoderPreNorm(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            model_dim, 
            num_heads, 
            num_layers, 
            dropout=0.1
        ):
        super(TransformerEncoderPreNorm, self).__init__()
        self.model_dim = model_dim
        self.transformer_encoder = nn.ModuleList(
            [TransformerEncoderLayerPreNorm(model_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def reset_cache(self):
        for layer in self.transformer_encoder:
            layer.reset_cache()

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        logger.debug(f"Input shape after embedding and positional encoding: {x.shape}")
        # x = x.permute(1, 0, 2)  # (seq_length, batch_size, model_dim) for transformer
        output = x
        for i, layer in enumerate(self.transformer_encoder):
            logger.debug(f"Layer do encoder {i}:")
            output = layer(output)
        logger.debug(f"Output shape after encoder layers: {output.shape}")
        # output = output.permute(1, 0, 2)  # (batch_size, seq_length, model_dim)
        return output