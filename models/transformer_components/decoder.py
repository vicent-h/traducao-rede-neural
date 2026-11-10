import torch
import torch.nn as nn
from torch.nn import functional as F
from models.transformer_components.mha import MultiHeadAttention
from models.transformer_components.attention import CachedMultiHeadAttention
from models.transformer_components.pe_sinusoidal import SinusoidalPositionalEncoding
from models.transformer_components.feed_forward import FeedForward

from logging import getLogger

logger = getLogger(__name__)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1, use_cache=False, cross_attn_cache=False):
        super(TransformerDecoderLayer, self).__init__()
        # self-attention in decoder should not use kv cache when generating
        # (we accumulate the full tgt sequence and do not rely on incremental self-attn caching)
        self.self_attn = CachedMultiHeadAttention(model_dim, num_heads, use_cache=use_cache)
        # Cross-attention should use the full encoder memory every step;
        # do not enable use_cache for cross-attention (it would incorrectly
        # append encoder keys across decoding steps).
        self.cross_attn = CachedMultiHeadAttention(model_dim, num_heads, use_cache=False)
        self.ff = FeedForward(model_dim, n_factor=4, dropout=dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.use_cache = use_cache

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def reset_cache(self):
        self.self_attn.reset_cache()
        self.cross_attn.reset_cache()

    def forward(self, tgt, memory):
        # tgt shape: (batch_size, seq_length_tgt, model_dim)
        # memory shape: (batch_size, seq_length_src, model_dim)
        if self.use_cache:
            mask = None
        else:
            mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.ff(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoderLayerPreNorm(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1, use_cache=False, cross_attn_cache=False):
        super(TransformerDecoderLayerPreNorm, self).__init__()
        # self-attention in decoder should not use kv cache when generating
        # (we accumulate the full tgt sequence and do not rely on incremental self-attn caching)
        self.self_attn = CachedMultiHeadAttention(model_dim, num_heads)
        # Cross-attention should use the full encoder memory every step;
        # do not enable use_cache for cross-attention (it would incorrectly
        # append encoder keys across decoding steps).
        self.cross_attn = CachedMultiHeadAttention(model_dim, num_heads)
        self.ff = FeedForward(model_dim, n_factor=4, dropout=dropout)
        self.norm1 = nn.RMSNorm(model_dim)
        self.norm2 = nn.RMSNorm(model_dim)
        self.norm3 = nn.RMSNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.use_cache = use_cache

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def reset_cache(self):
        self.self_attn.reset_cache()
        self.cross_attn.reset_cache()

    def forward(self, tgt, memory):
        # tgt shape: (batch_size, seq_length_tgt, model_dim)
        # memory shape: (batch_size, seq_length_src, model_dim)
        if self.use_cache:
            mask = None
        else:
            mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        tgt_norm = self.norm1(tgt)
        tgt2 = self.self_attn(tgt_norm, tgt_norm, tgt_norm, attn_mask=mask, use_cache=self.use_cache)
        tgt = tgt + self.dropout1(tgt2)
        tgt_norm = self.norm2(tgt)
        tgt2 = self.cross_attn(tgt_norm, memory, memory, use_cache=self.use_cache)
        tgt = tgt + self.dropout2(tgt2)
        tgt_norm = self.norm3(tgt)
        tgt2 = self.ff(tgt_norm)
        tgt = tgt + self.dropout3(tgt2)
        return tgt


    



class TransformerDecoder(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            model_dim, 
            num_heads, 
            num_layers, 
            dropout=0.1, 
            vocab_size=50_000,
            use_cache=False
        ):
        super(TransformerDecoder, self).__init__()
        self.model_dim = model_dim
        self.transformer_decoder = nn.ModuleList(
            [TransformerDecoderLayer(model_dim, num_heads, dropout, use_cache=use_cache) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(model_dim, vocab_size)

    def forward(self, x, memory):
        # x shape: (batch_size, seq_length_tgt, output_dim)
        # memory shape: (batch_size, seq_length_src, model_dim)
        # x = x.permute(1, 0, 2)  # (seq_length_tgt, batch_size, model_dim) for transformer
        # memory = memory.permute(1, 0, 2)  # (seq_length_src, batch_size, model_dim)
        output = x
        for i, layer in enumerate(self.transformer_decoder):
            logger.debug(f"Layer do decoder {i}:")
            output = layer(output, memory)
        logger.debug(f"Output shape after decoder layers: {output.shape}")
        # output = output.permute(1, 0, 2)  # (batch_size, seq_length_tgt, model_dim)
        output = self.output_layer(output)  # (batch_size, seq_length_tgt, output_dim)
        return output


class TransformerDecoderPreNorm(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            model_dim, 
            num_heads, 
            num_layers, 
            dropout=0.1, 
            vocab_size=50_000,
            use_cache=False,
        ):
        super(TransformerDecoderPreNorm, self).__init__()
        self.model_dim = model_dim
        self.transformer_decoder = nn.ModuleList(
            [TransformerDecoderLayerPreNorm(model_dim, num_heads, dropout, use_cache=use_cache) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(model_dim, vocab_size)

    def reset_cache(self):
        for layer in self.transformer_decoder:
            layer.reset_cache()

    def forward(self, x, memory):
        # x shape: (batch_size, seq_length_tgt, output_dim)
        # memory shape: (batch_size, seq_length_src, model_dim)
        # x = x.permute(1, 0, 2)  # (seq_length_tgt, batch_size, model_dim) for transformer
        # memory = memory.permute(1, 0, 2)  # (seq_length_src, batch_size, model_dim)
        output = x
        for i, layer in enumerate(self.transformer_decoder):
            logger.debug(f"Layer do decoder {i}:")
            output = layer(output, memory)
        logger.debug(f"Output shape after decoder layers: {output.shape}")
        # output = output.permute(1, 0, 2)  # (batch_size, seq_length_tgt, model_dim)
        output = self.output_layer(output)  # (batch_size, seq_length_tgt, output_dim)
        return output