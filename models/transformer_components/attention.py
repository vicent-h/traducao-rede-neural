import torch
import torch.nn as nn
from torch.nn import functional as F

from logging import getLogger

logger = getLogger(__name__)

class KVCache:
    def __init__(self, max_len):
        self.keys = None
        self.values = None
        self.max_len = max_len


    def add(self, keys, values):
        if self.keys is None or self.values is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = torch.cat([self.keys.to(keys.device), keys], dim=1)
            self.values = torch.cat([self.values.to(values.device), values], dim=1)

            if self.keys.size(1) > self.max_len:
                self.keys = self.keys[:, -self.max_len:, :]
                self.values = self.values[:, -self.max_len:, :]

        return self.keys, self.values

    def reset(self):
        self.keys = None
        self.values = None

class QKVAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(QKVAttention, self).__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

    def forward(
            self, 
            Q: torch.Tensor, 
            K: torch.Tensor, 
            V: torch.Tensor, 
            attn_mask=None
            ) -> torch.Tensor:
        batch_size = Q.size(0)

        K_transposed = K.transpose(-2, -1)  # (batch_size, num_heads, head_dim, seq_length_k)
        head_dim_sqrt = self.head_dim ** 0.5
        scores = torch.matmul(Q, K_transposed) / head_dim_sqrt  # (batch_size, num_heads, seq_length_q, seq_length_k)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == float('-inf'), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_length_q, seq_length_k)
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_length_q, head_dim)

        return attn_output


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, use_cache=False, cross_attn_cache=False):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.out = nn.Linear(model_dim, model_dim)
        
        self.QKVAttention = QKVAttention(model_dim, num_heads)
        
    def apply_attention_mask(self, scores, attn_mask):
        if attn_mask is not None:
            logger.debug(f"Attention mask shape: {attn_mask.shape}, Attention weights shape: {scores.shape}")
            scores = scores.masked_fill(attn_mask == float('-inf'), float('-inf'))
        return scores

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length_q, head_dim)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()  # (batch_size, seq_length_q, num_heads, head_dim)
        x = x.view(x.size(0), -1, self.num_heads * self.head_dim)  # (batch_size, seq_length_q, model_dim)
        return x

    def forward(
            self, 
            query: torch.Tensor, 
            key: torch.Tensor, 
            value: torch.Tensor, 
            attn_mask: torch.Tensor=None
            ):
        batch_size = query.size(0)

        q = self.query(query)  # (batch_size, seq_length_q, model_dim)
        k = self.key(key)      # (batch_size, seq_length_k, model_dim)
        v = self.value(value)  # (batch_size, seq_length_v, model_dim)

        q = self.separate_heads(q)  # (batch_size, num_heads, seq_length_q, head_dim)
        k = self.separate_heads(k)  # (batch_size, num_heads, seq_length_k, head_dim)
        v = self.separate_heads(v)  # (batch_size, num_heads, seq_length_v, head_dim)

        attn_output: torch.Tensor = self.QKVAttention(q, k, v, attn_mask)  # (batch_size, seq_length_q, model_dim)
        
        self.merge_heads(attn_output)  # (batch_size, seq_length_q, model_dim)
        
        logger.debug(f"Attention output after concatenation shape: {attn_output.shape}")
        output = self.out(attn_output)  # (batch_size, seq_length_q, model_dim)
        return output


class CachedMultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, max_len=512):
        super(CachedMultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.out = nn.Linear(model_dim, model_dim)
        
        self.QKVAttention = QKVAttention(model_dim, num_heads)
        self.cache = KVCache(max_len=max_len)  # You can adjust max_len as needed
        
    def apply_attention_mask(self, scores, attn_mask):
        if attn_mask is not None:
            logger.debug(f"Attention mask shape: {attn_mask.shape}, Attention weights shape: {scores.shape}")
            scores = scores.masked_fill(attn_mask == float('-inf'), float('-inf'))
        return scores

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length_q, head_dim)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()  # (batch_size, seq_length_q, num_heads, head_dim)
        x = x.view(x.size(0), -1, self.num_heads * self.head_dim)  # (batch_size, seq_length_q, model_dim)
        return x

    def reset_cache(self):
        self.cache.reset()

    def forward(
            self, 
            query: torch.Tensor, 
            key: torch.Tensor, 
            value: torch.Tensor, 
            attn_mask: torch.Tensor=None,
            use_cache: bool=False
            ):
        
        q = self.query(query)  # (batch_size, seq_length_q, model_dim)
        k = self.key(key)      # (batch_size, seq_length_k, model_dim)
        v = self.value(value)  # (batch_size, seq_length_v, model_dim)

        if use_cache:
            k, v = self.cache.add(k, v)  # Update the cache and get the cached keys and values

        q = self.separate_heads(q)  # (batch_size, num_heads, seq_length_q, head_dim)
        k = self.separate_heads(k)  # (batch_size, num_heads, seq_length_k, head_dim)
        v = self.separate_heads(v)  # (batch_size, num_heads, seq_length_v, head_dim)

        attn_output: torch.Tensor = self.QKVAttention(q, k, v, attn_mask)  # (batch_size, seq_length_q, model_dim)
        
        attn_output = self.merge_heads(attn_output)  # (batch_size, seq_length_q, model_dim)
        
        logger.debug(f"Attention output after concatenation shape: {attn_output.shape}")
        output = self.out(attn_output)  # (batch_size, seq_length_q, model_dim)
        return output