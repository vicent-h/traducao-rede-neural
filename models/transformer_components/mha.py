import torch
import torch.nn as nn
from torch.nn import functional as F

from logging import getLogger

logger = getLogger(__name__)

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, kv_cache=False, cross_attn_cache=False):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.out = nn.Linear(model_dim, model_dim)

        self.map_attention = None  # Para armazenar os pesos de atenção para depuração
        
        self.kv_cache = kv_cache
        self.cross_attn_cache = cross_attn_cache

        # Initialize caches as None and create on first use to avoid
        # shape-mismatch when concatenating along the sequence dimension.
        if self.kv_cache or self.cross_attn_cache:
            self.k_cache = None
            self.v_cache = None
        

    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None

    def apply_attention_mask(self, scores, attn_mask):
        if attn_mask is not None:
            logger.debug(f"Attention mask shape: {attn_mask.shape}, Attention weights shape: {scores.shape}")
            scores = scores.masked_fill(attn_mask == float('-inf'), float('-inf'))
        return scores

    def do_cache(self, key, value):
        if self.kv_cache:
            # If key/value contain a single time-step, treat as incremental generation
            # and append the new k/v to the cache. If a full sequence is provided
            # (initial call), compute full K/V and initialize the cache.
            if key.size(1) == 1:
                cur_k = self.key(key)      # (batch_size, 1, model_dim)
                cur_v = self.value(value)  # (batch_size, 1, model_dim)

                if self.k_cache is None or self.v_cache is None:
                    self.k_cache = cur_k.to(key.device)
                    self.v_cache = cur_v.to(value.device)
                else:
                    self.k_cache = torch.cat([self.k_cache.to(key.device), cur_k], dim=1)
                    self.v_cache = torch.cat([self.v_cache.to(value.device), cur_v], dim=1)

                K = self.k_cache  # (batch_size, seq_length_k, model_dim)
                V = self.v_cache
            else:
                # Full sequence provided: compute K/V for the whole input and
                # initialize the caches so subsequent incremental calls work.
                K = self.key(key)      # (batch_size, seq_length, model_dim)
                V = self.value(value)  # (batch_size, seq_length, model_dim)
                self.k_cache = K.to(key.device)
                self.v_cache = V.to(value.device)

        elif self.cross_attn_cache:
            if self.k_cache is None or self.v_cache is None:
                self.k_cache = self.key(key)      # (batch_size, seq_length, model_dim)
                self.v_cache = self.value(value)  # (batch_size, seq_length, model_dim)
            K = self.k_cache
            V = self.v_cache
        else:
            K = self.key(key)      # (batch_size, seq_length, model_dim)
            V = self.value(value)  # (batch_size, seq_length, model_dim)
        return K, V

            

    def forward(
            self, 
            query: torch.Tensor, 
            key: torch.Tensor, 
            value: torch.Tensor, 
            attn_mask: torch.Tensor=None
            ):
        batch_size = query.size(0)
        K, V = self.do_cache(key, value)  # (batch_size, seq_length_k, model_dim)
        Q: torch.Tensor = self.query(query)  # (batch_size, seq_length, model_dim)
            

        logger.debug(f'{query.shape=}')

        # if(attn_mask is not None):
        #     logger.debug(f'Kv cache: {self.kv_cache}, Reset cache: {reset_cache}')
        #     logger.debug(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim) # (batch_size, seq_length, num_heads, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim) # (batch_size, seq_length, num_heads, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim) # (batch_size, seq_length, num_heads, head_dim)

        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        K = K.transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        V = V.transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)

        logger.debug(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")

        # Scaled dot-product attention
        K_transposed = K.transpose(-2, -1) # (batch_size, num_heads, head_dim, seq_length_k)
        head_dim_sqrt = self.head_dim ** 0.5
        matmul = torch.matmul(Q, K_transposed)  # (batch_size, num_heads, seq_length_q, seq_length_k) 
        scores = matmul / (head_dim_sqrt)  # (batch_size, num_heads, seq_length_q, seq_length_k)

        scores = self.apply_attention_mask(scores, attn_mask)  # (batch_size, num_heads, seq_length_q, seq_length_k)
        
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_length_q, seq_length_k)
        self.map_attention = attn_weights  # Armazena os pesos de atenção para depuração

        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_length_q, head_dim)

        logger.debug(f"Attention output shape: {attn_output.shape}")

        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous() # (batch_size, seq_length_q, num_heads, head_dim)
        logger.debug(f"Attention output after transpose shape: {attn_output.shape}")
        
        attn_output = attn_output.view(batch_size, -1, self.num_heads * self.head_dim)  # (batch_size, seq_length_q, model_dim)
        
        logger.debug(f"Attention output after concatenation shape: {attn_output.shape}")
        output = self.out(attn_output)  # (batch_size, seq_length_q, model_dim)
        return output