import torch
import torch.nn as nn
from torch.nn import functional as F
from logging import getLogger
import logging

from utils.scheduler_sampling import LinearSchedulerSampling


logger = getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)
# logger.addHandler(handler)

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

    def forward(
            self, 
            query: torch.Tensor, 
            key: torch.Tensor, 
            value: torch.Tensor, 
            attn_mask: torch.Tensor=None
            ):
        batch_size = query.size(0)
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

            # Queries: allow full or single-step queries; compute normally.
            Q = self.query(query)
        elif self.cross_attn_cache:
            if self.k_cache is None or self.v_cache is None:
                self.k_cache = self.key(key)      # (batch_size, seq_length, model_dim)
                self.v_cache = self.value(value)  # (batch_size, seq_length, model_dim)
            K = self.k_cache
            V = self.v_cache
            Q = self.query(query)
        else:
            K = self.key(key)      # (batch_size, seq_length, model_dim)
            V = self.value(value)  # (batch_size, seq_length, model_dim)
            Q = self.query(query)  # (batch_size, seq_length, model_dim)
            

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
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_length_q, seq_length_k)
        
        if attn_mask is not None:
            logger.debug(f"Attention mask shape: {attn_mask.shape}, Attention weights shape: {scores.shape}")
            scores = scores.masked_fill(attn_mask == float('-inf'), float('-inf'))

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

    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        # Disable kv_cache for encoder self-attention to avoid cache persisting across
        # different batches (which may have different batch sizes).
        self.self_attn = MultiHeadAttention(model_dim, num_heads, kv_cache=False)
        self.linear1 = nn.Linear(model_dim, model_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(model_dim * 4, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # src shape: (seq_length, batch_size, model_dim)
        src2 = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1, kv_cache=False, cross_attn_cache=False):
        super(TransformerDecoderLayer, self).__init__()
        # self-attention in decoder should not use kv cache when generating
        # (we accumulate the full tgt sequence and do not rely on incremental self-attn caching)
        self.self_attn = MultiHeadAttention(model_dim, num_heads, kv_cache=kv_cache, cross_attn_cache=False)
        # Cross-attention should use the full encoder memory every step;
        # do not enable kv_cache for cross-attention (it would incorrectly
        # append encoder keys across decoding steps).
        self.multihead_attn = MultiHeadAttention(model_dim, num_heads, kv_cache=False, cross_attn_cache=cross_attn_cache)
        self.linear1 = nn.Linear(model_dim, model_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(model_dim * 4, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.kv_cache = kv_cache

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt, memory):
        # tgt shape: (batch_size, seq_length_tgt, model_dim)
        # memory shape: (batch_size, seq_length_src, model_dim)
        if self.kv_cache:
            mask = None
        else:
            mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

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

class TransformerDecoder(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            model_dim, 
            num_heads, 
            num_layers, 
            dropout=0.1, 
            vocab_size=50_000,
            kv_cache=False,
            cross_attn_cache=False
        ):
        super(TransformerDecoder, self).__init__()
        self.model_dim = model_dim
        self.transformer_decoder = nn.ModuleList(
            [TransformerDecoderLayer(model_dim, num_heads, dropout, kv_cache=kv_cache, cross_attn_cache=cross_attn_cache) for _ in range(num_layers)]
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
    
class Transformer(nn.Module):
    def __init__(
            self, 
            embedding_dim,
            encoder_hidden_dim,
            decoder_hidden_dim,
            encoder_num_layers,
            decoder_num_layers,
            encoder_num_heads,
            decoder_num_heads,
            encoder_dropout,
            decoder_dropout,
            vocab_size=50_000,
            pad_idx=0,
            kv_cache=False,
            cross_attn_cache=False,
            separate_embedding=False
        ):
        super(Transformer, self).__init__()
        self.kv_cache = kv_cache
        self.encoder = TransformerEncoder(
            embedding_dim, 
            encoder_hidden_dim, 
            encoder_num_heads, 
            encoder_num_layers, 
            encoder_dropout
            )
        self.decoder = TransformerDecoder(
            embedding_dim, 
            decoder_hidden_dim, 
            decoder_num_heads, 
            decoder_num_layers, 
            decoder_dropout, 
            vocab_size,
            kv_cache=kv_cache,
            cross_attn_cache=cross_attn_cache
            )

        self.separate_embedding = separate_embedding
        if separate_embedding:
            self.embedding_src = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.embedding_tgt = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.positional_encoding = SinusoidalPositionalEncoding(embedding_dim)

    def reset_kv_cache(self):
        # Reset any kv caches present in decoder layers (self and cross attention)
        logger.debug("----*Resetting KV cache for all decoder layers.*----")
        for layer in self.decoder.transformer_decoder:
            layer.self_attn.reset_kv_cache()
            layer.multihead_attn.reset_kv_cache()
            layer.actual_size_tgt = 0

    def encode(
            self,
            src: torch.Tensor
        ) -> torch.Tensor:
        if self.separate_embedding:
            src = self.embedding_src(src)
        else:
            src = self.embedding(src)
        
        src = src + self.positional_encoding(src)
        src = src.float()

        memory = self.encoder(src)  # (batch_size, seq_length_src, model_dim)

        return memory

    def decode(
            self, 
            tgt: torch.Tensor, 
            memory: torch.Tensor, 
            pe_custom: torch.Tensor = None
        ) -> torch.Tensor:
        if self.separate_embedding:
            tgt = self.embedding_tgt(tgt)
        else:
            tgt = self.embedding(tgt)

        if pe_custom is not None:
            tgt = tgt + pe_custom
        else:
            tgt += self.positional_encoding(tgt)
        tgt = tgt.float()

        output = self.decoder(tgt, memory)  # (batch_size, seq_length_tgt, output_dim)

        return output
    
    def forward(
            self, 
            src: torch.Tensor, 
            tgt: torch.Tensor):

        memory = self.encode(src)
        # src shape: (batch_size, seq_length_src, input_dim)


        output = self.decode(tgt, memory)
        # tgt shape: (batch_size, seq_length_tgt, output_dim)
        return output
    
    def train_step(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            criterion: nn.Module,
            teacher_forcing: bool = True,
            bos_token_id: int = 5,
            scheduler_sampling: LinearSchedulerSampling = None
    ) -> torch.Tensor:
        self.train()
        output = self(src, tgt[:, :-1])  # Exclude the last token for teacher forcing

        output_dim = output.shape[-1] # (vocab_size)
        
        output = output.reshape(-1, output_dim) # [batch_size * tgt_len, vocab_size]

        logger.debug(f'Output shape after reshape train step: {output.shape}')

        tgt = tgt[:, 1:].flatten() # [tgt_len * batch_size]
        loss: torch.Tensor = criterion(output, tgt)

        return loss
    
    def eval_step(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            criterion: nn.Module,
            bos_token_id: int = 5
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            output = self.forward(src, tgt[:, :-1])  # Exclude the last token for evaluation

            output_dim = output.shape[-1] # (vocab_size)
            
            output = output.reshape(-1, output_dim) # [batch_size * tgt_len, vocab_size]

            logger.debug(f'Output shape after reshape eval step: {output.shape}')

            tgt = tgt[:, 1:].flatten() # [tgt_len * batch_size]
            loss: torch.Tensor = criterion(output, tgt)

        return loss.item(), loss.item()
    
    def predict(
            self,
            src: torch.Tensor,
            bos_token_id: int = 5,
            eos_token_id: int = 6,
            max_len: int = 128,
            kv_cache = False,
            reset_cache: bool = False
    ):
        self.eval()
        
        # Reset cache at the beginning of each prediction
        if reset_cache:
            self.reset_kv_cache()

        with torch.no_grad():
            memory = self.encode(src)
            batch_size = src.size(0)
            device = src.device
            input_decoder = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            # positional encoding buffer is stored in self.positional_encoding.pe
            pos_enc = self.positional_encoding.pe[:, :1, :].to(device)

            seq_len = 1

            outputs = [[bos_token_id] for _ in range(batch_size)]

            for _ in range(max_len):

                logger.debug(f'{seq_len=}, {input_decoder.shape=}')
                output = self.decode(input_decoder, memory, pos_enc)  # (B, seq_len, vocab_size)
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)

                # Pass entire accumulated sequence to decoder, not just the new token
                if kv_cache:
                    pos_enc = self.positional_encoding.pe[:, seq_len-1:seq_len, :].to(device)
                    input_decoder = next_token
                else:
                    pos_enc = None
                    input_decoder = torch.cat((input_decoder, next_token), dim=1)

                seq_len += 1

                # Preenche apenas as sequências ainda ativas
                for i in range(batch_size):
                    if not finished[i]:
                        outputs[i].append(int(next_token[i, 0].item()))

                finished = finished | (next_token.squeeze(1) == eos_token_id)
                if finished.all():
                    break


        return outputs