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
    
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.out = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)

        Q = self.query(query)  # (batch_size, seq_length, model_dim)
        K = self.key(key)      # (batch_size, seq_length, model_dim)
        V = self.value(value)  # (batch_size, seq_length, model_dim)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim) # (batch_size, seq_length, num_heads, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim) # (batch_size, seq_length, num_heads, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim) # (batch_size, seq_length, num_heads, head_dim)

        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        K = K.transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        V = V.transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_length_q, seq_length_k)
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_length_q, seq_length_k)
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == float('-inf'), 0)
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_length_q, head_dim)

        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)  # (batch_size, seq_length_q, model_dim)
        output = self.out(attn_output)  # (batch_size, seq_length_q, model_dim)
        return output

    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads)
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
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads)
        self.multihead_attn = MultiHeadAttention(model_dim, num_heads)
        self.linear1 = nn.Linear(model_dim, model_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(model_dim * 4, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt, memory):
        # tgt shape: (seq_length_tgt, batch_size, model_dim)
        # memory shape: (seq_length_src, batch_size, model_dim)
        mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
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
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(model_dim)
        self.transformer_encoder = nn.ModuleList(
            [TransformerEncoderLayer(model_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        x = self.embedding(x)  # (batch_size, seq_length, model_dim)
        seq_length = x.size(1)
        x = x + self.positional_encoding(x)  # Add positional encoding
        x = x.permute(1, 0, 2)  # (seq_length, batch_size, model_dim) for transformer
        output = self.transformer_encoder(x)  # (seq_length, batch_size, model_dim)
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, model_dim)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(output_dim, model_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(model_dim)
        self.transformer_decoder = nn.ModuleList(
            [TransformerDecoderLayer(model_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(model_dim, output_dim)

    def forward(self, x, memory):
        # x shape: (batch_size, seq_length_tgt, output_dim)
        # memory shape: (batch_size, seq_length_src, model_dim)
        x = self.embedding(x)  # (batch_size, seq_length_tgt, model_dim)
        seq_length = x.size(1)
        x = x + self.positional_encoding(x)  # Add positional encoding
        x = x.permute(1, 0, 2)  # (seq_length_tgt, batch_size, model_dim) for transformer
        memory = memory.permute(1, 0, 2)  # (seq_length_src, batch_size, model_dim)
        output = self.transformer_decoder(x, memory)  # (seq_length_tgt, batch_size, model_dim)
        output = output.permute(1, 0, 2)  # (batch_size, seq_length_tgt, model_dim)
        output = self.output_layer(output)  # (batch_size, seq_length_tgt, output_dim)
        return output
    
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(input_dim, model_dim, num_heads, num_layers, dropout)
        self.decoder = TransformerDecoder(output_dim, model_dim, num_heads, num_layers, dropout)

    def forward(self, src, tgt):
        # src shape: (batch_size, seq_length_src, input_dim)
        # tgt shape: (batch_size, seq_length_tgt, output_dim)
        memory = self.encoder(src)  # (batch_size, seq_length_src, model_dim)
        output = self.decoder(tgt, memory)  # (batch_size, seq_length_tgt, output_dim)
        return output