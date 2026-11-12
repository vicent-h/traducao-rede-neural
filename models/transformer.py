import torch
import torch.nn as nn
from torch.nn import functional as F
from logging import getLogger
import logging

from utils.scheduler_sampling import LinearSchedulerSampling
from models.transformer_components.encoder import TransformerEncoderPreNorm, TransformerEncoder
from models.transformer_components.decoder import TransformerDecoderPreNorm, TransformerDecoder
from models.transformer_components.pe_sinusoidal import SinusoidalPositionalEncoding


logger = getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)
# logger.addHandler(handler)

IGNORE_INDEX = 0  # Default ignore index for loss computation

    
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
            separate_embedding=False
        ):
        super(Transformer, self).__init__()
        self.kv_cache = kv_cache
        self.encoder = TransformerEncoderPreNorm(
            embedding_dim, 
            encoder_hidden_dim, 
            encoder_num_heads, 
            encoder_num_layers, 
            encoder_dropout
            )
        self.decoder = TransformerDecoderPreNorm(
            embedding_dim, 
            decoder_hidden_dim, 
            decoder_num_heads, 
            decoder_num_layers, 
            decoder_dropout, 
            vocab_size,
            use_cache=kv_cache,
            )

        self.separate_embedding = separate_embedding
        if separate_embedding:
            self.embedding_src = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.embedding_tgt = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.positional_encoding = SinusoidalPositionalEncoding(embedding_dim)

    def reset_cache(self):
        # Reset any kv caches present in decoder layers (self and cross attention)
        logger.debug("----*Resetting KV cache for all decoder layers.*----")
        self.decoder.reset_cache()

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
        decoder_input = tgt[:, :-1]
        labels        = tgt[:, 1:].clone()
        labels[:, 0] = IGNORE_INDEX  # Ignore <2pt>/<2es>
        
        self.train()
        output = self(src, decoder_input)  # Exclude the last token for teacher forcing

        output_dim = output.shape[-1] # (vocab_size)
        
        output = output.reshape(-1, output_dim) # [batch_size * tgt_len, vocab_size]

        logger.debug(f'Output shape after reshape train step: {output.shape}')

        labels = labels.flatten() # [tgt_len * batch_size]
        loss: torch.Tensor = criterion(output, labels)

        return loss
    
    def eval_step(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            criterion: nn.Module,
            bos_token_id: int = 5
    ) -> torch.Tensor:
        decoder_input = tgt[:, :-1]
        labels        = tgt[:, 1:].clone()
        labels[:, 0] = IGNORE_INDEX  # Ignore <2pt>/<2es>

        self.eval()
        with torch.no_grad():
            output = self.forward(src, decoder_input)

            output_dim = output.shape[-1] # (vocab_size)
            
            output = output.reshape(-1, output_dim) # [batch_size * tgt_len, vocab_size]

            logger.debug(f'Output shape after reshape eval step: {output.shape}')

            labels = labels.flatten() # [tgt_len * batch_size]
            loss: torch.Tensor = criterion(output, labels)

        return loss.item(), loss.item()
    
    def predict(
            self,
            src: torch.Tensor,
            bos_token_id: int = 5,
            eos_token_id: int = 6,
            max_len: int = 128,
            kv_cache = False,
            reset_cache: bool = False,
            input_decoder: torch.Tensor = None
    ):
        self.eval()
        
        # Reset cache at the beginning of each prediction
        if reset_cache:
            self.reset_cache()

        with torch.no_grad():
            memory = self.encode(src)
            batch_size = src.size(0)
            device = src.device
            if input_decoder is None:
                input_decoder = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
                outputs = [[bos_token_id] for _ in range(batch_size)]
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            # positional encoding buffer is stored in self.positional_encoding.pe
            pos_enc = self.positional_encoding.pe[:, :input_decoder.size(1), :].to(device)

            seq_len = input_decoder.size(1)

            outputs = input_decoder.cpu().tolist()

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