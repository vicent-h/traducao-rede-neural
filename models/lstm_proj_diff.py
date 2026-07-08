import torch
import torch.nn as nn
from logging import getLogger
import logging

from utils.scheduler_sampling import LinearSchedulerSampling


logger = getLogger(__name__)
logger.setLevel(logging.WARNING)

# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)
# logger.addHandler(handler)


class LSTM(nn.Module):
    def __init__(
        self,
        embedding_dim,
        encoder_hidden_dim,
        decoder_hidden_dim,
        encoder_num_layers,
        decoder_num_layers,
        encoder_dropout,
        decoder_dropout,
        encoder_bidirectional,
        vocab_size,
        pad_idx,
        attention=False
    ):
        super(LSTM, self).__init__()
        self.attention = attention

        num_directions = 2 if encoder_bidirectional else 1

        self.encoder = nn.LSTM(
            embedding_dim,
            encoder_hidden_dim,
            num_layers=encoder_num_layers,
            dropout=encoder_dropout,
            bidirectional=encoder_bidirectional,
            batch_first=False
        )

        if attention:
            self.decoder = nn.LSTM(
                embedding_dim + decoder_hidden_dim,
                decoder_hidden_dim,
                num_layers=decoder_num_layers,
                dropout=decoder_dropout,
                batch_first=False
            )
            self.fc_out = nn.Linear(decoder_hidden_dim, vocab_size)

            self.proj_encoder_output = nn.Linear(
                encoder_hidden_dim * num_directions, decoder_hidden_dim
            )

        else:
            self.fc_out = nn.Linear(decoder_hidden_dim, vocab_size)
            self.decoder = nn.LSTM(
                embedding_dim,
                decoder_hidden_dim,
                num_layers=decoder_num_layers,
                dropout=decoder_dropout,
                batch_first=False
            )

        

        self.proj_hidden_h = nn.Linear(
            encoder_hidden_dim * num_directions, decoder_hidden_dim
        )

        self.proj_hidden_c = nn.Linear(
            encoder_hidden_dim * num_directions, decoder_hidden_dim
        )




        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

    def encode(self, src):

        logger.debug(f'Src shape: {src.shape}')  # [batch_size, src_len]

        embedded_src = self.embedding(src)
        embedded_src = embedded_src.permute(1, 0, 2)

        logger.debug(f'Embedded src shape: {embedded_src.shape}')
        # [src_len, batch_size, embedding_dim]

        encoder_outputs, (h, c) = self.encoder(embedded_src)

        logger.debug(f'Encoder hidden state shape: {h.shape}')
        logger.debug(f'Encoder cell state shape: {c.shape}')

        if self.encoder.bidirectional:
            h = torch.cat((h[-2], h[-1]), dim=1)
            c = torch.cat((c[-2], c[-1]), dim=1)
        else:
            h = h[-1]
            c = c[-1]

        logger.debug(f'Concatenated encoder hidden shape: {h.shape}')
        logger.debug(f'Concatenated encoder cell shape: {c.shape}')

        h0 = self.proj_hidden_h(h).unsqueeze(0)
        c0 = self.proj_hidden_c(c).unsqueeze(0)

        logger.debug(f'Projected encoder hidden shape: {h0.shape}')
        logger.debug(f'Projected encoder cell shape: {c0.shape}')

        if self.decoder.num_layers > 1:
            h = torch.cat([
                h0,
                h0.new_zeros(self.decoder.num_layers - 1, h0.size(1), h0.size(2))
            ], dim=0).contiguous()
            c = torch.cat([
                c0,
                c0.new_zeros(self.decoder.num_layers - 1, c0.size(1), c0.size(2))
            ], dim=0).contiguous()
        else:
            h = h0
            c = c0

        if self.attention:
            encoder_outputs = self.proj_encoder_output(encoder_outputs)
            # [src_len, batch_size, decoder_hidden_dim]

        logger.debug(f'Decoder initial hidden shape: {h.shape}')
        logger.debug(f'Decoder initial cell shape: {c.shape}')
        # [decoder_num_layers, batch_size, decoder_hidden_dim]

        return encoder_outputs, h, c


    def decode(self, tgt, h, c, encoder_outputs=None):

        logger.debug(f'Tgt shape: {tgt.shape}')
        # [batch_size, tgt_len]

        logger.debug(f'Input decoder hidden shape: {h.shape}')
        # [num_layers, batch_size, decoder_hidden_dim]

        logger.debug(f'Input decoder cell shape: {c.shape}')
        # [num_layers, batch_size, decoder_hidden_dim]

        embedded_tgt = self.embedding(tgt)
        # [batch_size, tgt_len, embedding_dim]

        if not self.attention:
            embedded_tgt = embedded_tgt.permute(1, 0, 2)
            # [tgt_len, batch_size, embedding_dim]

            logger.debug(f'Embedded tgt shape: {embedded_tgt.shape}')
            # [tgt_len, batch_size, embedding_dim]

            outputs, (h, c) = self.decoder(
                embedded_tgt,
                (h, c)
            )
            # outputs: [tgt_len, batch_size, decoder_hidden_dim]
            # h: [num_layers, batch_size, decoder_hidden_dim]
            # c: [num_layers, batch_size, decoder_hidden_dim]

            outputs = outputs.permute(1, 0, 2)
            # [batch_size, tgt_len, decoder_hidden_dim]

            predictions = self.fc_out(outputs)
            # [batch_size, tgt_len, vocab_size]

            return predictions, h, c
        
        encoder_outputs # [src_len, batch_size, decoder_hidden_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # [batch_size, src_len, decoder_hidden_dim]
        
        logits_total = []
        # list of [batch_size, 1, vocab_size]

        for i in range(embedded_tgt.size(1)):
            
            current_input = embedded_tgt[:, i:i+1, :] # [B, 1, E]
            # [batch_size, 1, embedding_dim]

            current_h = h[-1].unsqueeze(0) # [1, B, H]
            # [1, batch_size, decoder_hidden_dim]     

            score = torch.bmm(
                current_h.transpose(0,1),       # [B,1,H]
                encoder_outputs.transpose(1,2)  # [B,H,T]
            )

            attn_weights = torch.softmax(score, dim=-1) # [B, 1, T]
            # [batch_size, 1, seq_len]

            context = torch.bmm(
                attn_weights, # [batch_size, 1, seq_len]
                encoder_outputs # [batch_size, seq_len, decoder_hidden_dim]
            ) # [B, 1, T] x [B, T, H] -> [B, 1, H]

            decoder_input = torch.cat([current_input, context], dim=-1) # [B, 1, E+H]
            # [batch_size, 1, embedding_dim + decoder_hidden_dim]

            outputs, (h, c) = self.decoder(
                decoder_input.permute(1, 0, 2), # [1, B, E+H]
                (h, c)
            )
            # outputs: [1, batch_size, decoder_hidden_dim]
            # h: [num_layers, batch_size, decoder_hidden_dim]
            # c: [num_layers, batch_size, decoder_hidden_dim]

            outputs = outputs.permute(1, 0, 2) # [B, 1, H]

            logits = self.fc_out(outputs) # [B, 1, V]

            logits_total.append(logits) # [batch_size, 1, vocab_size]


        logits_total = torch.cat(logits_total, dim=1) # [B, T, V]
        # [batch_size, tgt_len, vocab_size]

        return logits_total, h, c
        # [batch_size, tgt_len, vocab_size], [num_layers, batch_size, decoder_hidden_dim], [num_layers, batch_size, decoder_hidden_dim]


    def forward(self, src, tgt):

        logger.debug('===== FORWARD START =====')

        encoder_outputs, h, c = self.encode(src)

        predictions, _, _ = self.decode(tgt, h, c, encoder_outputs)

        logger.debug('===== FORWARD END =====')

        return predictions

    def _select_next_token(self, predictions: torch.Tensor, prev_token: torch.Tensor) -> torch.Tensor:
        """
        Select next token avoiding repeating the previous token.

        If the top-1 (argmax) equals the previous token, return the top-2 token.
        predictions: [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
        prev_token: [batch_size, 1] or [batch_size]
        Returns: [batch_size]
        """
        # get logits for last time step if needed
        if predictions.dim() == 3:
            logits = predictions[:, -1]
        else:
            logits = predictions

        k = min(2, logits.size(-1))
        topk = logits.topk(k, dim=-1).indices  # [batch, k]
        top1 = topk[:, 0]
        top2 = topk[:, 1] if k > 1 else top1

        # normalize prev_token shape to [batch]
        prev = prev_token.squeeze(-1) if prev_token.dim() > 1 else prev_token

        selected = torch.where(top1 == prev, top2, top1)
        return selected
    
    def train_step(
            self, 
            src: torch.Tensor, 
            tgt: torch.Tensor, 
            criterion: nn.CrossEntropyLoss,
            teacher_forcing: bool = True,
            bos_token_id: int = 5,
            scheduler_sampling: LinearSchedulerSampling = None
        ) -> torch.Tensor:
        self.train()

        if teacher_forcing and scheduler_sampling.get_ratio() >= 1:
            output = self(src, tgt[:, :-1]) # torch.Size([64, 44, 10000])
        else:
            batch_size = src.size(0)
            max_len = tgt.size(1)
            device = src.device
            h, c = self.encode(src)
            current_token = torch.full(
                (batch_size, 1),
                bos_token_id,
                dtype=torch.long,
                device=device
            )

            outputs = []

            for t in range(max_len - 1):
                predictions, h, c = self.decode(
                    current_token,
                    h,
                    c
                )
                outputs.append(predictions)

                truth_token = tgt[:, t].unsqueeze(1)
                if teacher_forcing and scheduler_sampling.should_sample():
                    current_token = truth_token
                else:
                    next_sel = self._select_next_token(predictions, current_token)
                    current_token = next_sel.unsqueeze(1)
            output = torch.cat(outputs, dim=1)
            

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
            criterion: nn.CrossEntropyLoss,
            bos_token_id=5):
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            max_len = tgt.size(1)
            device = src.device
            encoder_outputs, h, c = self.encode(src)
            current_token = torch.full(
                (batch_size, 1),
                bos_token_id,
                dtype=torch.long,
                device=device
            )

            outputs = []

            for t in range(max_len - 1):
                predictions, h, c = self.decode(
                    current_token,
                    h,
                    c,
                    encoder_outputs
                )
                outputs.append(predictions)

                next_sel = self._select_next_token(predictions, current_token)
                current_token = next_sel.unsqueeze(1)
            output = torch.cat(outputs, dim=1)
            

            output_dim = output.shape[-1] # (vocab_size)
            
            output = output.reshape(-1, output_dim) # [batch_size * tgt_len, vocab_size]

            logger.debug(f'Output shape after reshape train step: {output.shape}')

            tgt_flat = tgt[:, 1:].flatten() # [tgt_len * batch_size]
            loss_no_tf: torch.Tensor = criterion(output, tgt_flat)

            output = self(src, tgt[:, :-1])
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)
            loss: torch.Tensor = criterion(output, tgt)
        return loss.item(), loss_no_tf.item()
    
    def predict(
        self,
        src,
        bos_token_id,
        eos_token_id,
        max_len=128
    ):

        self.eval()

        # src -> [batch_size, src_seq_len]

        batch_size = src.size(0)
        device = src.device

        with torch.no_grad():

            encoder_outputs, h, c = self.encode(src)

            # h -> [num_layers, batch_size, hidden_size]
            # c -> [num_layers, batch_size, hidden_size]

            current_token = torch.full(
                (batch_size, 1),
                bos_token_id,
                dtype=torch.long,
                device=device
            )

            # current_token -> [batch_size, 1]

            generated_tokens = []

            finished = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=device
            )

            # finished -> [batch_size]

            for _ in range(max_len):

                predictions, h, c = self.decode(
                    current_token,
                    h,
                    c,
                    encoder_outputs
                )

                # predictions -> [batch_size, 1, vocab_size]
                # h -> [num_layers, batch_size, hidden_size]
                # c -> [num_layers, batch_size, hidden_size]

                next_token = self._select_next_token(predictions, current_token)

                # predictions[:, -1] -> [batch_size, vocab_size]
                # next_token -> [batch_size]

                generated_tokens.append(next_token)

                # generated_tokens -> list de tensors [batch_size]

                finished |= (next_token == eos_token_id)

                # (next_token == eos_token_id) -> [batch_size]
                # finished -> [batch_size]

                if finished.all():
                    break

                current_token = next_token.unsqueeze(1)

                # current_token -> [batch_size, 1]

            generated_tokens = torch.stack(generated_tokens, dim=1)

            # generated_tokens -> [batch_size, generated_seq_len]

            return generated_tokens