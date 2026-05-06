import torch
import torch.nn as nn
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add handler to output logs to console
# handler = logging.StreamHandler()
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
        pad_idx
    ):
        super(LSTM, self).__init__()

        self.encoder = nn.LSTM(
            embedding_dim,
            encoder_hidden_dim,
            num_layers=encoder_num_layers,
            dropout=encoder_dropout,
            bidirectional=encoder_bidirectional,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            embedding_dim,
            decoder_hidden_dim,
            num_layers=decoder_num_layers,
            dropout=decoder_dropout,
            batch_first=True
        )

        num_directions = 2 if encoder_bidirectional else 1

        self.proj_hidden_h = nn.Linear(
            encoder_hidden_dim * num_directions, decoder_hidden_dim
        )

        self.proj_hidden_c = nn.Linear(
            encoder_hidden_dim * num_directions, decoder_hidden_dim
        )

        self.fc_out = nn.Linear(decoder_hidden_dim, vocab_size)

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]

        logger.info(f'Src shape: {src.shape}') # torch.Size([64, 45])
        logger.info(f'Tgt shape: {tgt.shape}') # torch.Size([64, 45])

        embedded_src = self.embedding(src) 
        embedded_tgt = self.embedding(tgt)


        logger.info(f"Embedded src shape: {embedded_src.shape}") # torch.Size([64, 45, 768])
        logger.info(f"Embedded tgt shape: {embedded_tgt.shape}") # torch.Size([64, 45, 768])

        _, (h, c) = self.encoder(embedded_src)
        h: torch.Tensor
        c: torch.Tensor

        logger.info(f"Encoder hidden state shape: {h.shape}") # torch.Size([6, 64, 2048])
        logger.info(f"Encoder cell state shape: {c.shape}") # torch.Size([6, 64, 2048])

        # encoder_outputs: [src_len, batch_size, encoder_hidden_dim * num_directions]
        # hidden: [encoder_num_layers * num_directions, batch_size, encoder_hidden_dim]
        # cell: [encoder_num_layers * num_directions, batch_size, encoder_hidden_dim]
        num_directions = 2 if self.encoder.bidirectional else 1
        batch_size = h.size(1)

        h = h.view(self.encoder.num_layers, num_directions, batch_size, self.encoder.hidden_size)
        c = c.view(self.encoder.num_layers, num_directions, batch_size, self.encoder.hidden_size)

        h = h[-1]  # [num_directions, batch, hidden]
        c = c[-1] # [num_directions, batch, hidden]

        logger.info(f"Reshaped encoder hidden state shape: {h.shape}") # torch.Size([2, 64, 2048])
        logger.info(f"Reshaped encoder cell state shape: {c.shape}") # torch.Size([2, 64, 2048])

        h = h.permute(1, 0, 2).reshape(batch_size, -1)
        c = c.permute(1, 0, 2).reshape(batch_size, -1)

        logger.info(f"Reshaped encoder hidden state shape: {h.shape}") # torch.Size([64, 4096])
        logger.info(f"Reshaped encoder cell state shape: {c.shape}") # torch.Size([64, 4096])

        # projeta
        h = self.proj_hidden_h(h).unsqueeze(0)
        c = self.proj_hidden_c(c).unsqueeze(0)

        logger.info(f'Projected encoder to decoder hidden state shape: {h.shape}') # torch.Size(1, 64, 2048)
        logger.info(f'Projected encoder to decoder cell state shape: {c.shape}') # torch.Size(1, 64, 2048)

        # replica para todas as layers do decoder
        h = h.repeat(self.decoder.num_layers, 1, 1)
        c = c.repeat(self.decoder.num_layers, 1, 1)

        logger.info(f"Projected encoder hidden state shape: {h.shape}") # torch.Size([3, 64, 2048])
        logger.info(f"Projected encoder cell state shape: {c.shape}") # torch.Size([3, 64, 2048])

        outputs, _ = self.decoder(embedded_tgt, (h.contiguous(), c.contiguous()))

        logger.info(f"Decoder output shape: {outputs.shape}") # torch.Size([64, 44, 2048])

        predictions = self.fc_out(outputs)

        logger.info(f"Prediction shape: {predictions.shape}") # torch.Size([64, 44, 10000])


        return predictions
    
    def train_step(
            self, 
            src: torch.Tensor, 
            tgt: torch.Tensor, 
            criterion: nn.CrossEntropyLoss, 
            optimizer: torch.optim.Optimizer
        ) -> torch.Tensor:
        self.train()
        optimizer.zero_grad()
        output = self(src, tgt[:, :-1]) # [tgt_len, batch_size, vocab_size]
        output_dim = output.shape[-1] # (vocab_size)
        
        output = output.reshape(-1, output_dim) # [batch_size, tgt_len, decoder_hidden_dim]

        tgt = tgt[:, 1:].flatten() # [tgt_len * batch_size]
        loss: torch.Tensor = criterion(output, tgt)
        return loss
    
    def eval_step(self, src, tgt, criterion):
        self.eval()
        with torch.no_grad():
            output = self(src, tgt[:, :-1]) # [tgt_len, batch_size, vocab_size]
            output_dim = output.shape[-1] # (vocab_size)
            output = output.view(-1, output_dim) # [batch_size, tgt_len, decoder_hidden_dim]
            tgt = tgt[:, 1:].reshape(-1) # [tgt_len * batch_size]
            loss = criterion(output, tgt)
        return loss.item()