import torch
import torch.nn as nn
import logging

logger = logging.getLogger()
# logger.setLevel(logging.INFO)

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

        self.proj_hidden = nn.Linear(
            encoder_hidden_dim * 2 * encoder_num_layers if encoder_bidirectional 
            else encoder_hidden_dim * encoder_num_layers, 
            decoder_hidden_dim*decoder_num_layers
        )

        self.fc_out = nn.Linear(decoder_hidden_dim, vocab_size)

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

    def forward(self, src, tgt):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]

        embedded_src = self.embedding(src)
        embedded_tgt = self.embedding(tgt)

        # embedded_src: [src_len, batch_size, embedding_dim]
        # embedded_tgt: [tgt_len, batch_size, embedding_dim]

        encoder_outputs, (h, c) = self.encoder(embedded_src)
        h: torch.Tensor
        c: torch.Tensor

        # encoder_outputs: [src_len, batch_size, encoder_hidden_dim * num_directions]
        # hidden: [encoder_num_layers * num_directions, batch_size, encoder_hidden_dim]
        # cell: [encoder_num_layers * num_directions, batch_size, encoder_hidden_dim]
        h = h.view(h.size(1), -1) # [batch_size, encoder_hidden_dim * num_directions]
        c = c.view(c.size(1), -1) # [batch_size, encoder_hidden_dim * num_directions]
        h = self.proj_hidden(h)
        c = self.proj_hidden(c)

        # hidden: [batch_size, decoder_hidden_dim]

        h = h.view(h.size(0), self.decoder.num_layers, -1).permute(1, 0, 2) # [decoder_num_layers, batch_size, decoder_hidden_dim]
        c = c.view(c.size(0), self.decoder.num_layers, -1).permute(1, 0, 2) # [decoder_num_layers, batch_size, decoder_hidden_dim]

        outputs, _ = self.decoder(embedded_tgt, (h.contiguous(), c.contiguous()))
        # outputs: [tgt_len, batch_size, decoder_hidden_dim]

        predictions = self.fc_out(outputs)

        # predictions: [tgt_len, batch_size, vocab_size]

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
        output = output.view(-1, output_dim) # [tgt_len * batch_size, vocab_size]
        tgt = tgt[:, 1:].flatten() # [tgt_len * batch_size]
        loss: torch.Tensor = criterion(output, tgt)
        return loss
    
    def eval_step(self, src, tgt, criterion):
        self.eval()
        with torch.no_grad():
            output = self(src, tgt[:, :-1]) # [tgt_len, batch_size, vocab_size]
            output_dim = output.shape[-1] # (vocab_size)
            output = output.view(-1, output_dim) # [tgt_len * batch_size, vocab_size]
            tgt = tgt[:, 1:].reshape(-1) # [tgt_len * batch_size]
            loss = criterion(output, tgt)
        return loss.item()