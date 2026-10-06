import torch
import torch.nn as nn
import math

class TransformerTranslator(nn.Module):
    """Transformer-based CUDA → HIP translator. No RNN = no MIOpen issues."""

    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dim_ff=512, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.pos_enc(self.embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.embed(tgt) * math.sqrt(self.d_model))

        tgt_seq_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=tgt.device)

        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (tgt == 0)

        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc_out(out)

    def translate(self, src, max_len=200, sos_token=1, eos_token=2):
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device

            src_emb = self.pos_enc(self.embed(src) * math.sqrt(self.d_model))
            memory = self.transformer.encoder(src_emb)

            tgt = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)

            for _ in range(max_len):
                tgt_emb = self.pos_enc(self.embed(tgt) * math.sqrt(self.d_model))
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=device)
                out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                logits = self.fc_out(out[:, -1:])
                next_token = logits.argmax(dim=-1)
                tgt = torch.cat([tgt, next_token], dim=1)
                if (next_token == eos_token).all():
                    break
            return tgt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
