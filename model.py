import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Simple LSTM-based classifier for sequences of categorical event types.
    Input: (batch, seq_len) of token IDs
    Output: (batch,) binary classification logits
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, lengths=None):
        # x: (batch, seq_len) token ids
        emb = self.embedding(x)
        if lengths is not None:
            # pack sequence for variable length
            packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out_packed, (hn, cn) = self.lstm(packed)
            # take last hidden of last layer
            h_last = hn[-1]
        else:
            out, (hn, cn) = self.lstm(emb)
            h_last = hn[-1]
        logits = self.fc(h_last).squeeze(-1)
        return logits


class Encoder(nn.Module):
    """
    Encoder that produces a fixed-size embedding per sequence.
    Can be used as a backbone for contrastive learning, nearest neighbor lookups,
    or as an input to schedulers.
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.1, projection_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, projection_dim)
        )

    def forward(self, x, lengths=None):
        emb = self.embedding(x)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out_packed, (hn, cn) = self.lstm(packed)
            h_last = hn[-1]
        else:
            out, (hn, cn) = self.lstm(emb)
            h_last = hn[-1]
        z = self.proj(h_last)
        return z


class ProjectionHead(nn.Module):
    """Optional projection head used for contrastive losses.
    Maps encoder output to a lower-dimensional embedding used by NN or contrastive objectives.
    """
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, out_dim)
        )

    def forward(self, x):
        return self.net(x)
