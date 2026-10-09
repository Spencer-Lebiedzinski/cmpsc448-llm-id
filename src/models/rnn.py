"""BiLSTM classifier: embedding -> (Bi)LSTM over packed sequences ->
[masked max-pool over time ; final hidden states] -> dropout -> linear."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_dim=128, hidden_dim=128, num_layers=1,
                 bidirectional=True, dropout=0.3, pad_idx=0):
        super().__init__()
        self.bidirectional = bidirectional
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        d = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * d, num_classes)

    def forward(self, ids, lengths):
        x = self.dropout(self.emb(ids))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, _) = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))  # (B, T, d)
        mask = torch.arange(ids.size(1), device=ids.device)[None, :] < lengths[:, None]
        maxpool = out.masked_fill(~mask[..., None], -1e4).max(dim=1).values
        last = torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]
        return self.fc(self.dropout(torch.cat([maxpool, last], dim=1)))
