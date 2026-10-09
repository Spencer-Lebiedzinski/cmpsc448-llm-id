"""Text CNN classifier (Kim, 2014): embedding -> parallel 1D convs (several widths)
-> ReLU -> masked max-over-time pooling -> dropout -> linear."""
import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_dim=128, num_filters=128,
                 kernel_sizes=(3, 4, 5), dropout=0.5, pad_idx=0):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, num_filters, k) for k in self.kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(self.kernel_sizes), num_classes)

    def forward(self, ids, lengths):
        # ids: (B, T) with T >= max(kernel_sizes) (collate pads to guarantee this)
        x = self.dropout(self.emb(ids)).transpose(1, 2)  # (B, E, T)
        pooled = []
        for k, conv in zip(self.kernel_sizes, self.convs):
            h = torch.relu(conv(x))  # (B, F, T-k+1)
            pos = torch.arange(h.size(2), device=ids.device)[None, :]
            valid = pos < (lengths[:, None] - k + 1).clamp(min=1)  # ignore windows over padding
            h = h.masked_fill(~valid[:, None, :], 0.0)
            pooled.append(h.max(dim=2).values)
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))
