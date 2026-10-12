"""The two required models: a text CNN and a BiLSTM."""
import torch
import torch.nn as nn


class TextCNN(nn.Module):
    """Kim (2014) style CNN. Parallel conv filters of width 3/4/5 over word embeddings, max pool over time."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, n_filters=128, widths=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, w, padding=w // 2) for w in widths])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(widths), n_classes)

    def forward(self, x, lengths):
        e = self.emb(x).transpose(1, 2)  # B x E x T
        mask = (x != 0).unsqueeze(1)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(e))[:, :, : x.size(1)]
            h = h.masked_fill(~mask, -1e4)
            pooled.append(h.max(dim=2).values)
        return self.fc(self.drop(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    """1 layer bidirectional LSTM, then mean pooling + max pooling over the hidden states."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, hidden=128, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden, n_classes)

    def forward(self, x, lengths):
        # not using pack_padded_sequence because its ~8x slower on the mac gpu (mps).
        # batches are length bucketed so there is very little padding, and pooling ignores pad positions
        e = self.drop(self.emb(x))
        out, _ = self.lstm(e)
        mask = (x != 0).unsqueeze(2).float()
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        mx = out.masked_fill(mask == 0, -1e4).max(1).values
        return self.fc(self.drop(torch.cat([mean, mx], dim=1)))


def build_model(name, vocab_size, n_classes):
    return {"cnn": TextCNN, "lstm": BiLSTM}[name](vocab_size, n_classes)
