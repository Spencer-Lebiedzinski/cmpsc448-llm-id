from .cnn import TextCNN
from .rnn import BiLSTMClassifier


def build_model(cfg: dict, vocab_size: int, num_classes: int, pad_idx: int = 0):
    if cfg["model"] == "cnn":
        return TextCNN(vocab_size, num_classes, emb_dim=cfg["emb_dim"], num_filters=cfg["num_filters"],
                       kernel_sizes=tuple(cfg["kernel_sizes"]), dropout=cfg["dropout"], pad_idx=pad_idx)
    if cfg["model"] == "rnn":
        return BiLSTMClassifier(vocab_size, num_classes, emb_dim=cfg["emb_dim"], hidden_dim=cfg["hidden_dim"],
                                num_layers=cfg["num_layers"], bidirectional=not cfg.get("unidirectional", False),
                                dropout=cfg["dropout"], pad_idx=pad_idx)
    raise ValueError(cfg["model"])
