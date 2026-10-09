"""Data loading / encoding / batching shared by train.py and eval.py."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .data.text import example_tokens

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(name: str, data_dir=PROCESSED) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / f"{name}.csv", keep_default_na=False)
    return df


def filter_domains(df: pd.DataFrame, domains):
    if not domains:
        return df
    return df[df["domain"].isin(domains)].reset_index(drop=True)


def stratified_subset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if not n or n >= len(df):
        return df
    frac = n / len(df)
    return (df.groupby("family").sample(frac=frac, random_state=seed)
              .sample(frac=1.0, random_state=seed).reset_index(drop=True))


def tokens_for(df: pd.DataFrame, cfg: dict):
    abl = [a for a in (cfg.get("ablation") or "none").split(",") if a]
    return [example_tokens(i, o, cfg["input_mode"], cfg["max_len"], cfg["max_input_len"], abl)
            for i, o in zip(df["input"], df["output"])]


class EncodedDataset(Dataset):
    def __init__(self, ids_list, labels):
        self.ids = ids_list
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def make_collate(pad_idx: int, min_len: int = 5):
    def collate(batch):
        seqs, labels = zip(*batch)
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        T = max(int(lengths.max()), min_len)
        out = torch.full((len(seqs), T), pad_idx, dtype=torch.long)
        for j, s in enumerate(seqs):
            out[j, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out, lengths, torch.tensor(labels, dtype=torch.long)
    return collate


def make_loader(df, cfg, vocab, label2id, shuffle):
    toks = tokens_for(df, cfg)
    ids = [vocab.encode(t) for t in toks]
    labels = [label2id[f] for f in df["family"]]
    return DataLoader(EncodedDataset(ids, labels), batch_size=cfg["batch_size"], shuffle=shuffle,
                      collate_fn=make_collate(vocab.pad_idx, max(cfg.get("kernel_sizes", [5]))))
