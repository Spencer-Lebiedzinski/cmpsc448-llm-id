"""Tokenizer, vocab and the data split shared by every experiment."""
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA = "data/llm_dataset.parquet"
SEED = 448

PAD, UNK, SEP, NL = "<pad>", "<unk>", "<sep>", "<nl>"

# keep markdown symbols and newlines as their own tokens since they might be part of the "style"
TOKEN_RE = re.compile(r"\n|```|\*\*|#{1,6}|[A-Za-z]+(?:'[a-z]+)?|\d|[^\sA-Za-z\d]")


def tokenize(text):
    return [NL if t == "\n" else t.lower() for t in TOKEN_RE.findall(text)]


def build_vocab(token_lists, max_size=30000, min_freq=2):
    counts = Counter(t for toks in token_lists for t in toks)
    itos = [PAD, UNK, SEP] + [t for t, c in counts.most_common(max_size) if c >= min_freq and t not in (PAD, UNK, SEP)]
    return {t: i for i, t in enumerate(itos)}


def encode(tokens, vocab, max_len):
    ids = [vocab.get(t, 1) for t in tokens[:max_len]]
    return ids if ids else [1]


def load_data():
    return pd.read_parquet(DATA)


def split_data(df, seed=SEED):
    """70/15/15 split grouped by prompt, so the same prompt never shows up in two splits
    (both answers of one arena battle share a prompt, that would leak)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr, rest = next(gss.split(df, groups=df.prompt_id))
    rest_df = df.iloc[rest]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    va, te = next(gss2.split(rest_df, groups=rest_df.prompt_id))
    return df.iloc[tr].reset_index(drop=True), rest_df.iloc[va].reset_index(drop=True), rest_df.iloc[te].reset_index(drop=True)


def make_tokens(df, mode, in_len=100, out_len=400, transform=None):
    """mode is input / output / both. transform is an optional function applied to the output text (for ablations)."""
    outs = df.LLM_output if transform is None else df.LLM_output.map(transform)
    if mode == "input":
        return [tokenize(t)[:in_len] for t in df.LLM_input]
    if mode == "output":
        return [tokenize(t)[:out_len] for t in outs]
    return [tokenize(i)[:in_len] + [SEP] + tokenize(o)[:out_len] for i, o in zip(df.LLM_input, outs)]


def max_len_for(mode, in_len=100, out_len=400):
    return {"input": in_len, "output": out_len, "both": in_len + out_len + 1}[mode]


LABELS = sorted(["Claude", "DeepSeek", "GPT", "Gemini", "Grok", "Llama", "Mistral", "Qwen"])
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def labels_of(df):
    return np.array([LABEL2ID[x] for x in df.LLM_name])
