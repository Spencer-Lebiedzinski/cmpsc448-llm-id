"""Shared tokenizer / vocabulary / text ablations used by BOTH the CNN and the RNN.

The tokenizer is intentionally simple (regex, word-level) and *keeps formatting
signals* (newlines -> <nl>, markdown markers like ``` ** ##) because formatting is
part of an LLM's "fingerprint" (RQ4). Ablations let you remove signals to measure
how much the classifier depends on them.
"""
import hashlib
import json
import random
import re
from collections import Counter
from typing import Dict, Iterable, List

PAD, UNK, SEP, NL = "<pad>", "<unk>", "<sep>", "<nl>"
SPECIALS = [PAD, UNK, SEP, NL]

TOKEN_RE = re.compile(r"```|\*\*|#{1,6}|\n|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\sA-Za-z\d]")


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    toks = []
    for m in TOKEN_RE.finditer(text or ""):
        t = m.group(0)
        if t == "\n":
            toks.append(NL)
        else:
            toks.append(t.lower() if lowercase else t)
    return toks


# ----------------------------------------------------------------------------- ablations (RQ4)
IDENTITY_RE = re.compile(
    r"\b(open ?ai|chat ?gpt|gpt[- ]?\d(\.\d)?(-turbo)?|gpt|anthropic|claude|google|deepmind|gemini|bard|"
    r"palm ?2?|meta( ai)?|llama ?\d?|mistral( ai)?|mixtral|qwen|alibaba|deepseek|lmsys|vicuna)\b",
    re.I,
)
MD_RE = re.compile(r"```[a-zA-Z]*|\*\*|__|^#{1,6}\s*|^\s*[-*+]\s+|^\s*\d+[.)]\s+", re.M)


def mask_identity(text: str) -> str:
    """Replace self-identifying vendor/model names (e.g. 'As an AI developed by OpenAI')."""
    return IDENTITY_RE.sub(" modelname ", text)


def strip_format(text: str) -> str:
    """Remove markdown markers / list bullets and collapse all whitespace (incl. newlines)."""
    return " ".join(MD_RE.sub(" ", text).split())


WORD_ONLY_RE = re.compile(r"[A-Za-z']+")


def words_only(text: str) -> str:
    """Keep only words: no punctuation, digits, markdown or line breaks (pure word choice)."""
    return " ".join(WORD_ONLY_RE.findall(text))


def shuffle_words(text: str) -> str:
    """Same bag of whitespace tokens in a random (but deterministic per text) order."""
    toks = text.split()
    random.Random(hashlib.md5(text.encode("utf-8")).hexdigest()).shuffle(toks)
    return " ".join(toks)


def apply_ablations(text: str, ablations: Iterable[str]) -> str:
    for a in ablations:
        if a == "mask_identity":
            text = mask_identity(text)
        elif a == "strip_format":
            text = strip_format(text)
        elif a == "words_only":
            text = words_only(text)
        elif a == "shuffle_words":
            text = shuffle_words(text)
        elif a in ("", "none") or a.startswith("truncate"):
            pass  # truncateN is applied at token level (see encode_example)
        else:
            raise ValueError(f"unknown ablation {a}")
    return text


def truncate_n(ablations: Iterable[str]):
    for a in ablations:
        if a.startswith("truncate"):
            return int(a[len("truncate"):])
    return None


# ----------------------------------------------------------------------------- vocab
class Vocab:
    def __init__(self, itos: List[str]):
        self.itos = itos
        self.stoi: Dict[str, int] = {t: i for i, t in enumerate(itos)}

    @classmethod
    def build(cls, token_lists: Iterable[List[str]], min_freq: int = 2, max_size: int = 30000):
        c = Counter()
        for toks in token_lists:
            c.update(toks)
        words = [w for w, f in c.most_common() if f >= min_freq and w not in SPECIALS]
        return cls(SPECIALS + words[: max_size - len(SPECIALS)])

    def encode(self, toks: List[str]) -> List[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in toks]

    @property
    def pad_idx(self):
        return self.stoi[PAD]

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))


def example_tokens(inp: str, out: str, mode: str, max_len: int, max_input_len: int,
                   ablations=()) -> List[str]:
    """Token sequence for one example under an input mode: 'input' | 'output' | 'both'."""
    inp_t = tokenize(apply_ablations(inp, ablations))
    out_t = tokenize(apply_ablations(out, ablations))
    tn = truncate_n(ablations)
    if tn is not None:
        out_t = out_t[:tn]
    if mode == "input":
        toks = inp_t[:max_len]
    elif mode == "output":
        toks = out_t[:max_len]
    elif mode == "both":
        inp_t = inp_t[:max_input_len]
        toks = inp_t + [SEP] + out_t[: max_len - len(inp_t) - 1]
    else:
        raise ValueError(mode)
    return toks or [UNK]
