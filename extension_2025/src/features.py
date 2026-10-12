"""Hand made stylometric features for RQ4."""
import re

import numpy as np
import pandas as pd

EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐✅]")
WORD_RE = re.compile(r"[A-Za-z']+")


def mattr(words, window=50):
    """moving average type token ratio, less sensitive to length than plain TTR"""
    if len(words) < window:
        return len(set(words)) / max(len(words), 1)
    return float(np.mean([len(set(words[i : i + window])) / window for i in range(0, len(words) - window + 1, 10)]))


def style_features(text):
    words = [w.lower() for w in WORD_RE.findall(text)]
    lines = text.split("\n")
    n_words = max(len(words), 1)
    sents = [s for s in re.split(r"[.!?]+\s", text) if s.strip()]
    return {
        "n_words": len(words),
        "log_words": np.log1p(len(words)),
        "n_lines": len(lines),
        "mattr": mattr(words),
        "avg_word_len": np.mean([len(w) for w in words]) if words else 0,
        "avg_sent_words": len(words) / max(len(sents), 1),
        "headers": sum(bool(re.match(r"\s*#{1,6}\s", l)) for l in lines),
        "bold": text.count("**") // 2,
        "bullets": sum(bool(re.match(r"\s*[-*•]\s", l)) for l in lines),
        "numbered": sum(bool(re.match(r"\s*\d+[.)]\s", l)) for l in lines),
        "code_blocks": text.count("```") // 2,
        "tables": sum(l.strip().startswith("|") for l in lines),
        "hr_rules": sum(l.strip() in ("---", "***", "___") for l in lines),
        "emoji_per_100w": 100 * len(EMOJI_RE.findall(text)) / n_words,
        "em_dash_per_100w": 100 * text.count("—") / n_words,
        "exclaim_per_100w": 100 * text.count("!") / n_words,
        "question_per_100w": 100 * text.count("?") / n_words,
        "colon_per_100w": 100 * text.count(":") / n_words,
        "latex": text.count("$") + text.count("\\("),
        "curly_apostrophe_share": text.count("\u2019") / max(text.count("\u2019") + text.count("'"), 1),
        "first_person_per_100w": 100 * sum(w in ("i", "i'm", "i'll", "i've", "me", "my") for w in words) / n_words,
        "you_per_100w": 100 * sum(w in ("you", "your", "you're") for w in words) / n_words,
    }


def feature_frame(texts):
    return pd.DataFrame([style_features(t) for t in texts])


# ---- text transforms used for the ablations ----

def strip_formatting(text):
    """remove markdown and layout: headers, bold, bullets, numbering, code fences, tables, emojis, newlines"""
    t = text.replace("```", " ").replace("**", "").replace("__", "")
    t = re.sub(r"(?m)^\s*#{1,6}\s*", "", t)
    t = re.sub(r"(?m)^\s*([-*•]|\d+[.)])\s+", "", t)
    t = re.sub(r"(?m)^\s*(---|\*\*\*|___)\s*$", "", t)
    t = t.replace("|", " ")
    t = EMOJI_RE.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def words_only(text):
    """only lowercase words, no punctuation, no formatting at all"""
    return " ".join(w.lower() for w in WORD_RE.findall(text))


def first_50_words(text):
    """kills the length signal and keeps only how the answer starts"""
    return " ".join(text.split(" ")[:50])


def drop_first_line(text):
    """removes the opening line (things like 'Of course!' or 'Okay, here is...')"""
    parts = text.strip().split("\n", 1)
    return parts[1] if len(parts) > 1 else text


def make_shuffle(seed=0):
    rng = np.random.default_rng(seed)

    def shuffle_words(text):
        """same bag of tokens, random order. tests if word order matters"""
        toks = text.split()
        return " ".join(toks[i] for i in rng.permutation(len(toks)))

    return shuffle_words
