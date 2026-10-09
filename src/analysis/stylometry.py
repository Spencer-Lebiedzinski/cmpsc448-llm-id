"""RQ4: What characteristics distinguish LLM families?

Computes interpretable stylometric features per response, summarizes them per
family (table + boxplots), and fits a features-only logistic regression to show
how much of the CNN/RNN accuracy hand-crafted style features alone can explain.

Usage:
  python -m src.analysis.stylometry                 # all splits, writes results/rq4/
  python -m src.analysis.stylometry --out results/rq4_smoke

TODO (for the report):
  * pair with the train.py --ablation runs (mask_identity, strip_format, truncateN) to
    test whether removing a signal hurts the neural models;
  * top discriminative n-grams per family (e.g. chi^2 on TF-IDF) / saliency on CNN filters;
  * qualitative examples of misclassifications.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..common import load_split
from ..data.text import IDENTITY_RE

ROOT = Path(__file__).resolve().parents[2]
WORD_RE = re.compile(r"[A-Za-z']+")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
APOLOGY_RE = re.compile(r"\b(sorry|apologi[sz]e|i cannot|i can't|i'm not able|as an ai)\b", re.I)


def mtld_like_ttr(words, window=50):
    """Mean type-token ratio over fixed windows (length-robust lexical diversity)."""
    if len(words) < window:
        return len(set(words)) / max(1, len(words))
    return float(np.mean([len(set(words[i:i + window])) / window
                          for i in range(0, len(words) - window + 1, window)]))


def features(text: str) -> dict:
    words = [w.lower() for w in WORD_RE.findall(text)]
    lines = text.split("\n")
    sents = [s for s in re.split(r"[.!?]+\s", text) if s.strip()]
    return {
        "n_chars": len(text),
        "n_words": len(words),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "avg_sent_len_words": len(words) / max(1, len(sents)),
        "ttr": len(set(words)) / max(1, len(words)),
        "windowed_ttr": mtld_like_ttr(words),
        "n_lines": len(lines),
        "frac_bullet_lines": sum(bool(re.match(r"\s*([-*+]|\d+[.)])\s", l)) for l in lines) / max(1, len(lines)),
        "n_headers": sum(bool(re.match(r"\s*#{1,6}\s", l)) for l in lines),
        "n_bold": text.count("**") // 2,
        "n_code_blocks": text.count("```") // 2,
        "n_exclaim": text.count("!"),
        "n_question": text.count("?"),
        "n_emoji": len(EMOJI_RE.findall(text)),
        "has_apology_refusal": int(bool(APOLOGY_RE.search(text))),
        "mentions_vendor_or_model": int(bool(IDENTITY_RE.search(text))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "rq4"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    splits = {s: load_split(s) for s in ("train", "val", "test")}
    feats = {s: pd.DataFrame([features(t) for t in df["output"]]) for s, df in splits.items()}
    full = pd.concat([feats[s].assign(family=splits[s]["family"].values) for s in splits], ignore_index=True)

    summary = full.groupby("family").median(numeric_only=True).T
    summary_mean = full.groupby("family").mean(numeric_only=True).T
    (out / "feature_medians_by_family.md").write_text(
        "# Median stylometric features per family\n\n" + summary.round(3).to_markdown() +
        "\n\n# Mean stylometric features per family\n\n" + summary_mean.round(3).to_markdown() + "\n")

    plot_feats = ["n_words", "windowed_ttr", "avg_word_len", "frac_bullet_lines", "n_bold", "n_lines"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, f in zip(axes.flat, plot_feats):
        sns.boxplot(data=full, x="family", y=f, ax=ax, showfliers=False)
        ax.set_title(f)
    fig.tight_layout()
    fig.savefig(out / "feature_boxplots.png", dpi=130)
    plt.close(fig)

    # features-only classifier (train -> test)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(np.log1p(feats["train"].clip(lower=0)), splits["train"]["family"])
    pred = clf.predict(np.log1p(feats["test"].clip(lower=0)))
    y = splits["test"]["family"]
    res = {"features_only_logreg_test_accuracy": float(accuracy_score(y, pred)),
           "features_only_logreg_test_macro_f1": float(f1_score(y, pred, average="macro")),
           "n_test": int(len(y))}
    lr = clf[-1]
    coefs = pd.DataFrame(lr.coef_, index=lr.classes_, columns=feats["train"].columns).T
    (out / "logreg_coefficients.md").write_text(
        "# Standardized logistic-regression coefficients (features-only model)\n\n"
        + coefs.round(3).to_markdown() + "\n")
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(summary.round(2).to_string())
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
