"""Extra report material: training-curve figure and top discriminative n-grams per family.

  python -m src.analysis.report_extras            # writes results/figures/, results/rq4/top_ngrams.md
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..common import load_split

ROOT = Path(__file__).resolve().parents[2]


def training_curves(runs_root: Path, out: Path, runs=("cnn_output", "rnn_output")):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for r in runs:
        h = json.loads((runs_root / r / "history.json").read_text())
        ep = [x["epoch"] for x in h]
        axes[0].plot(ep, [x["train_loss"] for x in h], marker="o", label=r)
        axes[1].plot(ep, [x["val_macro_f1"] for x in h], marker="o", label=r)
    axes[0].set_title("Train loss")
    axes[1].set_title("Validation macro-F1")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)


def top_ngrams(out_md: Path, k=15):
    tr = load_split("train")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000, sublinear_tf=True)
    X = vec.fit_transform(tr["output"])
    clf = LogisticRegression(max_iter=2000, C=4.0).fit(X, tr["family"])
    names = np.array(vec.get_feature_names_out())
    lines = ["# Top word n-grams per family (largest TF-IDF logistic-regression weights, train split)\n"]
    for fam, w in zip(clf.classes_, clf.coef_):
        lines.append(f"- **{fam}**: " + ", ".join(f"`{t}`" for t in names[np.argsort(-w)[:k]]))
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    args = ap.parse_args()
    figs = ROOT / "results" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    training_curves(Path(args.runs), figs)
    top_ngrams(ROOT / "results" / "rq4" / "top_ngrams.md")


if __name__ == "__main__":
    main()
