"""Makes all the figures + a summary table for the report from results/*.json."""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from text_utils import LABELS

R = "results"
F = "figures"
os.makedirs(F, exist_ok=True)
load = lambda n: json.load(open(f"{R}/{n}.json"))
exists = lambda n: os.path.exists(f"{R}/{n}.json")
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def confusion_plots():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, m in zip(axes, ["cnn", "lstm"]):
        cm = np.array(load(f"rq2_{m}_output")["test"]["confusion"], dtype=float)
        cm = cm / cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=7, color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=45)
        ax.set_yticks(range(len(LABELS)), LABELS)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{m.upper()} (output only), row normalized")
    fig.tight_layout()
    fig.savefig(f"{F}/rq1_confusion.png")


def curves():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for m in ["cnn", "lstm"]:
        h = pd.DataFrame(load(f"rq2_{m}_output")["history"])
        axes[0].plot(h.epoch, h.train_loss, marker="o", label=m.upper())
        axes[1].plot(h.epoch, h.val_macro_f1, marker="o", label=m.upper())
    axes[0].set_title("train loss")
    axes[1].set_title("validation macro F1")
    for a in axes:
        a.set_xlabel("epoch")
        a.legend()
    fig.tight_layout()
    fig.savefig(f"{F}/training_curves.png")


def rq2_bars():
    modes = ["input", "output", "both"]
    names = {"cnn": "CNN", "lstm": "BiLSTM", "tfidf": "TF-IDF + LR"}
    fig, ax = plt.subplots(figsize=(6.5, 3))
    w = 0.26
    for k, m in enumerate(names):
        acc = [load(f"rq2_{m}_{mode}")["test"]["accuracy"] for mode in modes]
        bars = ax.bar(np.arange(3) + (k - 1) * w, acc, w, label=names[m])
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.axhline(1 / 8, ls="--", c="gray", lw=1)
    ax.text(2.45, 1 / 8 + 0.01, "chance", fontsize=7, color="gray")
    ax.set_xticks(range(3), ["input only", "output only", "input + output"])
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{F}/rq2_bars.png")


def rq3_plot():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, held in zip(axes, ["creative_writing", "code"]):
        rows = []
        for m in ["cnn", "lstm"]:
            for kind in ["reference", "holdout"]:
                f1 = load(f"rq3_{m}_{kind}_{held}")["test"]["per_class_f1"]
                rows.append(pd.Series(f1, name=f"{m.upper()} {'trained with' if kind=='reference' else 'never saw'} {held}"))
        d = pd.DataFrame(rows).T
        d.plot.bar(ax=ax, width=0.8, legend=True)
        ax.set_title(f"F1 per family on {held} test rows")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(f"{F}/rq3_per_family.png")


def rq4_stats_plot():
    s = pd.read_csv(f"{R}/rq4_family_stats.csv", index_col=0)
    cols = [("median_words", "median words"), ("bold", "bold spans / resp"), ("headers", "headers / resp"),
            ("hr_rules", "--- rules / resp"), ("em_dash_per_100w", "em dashes / 100 words"),
            ("curly_apostrophe_share", "share of curly apostrophes"), ("emoji_per_100w", "emoji / 100 words"),
            ("mattr", "lexical diversity (MATTR)")]
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.6))
    for ax, (c, t) in zip(axes.flat, cols):
        ax.bar(s.index, s[c], color="0.45")
        ax.set_title(t, fontsize=8)
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        if c == "mattr":
            ax.set_ylim(0.7, 0.82)
    fig.tight_layout()
    fig.savefig(f"{F}/rq4_family_stats.png")


def rq4_ablation_plot():
    abl = ["full text", "strip_formatting", "words_only", "drop_first_line", "first_50_words", "shuffled_words"]
    fig, ax = plt.subplots(figsize=(7, 3))
    w = 0.38
    for k, m in enumerate(["cnn", "lstm"]):
        acc = [load(f"rq2_{m}_output" if a == "full text" else f"rq4_{m}_{a}")["test"]["accuracy"] for a in abl]
        b = ax.bar(np.arange(len(abl)) + (k - 0.5) * w, acc, w, label=m.upper())
        ax.bar_label(b, fmt="%.2f", fontsize=7)
    ax.axhline(1 / 8, ls="--", c="gray", lw=1)
    ax.set_xticks(range(len(abl)), [a.replace("_", " ") for a in abl], rotation=15)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{F}/rq4_ablation.png")


def summary():
    rows = []
    for f in sorted(glob.glob(f"{R}/*.json")):
        d = json.load(open(f))
        if "test" not in d:
            continue
        r = {"run": d.get("name", os.path.basename(f)), "test_acc": d["test"]["accuracy"], "test_macro_f1": d["test"]["macro_f1"],
             "epochs": len(d.get("history", [])), "n_train": d.get("n_train"), "n_test": d.get("n_test")}
        for k, v in d.get("extra", {}).items():
            r[f"{k}_acc"] = v["accuracy"]
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(f"{R}/summary.csv", index=False)
    print(t.round(4).to_string())


for fn in [confusion_plots, curves, rq2_bars, rq3_plot, rq4_stats_plot, rq4_ablation_plot, summary]:
    try:
        fn()
    except FileNotFoundError as e:
        print("skip", fn.__name__, "missing", e.filename)
