"""RQ4: what distinguishes the LLMs?

1. descriptive stats per family (length, lexical diversity, formatting)
2. classifier on hand made style features only
3. distinctive words per family (log odds ratio with a prior)
4. ablations: retrain the CNN/LSTM after removing certain signals from the output text
"""
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from features import drop_first_line, feature_frame, first_50_words, make_shuffle, strip_formatting, words_only
from text_utils import LABELS, labels_of, load_data, make_tokens, max_len_for, split_data, tokenize
from train import metrics, run

part = sys.argv[1] if len(sys.argv) > 1 else "all"
models = sys.argv[2].split(",") if len(sys.argv) > 2 else ["cnn"]
df = load_data()
tr, va, te = split_data(df)
y_tr, y_va, y_te = labels_of(tr), labels_of(va), labels_of(te)

if part in ("all", "stats"):
    feats = feature_frame(df.LLM_output)
    feats["LLM_name"] = df.LLM_name.values
    stats = feats.groupby("LLM_name").mean().round(3)
    stats["median_words"] = feats.groupby("LLM_name").n_words.median()
    stats.to_csv("results/rq4_family_stats.csv")
    print(stats.T.to_string())

    # style feature classifier
    f_tr, f_te = feature_frame(tr.LLM_output), feature_frame(te.LLM_output)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=448).fit(f_tr, y_tr)
    m = metrics(y_te, clf.predict(f_te))
    imp = permutation_importance(clf, f_te, y_te, n_repeats=5, random_state=448, scoring="accuracy")
    m["permutation_importance"] = dict(sorted(zip(f_te.columns, imp.importances_mean.round(4).tolist()), key=lambda x: -x[1]))
    # length only
    clf_len = HistGradientBoostingClassifier(random_state=448).fit(f_tr[["log_words"]], y_tr)
    m["length_only_accuracy"] = float((clf_len.predict(f_te[["log_words"]]) == y_te).mean())
    json.dump(m, open("results/rq4_style_features.json", "w"), indent=1)
    print("style features acc", m["accuracy"], "f1", m["macro_f1"], "length only", m["length_only_accuracy"])
    print(m["permutation_importance"])

    # distinctive words: share of a family's responses that contain the word vs share for the other families.
    # only words used in at least 300 responses so we dont just pick up topic words from a few long answers
    doc_sets = [set(t for t in tokenize(x) if t.isalpha()) for x in df.LLM_output]
    df_all = Counter(w for d in doc_sets for w in d)
    vocab_ok = {w for w, c in df_all.items() if c >= 300}
    top = {}
    for fam in LABELS:
        mask = (df.LLM_name == fam).values
        fam_c = Counter(w for d, m in zip(doc_sets, mask) if m for w in d if w in vocab_ok)
        n_f, n_o = mask.sum(), (~mask).sum()
        lift = {w: ((fam_c[w] + 1) / n_f) / ((df_all[w] - fam_c[w] + 1) / n_o) for w in vocab_ok}
        openers = Counter(" ".join(x.split()[:2]) for x in df.LLM_output[mask])
        top[fam] = {
            "words": [(w, round(l, 2), round(fam_c[w] / n_f, 3)) for w, l in sorted(lift.items(), key=lambda x: -x[1])[:15]],
            "top_openers": openers.most_common(6),
        }
        print(fam, [w for w, _, _ in top[fam]["words"][:12]], top[fam]["top_openers"][:4])
    json.dump(top, open("results/rq4_distinctive_words.json", "w"), indent=1)

if part in ("all", "ablate"):
    ablations = {
        "strip_formatting": strip_formatting,
        "words_only": words_only,
        "first_50_words": first_50_words,
        "drop_first_line": drop_first_line,
        "shuffled_words": make_shuffle(0),
    }
    for name, fn in ablations.items():
        tr_tok, va_tok, te_tok = (make_tokens(d, "output", transform=fn) for d in (tr, va, te))
        for model in models:
            run(f"rq4_{model}_{name}", model, tr_tok, y_tr, va_tok, y_va, te_tok, y_te, max_len_for("output"))
