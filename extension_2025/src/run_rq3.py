"""RQ3: do fingerprints transfer across task domains?

For a held out domain X we train on every other domain and test on domain X.
As a reference we train on a random subset of the full training set with the SAME size
(so it does see domain X) and test on the same domain X test rows.
"""
import sys

import numpy as np

from text_utils import labels_of, load_data, make_tokens, max_len_for, split_data
from train import run

models = sys.argv[1].split(",") if len(sys.argv) > 1 else ["cnn", "lstm"]
df = load_data()
tr, va, te = split_data(df)
rng = np.random.default_rng(448)

for held in ["creative_writing", "code"]:
    tr_x, va_x = tr[tr.domain != held], va[va.domain != held]
    te_in, te_out = te[te.domain != held], te[te.domain == held]
    tr_ref = tr.iloc[rng.permutation(len(tr))[: len(tr_x)]]  # same size, mixed domains
    tok = lambda d: make_tokens(d, "output")
    extra = {"seen_domains": (tok(te_in), labels_of(te_in))}
    for model in models:
        run(f"rq3_{model}_holdout_{held}", model, tok(tr_x), labels_of(tr_x), tok(va_x), labels_of(va_x),
            tok(te_out), labels_of(te_out), max_len_for("output"), extra_tests=extra)
        run(f"rq3_{model}_reference_{held}", model, tok(tr_ref), labels_of(tr_ref), tok(va), labels_of(va),
            tok(te_out), labels_of(te_out), max_len_for("output"), extra_tests=extra)
