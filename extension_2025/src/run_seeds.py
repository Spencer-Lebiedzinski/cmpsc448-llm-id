"""Repeat the RQ1 output only runs with 2 more seeds (different init + batch order, same split) for error bars."""
from text_utils import labels_of, load_data, make_tokens, max_len_for, split_data
from train import run

df = load_data()
tr, va, te = split_data(df)
toks = [make_tokens(d, "output") for d in (tr, va, te)]
for seed in (1, 2):
    for model in ("cnn", "lstm"):
        run(f"rq1_{model}_output_seed{seed}", model, toks[0], labels_of(tr), toks[1], labels_of(va), toks[2], labels_of(te),
            max_len_for("output"), seed=seed)
