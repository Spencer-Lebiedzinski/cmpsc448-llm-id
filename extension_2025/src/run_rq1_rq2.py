"""RQ1 (output only) and RQ2 (input only vs output only vs input + output)."""
import json
import sys

from baseline import tfidf_lr
from text_utils import labels_of, load_data, make_tokens, max_len_for, split_data
from train import metrics, run

models = sys.argv[1].split(",") if len(sys.argv) > 1 else ["cnn", "lstm"]
df = load_data()
tr, va, te = split_data(df)
print("split sizes", len(tr), len(va), len(te))
y_tr, y_va, y_te = labels_of(tr), labels_of(va), labels_of(te)

for mode in ["output", "input", "both"]:
    tr_tok, va_tok, te_tok = (make_tokens(d, mode) for d in (tr, va, te))
    if "tfidf" in models:
        _, pred = tfidf_lr(tr_tok, y_tr, te_tok)
        m = metrics(y_te, pred)
        json.dump({"name": f"rq2_tfidf_{mode}", "test": m}, open(f"results/rq2_tfidf_{mode}.json", "w"), indent=1)
        print(f"tfidf {mode} acc {m['accuracy']:.4f} f1 {m['macro_f1']:.4f}", flush=True)
    for model in [m for m in models if m != "tfidf"]:
        run(f"rq2_{model}_{mode}", model, tr_tok, y_tr, va_tok, y_va, te_tok, y_te, max_len_for(mode))
