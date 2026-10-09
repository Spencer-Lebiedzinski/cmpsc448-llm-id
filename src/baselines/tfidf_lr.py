"""Optional non-neural baseline: TF-IDF (word 1-2grams + char 3-5grams) + logistic regression.
Useful as a sanity check / reference point next to the CNN and RNN in the results table.

  python -m src.baselines.tfidf_lr --input_mode output
"""
import argparse
import json
from pathlib import Path

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from ..common import load_split

ROOT = Path(__file__).resolve().parents[2]


def text_for(df, mode):
    if mode == "input":
        return df["input"].tolist()
    if mode == "output":
        return df["output"].tolist()
    return (df["input"] + " [SEP] " + df["output"]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_mode", choices=["input", "output", "both"], default="output")
    ap.add_argument("--out", default=str(ROOT / "results" / "baselines"))
    args = ap.parse_args()
    tr, te = load_split("train"), load_split("test")
    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=100000, sublinear_tf=True)
    Xtr = hstack([wv.fit_transform(text_for(tr, args.input_mode)), cv.fit_transform(text_for(tr, args.input_mode))])
    Xte = hstack([wv.transform(text_for(te, args.input_mode)), cv.transform(text_for(te, args.input_mode))])
    clf = LogisticRegression(max_iter=2000, C=4.0).fit(Xtr, tr["family"])
    p = clf.predict(Xte)
    res = {"input_mode": args.input_mode, "n_test": len(te), "accuracy": accuracy_score(te["family"], p),
           "macro_f1": f1_score(te["family"], p, average="macro")}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"tfidf_lr_{args.input_mode}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
