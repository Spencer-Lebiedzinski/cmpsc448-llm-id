"""TF-IDF + logistic regression baseline (not required, just to compare the neural models against)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from text_utils import NL


def tfidf_lr(tr_tok, y_tr, te_tok):
    join = lambda toks: [" ".join(t) for t in toks]
    clf = make_pipeline(
        TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True),
        LogisticRegression(C=4.0, max_iter=2000),
    )
    clf.fit(join(tr_tok), y_tr)
    return clf, clf.predict(join(te_tok))
