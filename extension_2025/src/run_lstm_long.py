"""The output only BiLSTM was still improving a bit at epoch 12, so train it longer once."""
from text_utils import labels_of, load_data, make_tokens, max_len_for, split_data
from train import run

df = load_data()
tr, va, te = split_data(df)
toks = [make_tokens(d, "output") for d in (tr, va, te)]
run("rq1_lstm_output_25ep", "lstm", toks[0], labels_of(tr), toks[1], labels_of(va), toks[2], labels_of(te),
    max_len_for("output"), epochs=25, patience=4)
