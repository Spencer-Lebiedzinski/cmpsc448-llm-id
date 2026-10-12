"""Training loop used by all experiments."""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from models import build_model
from text_utils import LABELS, build_vocab, encode

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def pad_batch(seqs):
    lengths = torch.tensor([len(s) for s in seqs])
    x = torch.zeros(len(seqs), int(lengths.max()), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
    return x, lengths


def batches(ids, y, bs, shuffle, rng):
    """Length bucketing: shuffle, sort inside chunks of 50 batches by length, then shuffle the batches."""
    if shuffle:
        order = rng.permutation(len(ids))
        chunk = bs * 50
        groups = []
        for c in range(0, len(order), chunk):
            part = sorted(order[c : c + chunk], key=lambda j: len(ids[j]))
            groups += [part[i : i + bs] for i in range(0, len(part), bs)]
        groups = [groups[k] for k in rng.permutation(len(groups))]
    else:
        order = np.arange(len(ids))
        groups = [order[i : i + bs] for i in range(0, len(order), bs)]
    for idx in groups:
        idx = np.array(idx)
        x, lengths = pad_batch([ids[j] for j in idx])
        yield x, lengths, torch.tensor(y[idx])


@torch.no_grad()
def predict(model, ids, device, bs=256):
    model.eval()
    preds = []
    for x, lengths, _ in batches(ids, np.zeros(len(ids), dtype=int), bs, False, None):
        preds.append(model(x.to(device), lengths).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_f1": dict(zip(LABELS, f1_score(y_true, y_pred, average=None, labels=range(len(LABELS))).round(4).tolist())),
        "confusion": confusion_matrix(y_true, y_pred, labels=range(len(LABELS))).tolist(),
    }


def run(name, model_name, tr_tok, y_tr, va_tok, y_va, te_tok, y_te, max_len,
        epochs=12, patience=3, lr=1e-3, bs=64, seed=448, out_dir="results", device=None, extra_tests=None):
    """Train with Adam + early stopping on val macro F1, evaluate the best checkpoint on test."""
    device = device or DEVICE
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vocab = build_vocab(tr_tok)
    tr = [encode(t, vocab, max_len) for t in tr_tok]
    va = [encode(t, vocab, max_len) for t in va_tok]
    te = [encode(t, vocab, max_len) for t in te_tok]

    model = build_model(model_name, len(vocab), len(LABELS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, bad, history = -1, None, 0, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, lengths, yb in batches(tr, y_tr, bs, True, rng):
            opt.zero_grad()
            loss = loss_fn(model(x.to(device), lengths), yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(yb)
        va_f1 = f1_score(y_va, predict(model, va, device), average="macro")
        history.append({"epoch": ep, "train_loss": total / len(tr), "val_macro_f1": float(va_f1)})
        print(f"[{name}] epoch {ep} loss {total/len(tr):.4f} val_f1 {va_f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if va_f1 > best:
            best, bad = va_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    y_pred = predict(model, te, device)
    res = {"name": name, "model": model_name, "vocab_size": len(vocab), "max_len": max_len,
           "n_train": len(tr), "n_val": len(va), "n_test": len(te), "best_val_macro_f1": float(best),
           "history": history, "train_seconds": time.time() - t0, "test": metrics(y_te, y_pred)}
    # optional extra test sets, {name: (tokens, labels)}
    for tname, (toks, ys) in (extra_tests or {}).items():
        res.setdefault("extra", {})[tname] = metrics(ys, predict(model, [encode(t, vocab, max_len) for t in toks], device))
        print(f"[{name}] {tname} acc {res['extra'][tname]['accuracy']:.4f}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{name}.json", "w") as f:
        json.dump(res, f, indent=1)
    np.save(f"{out_dir}/{name}_pred.npy", y_pred)
    print(f"[{name}] TEST acc {res['test']['accuracy']:.4f} macroF1 {res['test']['macro_f1']:.4f}", flush=True)
    return res, y_pred
