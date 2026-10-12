"""Train a CNN or BiLSTM to identify the LLM family that wrote a response.

Examples:
  # RQ1: output only
  python -m src.train --model cnn --input_mode output
  python -m src.train --model rnn --input_mode output
  # RQ2: input only / input+output
  python -m src.train --model cnn --input_mode input
  python -m src.train --model cnn --input_mode both
  # RQ3: train on coding+math+general, test on writing (and in-domain test)
  python -m src.train --model cnn --input_mode output --train_domains coding,math,general --test_domains writing
  # RQ4 ablations: hide self-identification / strip formatting / cap length
  python -m src.train --model cnn --input_mode output --ablation mask_identity,strip_format
  # quick smoke test
  python -m src.train --model rnn --input_mode output --subset 3000 --epochs 2 --run_name smoke_rnn

Outputs go to runs/<run_name>/: config.json, vocab.json, labels.json, model.pt (best val macro-F1),
history.json, metrics_<tag>.json, report_<tag>.txt, confusion_<tag>.png.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from .common import filter_domains, load_split, make_loader, set_seed, stratified_subset, tokens_for
from .data.text import Vocab
from .eval import evaluate_and_save, predict
from .models import build_model

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["cnn", "rnn"], required=True)
    ap.add_argument("--input_mode", choices=["input", "output", "both"], default="output")
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--runs_dir", default=str(ROOT / "runs"))
    # data / text
    ap.add_argument("--max_len", type=int, default=400, help="max tokens per example")
    ap.add_argument("--max_input_len", type=int, default=100, help="max prompt tokens in 'both' mode")
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--max_vocab", type=int, default=30000)
    ap.add_argument("--subset", type=int, default=0, help="use only N training examples (stratified)")
    ap.add_argument("--ablation", default="none",
                    help="comma list of: none, mask_identity, strip_format, truncateN (e.g. truncate64)")
    ap.add_argument("--train_domains", default="", help="RQ3: comma list of domains for train/val")
    ap.add_argument("--test_domains", default="", help="RQ3: comma list of domains for the held-out test")
    # model
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--num_filters", type=int, default=128)
    ap.add_argument("--kernel_sizes", default="3,4,5")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--unidirectional", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.5)
    # optimisation
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3, help="early stopping on val macro-F1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    cfg = vars(args).copy()
    cfg["kernel_sizes"] = [int(k) for k in args.kernel_sizes.split(",")]
    cfg["train_domains"] = [d for d in args.train_domains.split(",") if d]
    cfg["test_domains"] = [d for d in args.test_domains.split(",") if d]
    run_name = args.run_name or "_".join(
        [args.model, args.input_mode]
        + ([args.ablation.replace(",", "+")] if args.ablation != "none" else [])
        + ([f"train-{'+'.join(cfg['train_domains'])}"] if cfg["train_domains"] else []))
    out = Path(args.runs_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)

    train = filter_domains(load_split("train", args.data_dir), cfg["train_domains"])
    val = filter_domains(load_split("val", args.data_dir), cfg["train_domains"])
    test_all = load_split("test", args.data_dir)
    train = stratified_subset(train, args.subset, args.seed)
    if args.subset:
        val = stratified_subset(val, max(200, args.subset // 5), args.seed)
        test_all = stratified_subset(test_all, max(400, args.subset // 3), args.seed)
    labels = sorted(train["family"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    cfg["train_n"], cfg["val_n"] = len(train), len(val)

    t0 = time.time()
    vocab = Vocab.build(tokens_for(train, cfg), min_freq=args.min_freq, max_size=args.max_vocab)
    train_dl = make_loader(train, cfg, vocab, label2id, shuffle=True)
    val_dl = make_loader(val, cfg, vocab, label2id, shuffle=False)
    print(f"[train] {run_name}: train={len(train)} val={len(val)} labels={labels} vocab={len(vocab)} "
          f"(encode {time.time() - t0:.1f}s)")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    cfg["device"] = str(device)
    model = build_model(cfg, len(vocab), len(labels), vocab.pad_idx).to(device)
    cfg["n_params"] = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    vocab.save(out / "vocab.json")
    (out / "labels.json").write_text(json.dumps(labels))
    best_f1, best_epoch, bad, history = -1.0, -1, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for ids, lengths, y in train_dl:
            ids, lengths, y = ids.to(device), lengths.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(ids, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        yv, pv = predict(model, val_dl)
        val_acc = float((yv == pv).mean())
        val_f1 = float(f1_score(yv, pv, average="macro"))
        rec = {"epoch": epoch, "train_loss": tot / n, "val_acc": val_acc, "val_macro_f1": val_f1,
               "sec": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"[train] epoch {epoch}: loss={rec['train_loss']:.4f} val_acc={val_acc:.4f} "
              f"val_f1={val_f1:.4f} ({rec['sec']}s)")
        if val_f1 > best_f1:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            torch.save(model.state_dict(), out / "model.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("[train] early stopping")
                break
    cfg.update({"epochs_run": len(history), "best_epoch": best_epoch, "best_val_macro_f1": best_f1})
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))

    # ---- final evaluation with the best checkpoint
    model.load_state_dict(torch.load(out / "model.pt", map_location=device))
    evals = {}
    if cfg["test_domains"]:  # RQ3: out-of-domain + in-domain test
        evals["test_ood"] = filter_domains(test_all, cfg["test_domains"])
        if cfg["train_domains"]:
            evals["test_indomain"] = filter_domains(test_all, cfg["train_domains"])
    else:
        evals["test"] = filter_domains(test_all, cfg["train_domains"])
    results = {}
    for tag, df in evals.items():
        df = df[df["family"].isin(labels)]
        dl = make_loader(df, cfg, vocab, label2id, shuffle=False)
        m = evaluate_and_save(model, dl, labels, out, tag=tag,
                              title=f"{args.model.upper()} / {args.input_mode} / {tag} (n={len(df)})")
        results[tag] = {"n": m["n"], "accuracy": round(m["accuracy"], 4), "macro_f1": round(m["macro_f1"], 4)}
    print(f"[train] best epoch {best_epoch} (val macro-F1 {best_f1:.4f}); results: {json.dumps(results)}")


if __name__ == "__main__":
    main()
