# Who Wrote It? Identifying LLMs from Their Responses
CMPSC 448 Midterm Project (due **Sun Oct 11, 2026, 11:59pm ET**)

**Author:** Spencer Lebiedzinski (individual project)

We classify which **LLM family** (GPT, Claude, Google, Llama, Mistral) wrote a chatbot response,
using a **Text CNN** and a **BiLSTM (RNN)** implemented in PyTorch. The two models share one tokenizer and vocabulary.

| RQ | Question | How to run |
|---|---|---|
| RQ1 | Identify the LLM from its output | `--input_mode output` for `cnn` and `rnn` |
| RQ2 | Input only vs output only vs input + output | `--input_mode input / output / both` |
| RQ3 (+5) | Cross-domain generalization | `--train_domains coding,math,general --test_domains writing` |
| RQ4 (+5) | What distinguishes LLMs? | `src/analysis/stylometry.py`, plus `--ablation mask_identity / strip_format / truncate64` |

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU is fine
pip install -r requirements.txt
```

## 1. Build the dataset
```bash
python -m src.data.prepare_dataset            # downloads Arena-55k (public), writes data/processed/*.csv
```
Defaults: 5 families x 4,000 examples, 80/10/10 group split by prompt. See `data/README.md`
(provenance and licenses) and `data/STATS.md` (counts). The TA-suggested `lmsys/chatbot_arena_conversations`
is **gated** on Hugging Face. To add it, set `HF_TOKEN` and pass `--source arena33k` or `--source both`.

## 2. Train and evaluate
```bash
python -m src.train --model cnn --input_mode output            # RQ1 CNN
python -m src.train --model rnn --input_mode output --threads 4 # RQ1 BiLSTM
python -m src.train --model cnn --input_mode input              # RQ2
python -m src.train --model cnn --input_mode both               # RQ2
python -m src.train --model cnn --input_mode output --train_domains coding,math,general --test_domains writing  # RQ3
python -m src.train --model cnn --input_mode output --ablation mask_identity,strip_format                     # RQ4
python -m src.eval  --run_dir runs/cnn_output --domains coding  # re-evaluate a run on a subset
python -m src.eval  --summarize runs                            # results/summary.md table of every run
```
Each run writes to `runs/<name>/`: `config.json`, `history.json` (per-epoch loss and val metrics),
`metrics_<tag>.json` (accuracy, macro-F1, per-class P/R/F1, confusion matrix), `report_<tag>.txt`,
`confusion_<tag>.png`, and `model.pt` (best val macro-F1; git-ignored).

Main flags: `--max_len 400`, `--max_input_len 100` (prompt budget in `both` mode), `--epochs 10`,
`--patience 3`, `--batch_size 64`, `--lr 1e-3`, `--emb_dim 128`, `--num_filters 128`,
`--kernel_sizes 3,4,5`, `--hidden_dim 128`, `--num_layers 1`, `--dropout 0.5`, `--subset N`, `--seed 42`.

Extras: `python -m src.baselines.tfidf_lr --input_mode output` (TF-IDF + logistic regression reference) and
`python -m src.analysis.stylometry` (RQ4 features, boxplots, and a features-only classifier, written to `results/rq4/`).

## 3. Scripts
- `scripts/smoke_test.sh`: the end-to-end check on a subset (about 10 min on CPU). Output goes to `results/smoke_summary.md`.
- `scripts/run_experiments.sh`: the full grid for the report (RQ1 to RQ4, both models). CNN runs take minutes.
  BiLSTM runs take roughly 30 to 60 min each on CPU at `max_len 400`. Use `--max_len 256` or a Colab GPU to speed them up.

## Layout
```
src/data/       prepare_dataset.py (download, clean, balance, split), families.py, domains.py, text.py (tokenizer, vocab, ablations)
src/models/     cnn.py (TextCNN), rnn.py (BiLSTMClassifier)
src/train.py    training loop with early stopping on val macro-F1
src/eval.py     metrics, confusion matrices, run summaries
src/analysis/   stylometry.py (RQ4)
src/baselines/  tfidf_lr.py
report/         outline.md (rubric-aligned report outline)
results/        committed metrics and figures
```

## Use of LLM assistance
The course allows LLM help. Note in the report which parts were LLM-assisted (for example, scaffolding code) and confirm
that the team verified and understands everything submitted.
