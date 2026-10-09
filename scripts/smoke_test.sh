#!/usr/bin/env bash
# End-to-end smoke test: data pipeline -> CNN + RNN (subset, few epochs) -> eval -> RQ4 features.
# Takes ~10 min on an 8-core CPU. Results: runs/smoke/*, results/smoke_summary.md
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SUB=${SUB:-4000}; EP=${EP:-3}; R=runs/smoke
[ -f data/processed/train.csv ] || $PY -m src.data.prepare_dataset
$PY -m src.train --model cnn --input_mode output --subset $SUB --epochs $EP --runs_dir $R --run_name cnn_output
$PY -m src.train --model rnn --input_mode output --subset $SUB --epochs $EP --runs_dir $R --run_name rnn_output --threads 4
$PY -m src.train --model cnn --input_mode input  --subset $SUB --epochs $EP --runs_dir $R --run_name cnn_input
$PY -m src.train --model cnn --input_mode both   --subset $SUB --epochs $EP --runs_dir $R --run_name cnn_both
$PY -m src.train --model cnn --input_mode output --subset $SUB --epochs $EP --runs_dir $R --run_name cnn_output_rq3 \
    --train_domains coding,math,general --test_domains writing
$PY -m src.analysis.stylometry --out results/rq4_smoke
$PY -m src.eval --summarize $R --summary_out results/smoke_summary.md
