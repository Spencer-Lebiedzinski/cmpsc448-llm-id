#!/usr/bin/env bash
# Full experiment grid for the report (CPU: CNN runs take minutes; RNN runs ~30-60 min each
# at max_len 400 -- use --max_len 256 or a GPU/Colab to speed up). Results -> runs/full, results/summary.md
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
R=runs/full; EP=${EP:-10}
for M in cnn rnn; do
  # RQ1 (output) + RQ2 (input / both)
  for MODE in output input both; do
    $PY -m src.train --model $M --input_mode $MODE --epochs $EP --runs_dir $R --threads 4
  done
  # RQ3: leave-one-domain-out (train on 3 domains, test on the 4th)
  for HOLD in writing coding math general; do
    TR=$(echo coding math writing general | tr ' ' '\n' | grep -v "^$HOLD$" | paste -sd, -)
    $PY -m src.train --model $M --input_mode output --epochs $EP --runs_dir $R --threads 4 \
        --train_domains $TR --test_domains $HOLD --run_name ${M}_output_holdout-$HOLD
  done
  # RQ4 ablations
  for AB in mask_identity strip_format truncate64 mask_identity,strip_format; do
    $PY -m src.train --model $M --input_mode output --epochs $EP --runs_dir $R --threads 4 --ablation $AB
  done
done
for MODE in output input both; do $PY -m src.baselines.tfidf_lr --input_mode $MODE; done
$PY -m src.analysis.stylometry --out results/rq4
$PY -m src.eval --summarize $R --summary_out results/summary.md
