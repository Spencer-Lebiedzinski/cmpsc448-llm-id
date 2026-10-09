# Results so far (scaffold verification, 2026-10-05)

## Smoke test (`scripts/smoke_test.sh`): 4,000-example stratified train subset, 3 epochs, CPU
Test set: stratified subset of 1,333 held-out examples. Chance is 20%. Source: `results/smoke_summary.md`.

| run | model | input | accuracy | macro-F1 | notes |
|---|---|---|---|---|---|
| cnn_output | CNN | output | 0.458 | 0.407 | about 10 s/epoch |
| rnn_output | BiLSTM | output | 0.452 | 0.451 | about 95 to 135 s/epoch (4 threads); best at epoch 2 |
| cnn_input | CNN | input | 0.192 | 0.182 | at chance, as expected with the prompt-group split |
| cnn_both | CNN | both | 0.461 | 0.437 | |
| cnn_output_rq3 | CNN | output, train coding+math+general | in-domain 0.449 / OOD writing 0.373 | 0.398 / 0.318 | OOD n=201 only |

RQ4 features-only logistic regression (16 stylometric features, full train to full test): acc 0.409, macro-F1 0.401 (`results/rq4_smoke/`).

## First full-data runs (16,057 train / 1,961 test)
| run | accuracy | macro-F1 | notes |
|---|---|---|---|
| CNN output, 8 epochs (`runs/full/cnn_output`) | 0.623 | 0.596 | val F1 still rising at epoch 8, so train longer |
| TF-IDF word+char n-grams + LogReg (`results/baselines/`) | 0.627 | 0.624 | non-neural reference |

These numbers come from short, untuned runs that only verify the pipeline. They are not final results.
