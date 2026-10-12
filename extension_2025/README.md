# Extension: the same study on 2025 models (8 families)

Section 4.4 of `report/report.md` uses the results in this folder. The main project uses Arena-55k (models from 2023 to
early 2024). This extension repeats RQ1 to RQ4 on current models from LMArena's
[`arena-human-preference-140k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) release
(CC-BY-4.0 prompts, battles from April to July 2025).

- 8 families, 4,000 responses each (32,000 total, chance = 12.5%): Claude, DeepSeek, GPT, Gemini, Grok, Llama, Mistral, Qwen
  (44 model versions, for example claude-opus-4, o3, gpt-4.1, gemini-2.5-pro, llama-4-maverick, qwen3-235b, deepseek-r1, grok-3).
- English, first turn, text only. 70/15/15 split grouped by prompt (md5 of the lowercased prompt).
- Domains come from LMArena's own category tags (`is_code`, `math_v0.1`, `creative_writing_v0.1`), with everything else counted as general.
- Same model designs as the main project: Kim-style TextCNN (widths 3/4/5, 128 filters) and a 1-layer BiLSTM (128 per direction, mean + max pooling).
  Adam 1e-3, batch 64, max 12 epochs, early stopping on val macro-F1. Trained on Apple MPS.

## Run (from this folder)
```
pip install -r requirements.txt
# download the 7 parquet shards of lmarena-ai/arena-human-preference-140k into data/raw/data/, then
python src/build_dataset.py
python src/run_rq1_rq2.py tfidf && python src/run_rq1_rq2.py cnn,lstm
python src/run_rq3.py cnn,lstm
python src/run_rq4.py stats && python src/run_rq4.py ablate cnn,lstm
python src/run_seeds.py && python src/run_lstm_long.py
python src/make_figures.py
```
Results are in `results/*.json`, `results/summary.csv` is the table of all runs, and the plots are in `figures/`.
The processed dataset is not committed (39 MB). `build_dataset.py` rebuilds it deterministically (seed 448).
