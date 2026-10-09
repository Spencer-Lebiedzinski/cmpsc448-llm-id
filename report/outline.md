# Who Wrote It? Identifying LLMs from Their Responses
CMPSC 448 Midterm Project Report (Fall 2026)

**Team leader:** Spencer Lebiedzinski. **Team members:** _list everyone, or "individual project"_
**Repository:** _GitHub URL_

> The PDF is 80% of the grade. Five rubric sections, 20 points each. Export to PDF, for example with
> `pandoc report/outline.md -o report/report.pdf`, or by porting this outline to Overleaf/LaTeX.
> Every number below should come from `results/summary.md` or `runs/full/*/metrics_*.json`.

---

## 1. Problem Definition and Dataset Curation (20 pts)
### 1.1 Problem
- The problem is multi-class text classification: given an LLM response (and optionally its prompt), predict the LLM family.
- Motivation: AI-text attribution, provenance, accountability.
- State RQ1 to RQ4 explicitly. RQ1 and RQ2 are required; RQ3 and RQ4 are attempted for extra credit.

### 1.2 Data sources and collection
- Arena-55k (`lmarena-ai/arena-human-preference-55k`, Apache-2.0): real user prompts from Chatbot Arena, each answered by 2 models.
- Explain why it was chosen: real prompts, many families, paired responses to the *same* prompt (which controls for topic), public.
- Note the TA-suggested `lmsys/chatbot_arena_conversations` (gated; say whether it was used).
- Ethics: public research data with no private or confidential info. Processed data is not redistributed.

### 1.3 Curation pipeline (cite `data/README.md`)
- First turn only. Model-to-family mapping, where only first-party models are kept (table: family, then models included).
- Filters: empty or short responses, non-English prompts, de-duplication.
- Balancing: undersample to N per family. **Table: class counts per split** (copy from `data/STATS.md`).
- Group split by prompt, so no prompt appears in two splits. Explain why this matters for RQ2.
- Heuristic domain tags (coding, math, writing, general). Give the family x domain table and the tag precision from a manual check of about 50 prompts per tag.
- Descriptive stats: response length distribution per family (figure from `results/rq4/feature_boxplots.png`).

### 1.4 Limitations of the data
- Era bias (2023 models), model versions pooled within a family, Arena sampling windows differ by family (topic leakage), self-identification.

## 2. CNN / RNN Implementation and Training (20 pts)
### 2.1 Shared preprocessing
- Regex word tokenizer that keeps formatting tokens (`<nl>`, `**`, `#`, backticks), lowercasing, vocab built on train only (min_freq 2, max 30k).
- Max length 400 tokens. In `both` mode the sequence is the prompt (≤100 tokens), then `<sep>`, then the response.

### 2.2 Text CNN
- Embedding 128, then Conv1d with widths {3,4,5} and 128 filters each, ReLU, masked max-over-time pooling, dropout 0.5, linear layer to 5 classes.
- Parameter count (from `config.json` `n_params`). Include a diagram.

### 2.3 BiLSTM (RNN)
- Embedding 128, then 1-layer bidirectional LSTM (128 hidden per direction) over packed sequences. The final features are the concatenation of masked max-pooled outputs and the final forward and backward hidden states, followed by dropout and a linear layer.
- Parameter count. Include a diagram.

### 2.4 Training details
- Cross-entropy loss, AdamW (lr 1e-3, wd 1e-4), batch size 64, gradient clipping at 1.0, up to 10 epochs, early stopping on val macro-F1 (patience 3), best checkpoint kept, seed 42.
- Hardware and wall-clock time per epoch (CPU). Training curves (from `history.json`).
- Hyperparameter search, if any: what was tried and what was chosen.

## 3. Results and Presentation (20 pts)
### 3.1 RQ1: identifying the LLM from its output
- **Table:** CNN vs BiLSTM on test (accuracy, macro-F1, chance = 20%), plus the TF-IDF+LR reference.
- **Figure:** confusion matrices for both models (`confusion_test.png`). Which families get confused (for example, GPT vs Claude)?
- Per-class F1.

### 3.2 RQ2: does the prompt help?
- **Table:** {CNN, BiLSTM} x {input, output, both}.
- Discuss (i) whether input-only beats chance, (ii) whether adding the input improves on output-only, and (iii) why input-only might work: Arena sampling periods differ by family, so prompt topics shift over time. Note that both models in a battle share a prompt, which caps input-only accuracy.

### 3.3 RQ3: cross-domain generalization (+5)
- Leave-one-domain-out: in-domain vs out-of-domain accuracy for each held-out domain and each model.
- Answer: (i) does performance drop on unseen tasks, (ii) which families stay identifiable, (iii) is the model learning a general fingerprint or task-specific patterns?

## 4. In-depth Analyses (20 pts) (RQ4, +5)
- Stylometric features per family (length, windowed TTR, word length, bullets, bold, headers, code blocks, refusals, self-mentions). Include a table and boxplots.
- Features-only logistic regression accuracy vs CNN/RNN. How much does style explain?
- **Ablations** (see `train.py --ablation`): mask_identity, strip_format, truncate64, and combinations. Report the accuracy drop relative to the full output model.
- Signals the classifier uses: top n-grams per family, CNN filter activations, and misclassified examples.
- Optional: per-model (fine-grained) confusion within families, effect of training-set size (learning curve), effect of max_len.

## 5. Lessons and Experience Learned (20 pts)
- Data pitfalls: leakage through shared prompts and splits, era and topic bias, label noise from the family mapping.
- Modeling: CNN vs RNN trade-offs (speed, accuracy, sensitivity to length), what did and didn't help.
- Process: team coordination, compute limits on CPU, reproducibility (seeds, scripts).
- Responsible use of LLM assistance (the course allows it): what was assisted and how it was verified.
- What we would do next: more recent models, larger data, transformer baselines, calibrated open-set detection.

## References
- Chiang et al. 2024, Chatbot Arena (arXiv:2403.04132). Kim 2014, CNNs for Sentence Classification. Hochreiter & Schmidhuber 1997, LSTM.
- Dataset cards (URLs above).

## Appendix
- Full hyperparameters and commands (`scripts/run_experiments.sh`), and extra confusion matrices.
