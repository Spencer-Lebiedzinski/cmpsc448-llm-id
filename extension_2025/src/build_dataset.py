"""Build the (LLM_name, LLM_input, LLM_output) dataset from LMArena arena-human-preference-140k.

Each arena battle has one prompt and two responses (model_a, model_b), so one battle
gives two rows. We keep english, single turn, text only conversations and map
model names to families.
"""
import glob
import hashlib
import re

import numpy as np
import pandas as pd

RAW = "data/raw/data/*.parquet"
OUT = "data/llm_dataset.parquet"
PER_FAMILY = 4000
SEED = 448

# order matters, first match wins
FAMILY_RULES = [
    ("Claude", r"^claude"),
    ("Gemini", r"^gemini"),
    ("GPT", r"^(gpt|chatgpt|o1|o3|o4)"),
    ("Qwen", r"^(qwen|qwq)"),
    ("DeepSeek", r"^deepseek"),
    ("Llama", r"^llama"),
    ("Grok", r"^grok"),
    ("Mistral", r"^(mistral|magistral)"),
]


def family_of(model):
    for fam, pat in FAMILY_RULES:
        if re.match(pat, model.lower()):
            return fam
    return None


def text_of(turn):
    return "\n".join(c["text"] for c in turn["content"] if c["type"] == "text" and c["text"])


def has_image(turn):
    return any(c["type"] != "text" or c.get("image") for c in turn["content"])


def domain_of(row):
    tags = row["category_tag"]
    if row["is_code"]:
        return "code"
    if tags["math_v0.1"]["math"]:
        return "math"
    if tags["creative_writing_v0.1"]["creative_writing"]:
        return "creative_writing"
    return "general"


def battle_rows(row):
    out = []
    for side in ("a", "b"):
        conv = row[f"conversation_{side}"]
        if len(conv) != 2 or conv[0]["role"] != "user" or conv[1]["role"] != "assistant":
            return []
        if has_image(conv[0]):
            return []
        prompt, resp = text_of(conv[0]).strip(), text_of(conv[1]).strip()
        model = row[f"model_{side}"]
        fam = family_of(model)
        if fam is None or len(prompt) < 5 or len(resp) < 20:
            continue
        out.append({
            "battle_id": row["id"],
            "LLM_name": fam,
            "model": model,
            "LLM_input": prompt,
            "LLM_output": resp,
            "domain": domain_of(row),
            "timestamp": row["timestamp"],
            "prompt_id": hashlib.md5(prompt.lower().encode()).hexdigest()[:12],
        })
    return out


def main():
    rows = []
    for f in sorted(glob.glob(RAW)):
        df = pd.read_parquet(f)
        df = df[(df.language == "en") & (df.evaluation_order == 1)]
        for _, r in df.iterrows():
            rows.extend(battle_rows(r))
        print(f, len(rows))
    full = pd.DataFrame(rows)
    print(full.LLM_name.value_counts())
    print(pd.crosstab(full.LLM_name, full.domain))

    # balanced sample, same number per family
    rng = np.random.default_rng(SEED)
    keep = []
    for fam, g in full.groupby("LLM_name"):
        if len(g) < PER_FAMILY:
            print("dropping family (too few rows):", fam, len(g))
            continue
        keep.append(g.iloc[rng.permutation(len(g))[:PER_FAMILY]])
    data = pd.concat(keep).reset_index(drop=True)
    data.to_parquet(OUT, compression="zstd", compression_level=19)
    print("saved", OUT, data.shape)
    print(data.LLM_name.value_counts())
    print(pd.crosstab(data.LLM_name, data.domain))


if __name__ == "__main__":
    main()
