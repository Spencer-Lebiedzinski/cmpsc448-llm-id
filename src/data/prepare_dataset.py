"""Build the (LLM_name, LLM_Input, LLM_output) dataset.

Sources (see data/README.md for provenance + licenses):
  * arena55k (default, public, Apache-2.0):
        https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k
        Chatbot Arena battles (Kaggle "LMSYS - Chatbot Arena" competition data).
  * arena33k (TA-suggested, GATED on Hugging Face -> needs an HF account that has
        accepted the dataset terms, plus HF_TOKEN in the environment):
        https://huggingface.co/datasets/lmsys/chatbot_arena_conversations

Each Arena battle = one user prompt answered by two models (A and B), so each
battle yields two examples sharing the same prompt. We keep only the FIRST turn
(first user message + first model reply).

Pipeline: download -> parse first turn -> map model -> family -> filter
(non-empty, min length, mostly-ASCII/English prompt) -> dedupe -> tag domain ->
balance classes (undersample to a common size) -> GROUP split by prompt
(no prompt appears in more than one split, so input-only models can't memorise
prompts across splits) -> write CSVs + stats.

Usage:
  python -m src.data.prepare_dataset                       # defaults (arena55k, 5 families)
  python -m src.data.prepare_dataset --source arena33k     # needs HF_TOKEN + accepted terms
  python -m src.data.prepare_dataset --families GPT,Claude,Google,Llama,Mistral,Qwen --per_class 2000
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .domains import tag_domain
from .families import DEFAULT_FAMILIES, model_to_family

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

ARENA55K_URL = "https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k/resolve/main/train.csv"
ARENA33K_REPO = "lmsys/chatbot_arena_conversations"


# --------------------------------------------------------------------------- download
def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[data] using cached {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    return dest


def _first(x):
    """Arena55k stores turns as JSON-encoded lists of strings."""
    try:
        lst = json.loads(x)
    except Exception:
        return None
    if isinstance(lst, list) and lst and isinstance(lst[0], str):
        return lst[0]
    return None


def load_arena55k() -> pd.DataFrame:
    path = download(ARENA55K_URL, RAW / "arena55k_train.csv")
    df = pd.read_csv(path)
    rows = []
    for r in df.itertuples(index=False):
        prompt = _first(r.prompt)
        for side in ("a", "b"):
            rows.append({
                "source": "arena55k",
                "battle_id": f"a55k-{r.id}",
                "model": getattr(r, f"model_{side}"),
                "input": prompt,
                "output": _first(getattr(r, f"response_{side}")),
            })
    return pd.DataFrame(rows)


def load_arena33k() -> pd.DataFrame:
    """TA-suggested dataset. Gated: requires HF login + accepting terms on the website."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        token = os.environ.get("HF_TOKEN")
        files = [f for f in list_repo_files(ARENA33K_REPO, repo_type="dataset", token=token)
                 if f.endswith(".parquet")]
        path = hf_hub_download(ARENA33K_REPO, files[0], repo_type="dataset", token=token,
                               local_dir=RAW / "arena33k")
    except Exception as e:  # GatedRepoError / 401 etc.
        print(f"[data] ERROR: could not access gated dataset {ARENA33K_REPO}: {type(e).__name__}: {e}\n"
              "        -> log in at huggingface.co, accept the dataset terms, create a read token, "
              "and `export HF_TOKEN=...`; or use --source arena55k (public).", file=sys.stderr)
        raise SystemExit(2)
    df = pd.read_parquet(path)
    if "language" in df.columns:
        df = df[df["language"] == "English"]
    rows = []
    for r in df.itertuples(index=False):
        for side in ("a", "b"):
            conv = list(getattr(r, f"conversation_{side}"))
            user = next((t["content"] for t in conv if t["role"] == "user"), None)
            asst = next((t["content"] for t in conv if t["role"] == "assistant"), None)
            rows.append({"source": "arena33k", "battle_id": f"a33k-{r.question_id}",
                         "model": getattr(r, f"model_{side}"), "input": user, "output": asst})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- helpers
def ascii_ratio(s: str) -> float:
    return sum(c.isascii() for c in s) / max(1, len(s))


def prompt_hash(s: str) -> str:
    return hashlib.sha1(" ".join(s.lower().split()).encode("utf-8")).hexdigest()[:16]


def group_split(df: pd.DataFrame, fracs=(0.8, 0.1, 0.1), seed=42) -> pd.Series:
    groups = np.array(sorted(df["prompt_id"].unique()), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_tr, n_va = int(fracs[0] * n), int(fracs[1] * n)
    assign = {g: "train" for g in groups[:n_tr]}
    assign.update({g: "val" for g in groups[n_tr:n_tr + n_va]})
    assign.update({g: "test" for g in groups[n_tr + n_va:]})
    return df["prompt_id"].map(assign)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="arena55k", choices=["arena55k", "arena33k", "both"])
    ap.add_argument("--families", default=",".join(DEFAULT_FAMILIES),
                    help="comma-separated family labels to keep (see families.py)")
    ap.add_argument("--per_class", type=int, default=4000,
                    help="max examples per family; all families are undersampled to "
                         "min(per_class, smallest family size) for balance")
    ap.add_argument("--min_output_chars", type=int, default=20)
    ap.add_argument("--min_ascii_ratio", type=float, default=0.9,
                    help="drop prompts that are mostly non-ASCII (rough English filter)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    frames = []
    if args.source in ("arena55k", "both"):
        frames.append(load_arena55k())
    if args.source in ("arena33k", "both"):
        frames.append(load_arena33k())
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    log = {"raw_examples": n0}

    df["family"] = df["model"].map(model_to_family)
    keep = [f.strip() for f in args.families.split(",") if f.strip()]
    df = df[df["family"].isin(keep)]
    log["after_family_filter"] = len(df)

    df = df.dropna(subset=["input", "output"])
    df["input"] = df["input"].astype(str).str.strip()
    df["output"] = df["output"].astype(str).str.strip()
    df = df[(df["input"].str.len() > 0) & (df["output"].str.len() >= args.min_output_chars)]
    log["after_empty_short_filter"] = len(df)

    df = df[df["input"].map(ascii_ratio) >= args.min_ascii_ratio]
    log["after_ascii_filter"] = len(df)

    df = df.drop_duplicates(subset=["family", "output"])
    log["after_dedupe"] = len(df)

    df["prompt_id"] = df["input"].map(prompt_hash)
    df["domain"] = df["input"].map(tag_domain)

    pre_counts = df["family"].value_counts().to_dict()
    n_per = min(args.per_class, min(pre_counts.values()))
    df = df.groupby("family").sample(n=n_per, random_state=args.seed).reset_index(drop=True)
    log["per_class"] = n_per
    log["after_balance"] = len(df)

    df["split"] = group_split(df, seed=args.seed)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    df.insert(0, "id", np.arange(len(df)))

    cols = ["id", "split", "family", "model", "input", "output", "domain", "prompt_id", "battle_id", "source"]
    OUT.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(OUT / "all.csv", index=False)
    for s in ("train", "val", "test"):
        df[df["split"] == s][cols].to_csv(OUT / f"{s}.csv", index=False)

    stats = {
        "args": vars(args),
        "filtering": log,
        "family_counts_before_balance": pre_counts,
        "family_by_split": pd.crosstab(df["family"], df["split"]).to_dict(),
        "family_by_domain": pd.crosstab(df["family"], df["domain"]).to_dict(),
        "models_per_family": {f: g["model"].value_counts().to_dict() for f, g in df.groupby("family")},
        "unique_prompts": int(df["prompt_id"].nunique()),
        "prompt_overlap_between_splits": int(
            df.groupby("prompt_id")["split"].nunique().gt(1).sum()),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, default=int))
    with open(ROOT / "data" / "STATS.md", "w") as f:
        f.write("# Processed dataset statistics (auto-generated by prepare_dataset.py)\n\n")
        f.write(f"Source(s): `{args.source}`  |  families: {keep}  |  per-class: {n_per}  |  seed: {args.seed}\n\n")
        f.write("## Filtering funnel\n\n" + "\n".join(f"- {k}: {v}" for k, v in log.items()) + "\n\n")
        f.write("## Family counts before balancing\n\n" + "\n".join(f"- {k}: {v}" for k, v in pre_counts.items()) + "\n\n")
        f.write("## Family x split\n\n" + pd.crosstab(df["family"], df["split"], margins=True).to_markdown() + "\n\n")
        f.write("## Family x domain (heuristic tags)\n\n" + pd.crosstab(df["family"], df["domain"], margins=True).to_markdown() + "\n\n")
        f.write("## Fine-grained models per family\n\n")
        for fam, g in df.groupby("family"):
            f.write(f"- **{fam}**: " + ", ".join(f"{m} ({c})" for m, c in g["model"].value_counts().items()) + "\n")
    print(json.dumps(log, indent=2))
    print(pd.crosstab(df["family"], df["split"], margins=True))
    print(pd.crosstab(df["family"], df["domain"], margins=True))
    print(f"[data] wrote {OUT}/{{all,train,val,test}}.csv, stats.json, data/STATS.md")


if __name__ == "__main__":
    main()
