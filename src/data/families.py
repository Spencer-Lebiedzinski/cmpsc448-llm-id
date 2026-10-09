"""Map fine-grained Chatbot Arena model names to LLM *families* (vendor lineages).

Only first-party models are mapped. Community fine-tunes of open models
(e.g. vicuna, zephyr, openhermes, wizardlm, tulu) are deliberately EXCLUDED,
because their "family" is ambiguous (base-model lineage vs. fine-tuner style).
Document this choice in the report (Dataset curation section).
"""
import re
from typing import Optional

# (family label, regex on lower-cased model name). First match wins.
FAMILY_RULES = [
    ("GPT", r"^gpt-(3\.5|4)"),                                 # OpenAI
    ("Claude", r"^claude-"),                                   # Anthropic
    ("Google", r"^(gemini-|palm-2|bard)"),                    # Google (Gemini / PaLM 2)
    ("Llama", r"^(llama-2-\d+b-chat|codellama-|llama-13b|llama-3)"),  # Meta (official releases)
    ("Mistral", r"^(mistral-|mixtral-)"),                     # Mistral AI (official releases)
    ("Qwen", r"^qwen"),                                        # Alibaba
    ("DeepSeek", r"^deepseek"),                               # DeepSeek
]
_COMPILED = [(fam, re.compile(rx)) for fam, rx in FAMILY_RULES]

DEFAULT_FAMILIES = ["GPT", "Claude", "Google", "Llama", "Mistral"]


def model_to_family(model_name: str) -> Optional[str]:
    if not isinstance(model_name, str):
        return None
    name = model_name.strip().lower()
    for fam, rx in _COMPILED:
        if rx.search(name):
            return fam
    return None


if __name__ == "__main__":
    for m in ["gpt-4-1106-preview", "claude-instant-1", "palm-2", "gemini-pro-dev-api",
              "llama-2-70b-chat", "llama2-70b-steerlm-chat", "mixtral-8x7b-instruct-v0.1",
              "openhermes-2.5-mistral-7b", "vicuna-13b", "qwen1.5-72b-chat"]:
        print(f"{m:35s} -> {model_to_family(m)}")
