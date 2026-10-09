"""Heuristic task/domain tagging of user prompts (for RQ3 cross-domain generalization).

The Arena-55k release has no topic labels, so we tag prompts with transparent
keyword rules. Priority: coding > math > writing > general.
These are *heuristic* labels -- report that, and (ideally) hand-check ~50 random
prompts per tag to estimate tag precision for the report.
"""
import re

CODE_RE = re.compile(
    r"```|\b(python|javascript|typescript|java|c\+\+|c#|golang|rust|sql|html|css|regex|"
    r"function|code|coding|script|compile|compiler|debug|bug|api|json|yaml|bash|shell|linux|"
    r"pandas|numpy|react|django|docker|kubernetes|algorithm|program|programming|"
    r"stack ?trace|exception|syntax)\b|\bdef \w+\(|\bimport \w+|#include",
    re.I,
)
MATH_RE = re.compile(
    r"\b(solve|calculate|compute|equation|integral|derivative|probability|prove|proof|theorem|"
    r"math|mathematics|algebra|geometry|arithmetic|sum of|how many|percent(age)?|factorial|"
    r"prime number|matrix)\b|\d+\s*[\+\-\*/\^x=]\s*\d+",
    re.I,
)
WRITING_RE = re.compile(
    r"\b(write|compose|draft|poem|poetry|story|stories|essay|email|e-mail|letter|song|lyrics|"
    r"haiku|limerick|sonnet|rap|joke|rewrite|paraphrase|rephrase|summari[sz]e|"
    r"role ?play|pretend|fiction|screenplay|script for|slogan|tweet|blog)\b",
    re.I,
)

DOMAINS = ["coding", "math", "writing", "general"]


def tag_domain(prompt: str) -> str:
    if not isinstance(prompt, str):
        return "general"
    if CODE_RE.search(prompt):
        return "coding"
    if MATH_RE.search(prompt):
        return "math"
    if WRITING_RE.search(prompt):
        return "writing"
    return "general"
