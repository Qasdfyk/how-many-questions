"""Question pool Q1-Q7: wording shown to the user/agent and fact formatting for the VLM prompt.

An answer of None means "no information" (user said "I don't know") and yields no fact.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

QIDS = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7")


@dataclass(frozen=True)
class Question:
    qid: str
    text: str            # question asked to the user (English, as the agent sees it)
    kind: str            # numeric | category | binary | text | list | count


POOL = {
    "Q1": Question("Q1", "How much did the meal weigh (in grams)?", "numeric"),
    "Q2": Question("Q2", "Was the portion small, medium or large?", "category"),
    "Q3": Question("Q3", "Was the dish fried, or was oil/butter added?", "binary"),
    "Q4": Question("Q4", "Was there a sauce or dressing?", "binary"),
    "Q5": Question("Q5", "What is the main ingredient?", "text"),
    "Q6": Question("Q6", "Please list the ingredients.", "list"),
    "Q7": Question("Q7", "How many separate food items are on the plate?", "count"),
}


def format_fact(qid: str, answer, q2_thresholds: tuple[float, float]) -> str | None:
    if answer is None:
        return None
    if qid == "Q1":
        return f"The user reports the meal weighed about {answer:.0f} g."
    if qid == "Q2":
        t1, t2 = q2_thresholds
        ranges = {"small": f"under ~{t1:.0f} g", "medium": f"~{t1:.0f}-{t2:.0f} g", "large": f"over ~{t2:.0f} g"}
        return f"The user says the portion was {answer} ({ranges[answer]})."
    if qid == "Q3":
        return ("The user says the dish was fried or had oil/butter added." if answer
                else "The user says the dish was not fried and had no added oil/butter.")
    if qid == "Q4":
        return ("The user says there was a sauce or dressing." if answer
                else "The user says there was no sauce or dressing.")
    if qid == "Q5":
        return f"The user says the main ingredient is {answer}."
    if qid == "Q6":
        return f"The user lists the ingredients: {', '.join(answer)}."
    if qid == "Q7":
        return f"The user counts {int(answer)} separate food item(s) on the plate."
    raise KeyError(qid)


def subsets(pool=QIDS, max_size: int = 3) -> list[tuple[str, ...]]:
    """All subsets of the pool with size <= max_size, in canonical (pool) order."""
    out = []
    for k in range(max_size + 1):
        out += list(combinations(pool, k))
    return out


def subset_key(s) -> str:
    """Canonical string key of a question subset ('' for the empty set, 'ALL' kept as is)."""
    if isinstance(s, str):
        return s
    return "+".join(q for q in QIDS if q in s)


def parse_key(key: str) -> tuple[str, ...]:
    if key == "ALL":
        return QIDS
    return tuple(key.split("+")) if key else ()
