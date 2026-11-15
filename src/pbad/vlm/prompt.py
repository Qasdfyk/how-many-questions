"""Prompts for estimation (image + list of user facts -> JSON) and for the S3 agent, plus parsers."""
from __future__ import annotations

import json
import re

SYSTEM = ("You are an expert dietitian. You estimate the energy content of meals from photos "
          "taken from directly above the plate.")

_SCHEMA = '{"kcal": <number>, "mass_g": <number>, "confidence": <number between 0 and 1>}'
_SCHEMA_COT = '{"reasoning": "<one or two short sentences>", "kcal": <number>, "mass_g": <number>, "confidence": <number between 0 and 1>}'


def estimation_text(facts: list[str], cot: bool) -> str:
    lines = ["Estimate the total calories (kcal) and the total mass (grams) of all food in the photo."]
    if facts:
        lines.append("Additional information from the person who ate the meal:")
        lines += [f"- {f}" for f in facts]
        lines.append("Use this information together with the photo.")
    else:
        lines.append("No additional information is available.")
    lines.append("'confidence' is your probability that your kcal estimate is within 20% of the truth.")
    lines.append(f"Answer with a single JSON object only: {_SCHEMA_COT if cot else _SCHEMA}")
    return "\n".join(lines)


def agent_text(facts: list[str], remaining: dict[str, str], asked: int, budget: int) -> str:
    lines = [
        "You will estimate the calories of the meal in the photo. Before estimating you may ask the person "
        f"who ate it a limited number of questions ({budget} in total, {budget - asked} left).",
        "Choose the ONE question whose answer would most reduce your uncertainty about the calories.",
    ]
    if facts:
        lines.append("What you already know:")
        lines += [f"- {f}" for f in facts]
    lines.append("Available questions:")
    lines += [f"{qid}: {text}" for qid, text in remaining.items()]
    lines.append('Answer with a single JSON object only: {"question": "<ID>"}')
    return "\n".join(lines)


_NUM = r"-?\d+(?:\.\d+)?"


def _extract_json(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?", "", text)
    for m in re.finditer(r"\{[^{}]*\}", text, flags=re.S):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _num(x) -> float | None:
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(_NUM, str(x).replace(",", ""))
    return float(m.group(0)) if m else None


def parse_estimate(text: str, kcal_range=(5, 4000), mass_range=(5, 3000)) -> dict:
    """Parse model output. Returns {kcal, mass_g, confidence, ok, clamped}. ok=False if no kcal found."""
    obj = _extract_json(text or "") or {}
    kcal, mass, conf = _num(obj.get("kcal")), _num(obj.get("mass_g")), _num(obj.get("confidence"))
    if kcal is None:  # regex fallback for broken JSON
        m = re.search(r'"?kcal"?\s*[:=]\s*"?(' + _NUM + ")", text or "")
        kcal = float(m.group(1)) if m else None
        m = re.search(r'"?mass_g"?\s*[:=]\s*"?(' + _NUM + ")", text or "")
        mass = float(m.group(1)) if m else mass
        m = re.search(r'"?confidence"?\s*[:=]\s*"?(' + _NUM + ")", text or "")
        conf = float(m.group(1)) if m else conf
    if kcal is None:
        return {"kcal": float("nan"), "mass_g": float("nan"), "confidence": float("nan"),
                "ok": False, "clamped": False}
    clamped = not (kcal_range[0] <= kcal <= kcal_range[1])
    kcal = min(max(kcal, kcal_range[0]), kcal_range[1])
    if mass is not None:
        mass = min(max(mass, mass_range[0]), mass_range[1])
    if conf is not None:
        conf = conf / 100 if conf > 1 else conf
        conf = min(max(conf, 0.0), 1.0)
    nan = float("nan")
    return {"kcal": kcal, "mass_g": nan if mass is None else mass,
            "confidence": nan if conf is None else conf, "ok": True, "clamped": clamped}


def parse_choice(text: str, allowed: list[str]) -> str | None:
    obj = _extract_json(text or "") or {}
    cand = str(obj.get("question", "")).strip().upper()
    if cand in allowed:
        return cand
    for m in re.finditer(r"Q\d", (text or "").upper()):
        if m.group(0) in allowed:
            return m.group(0)
    return None
