import math

from pbad.vlm.prompt import estimation_text, parse_choice, parse_estimate


def test_clean_json():
    p = parse_estimate('{"kcal": 450, "mass_g": 320, "confidence": 0.6}')
    assert p["ok"] and p["kcal"] == 450 and p["mass_g"] == 320 and p["confidence"] == 0.6


def test_fenced_and_text_around():
    p = parse_estimate('Sure!\n```json\n{"reasoning": "rice {big}", "kcal": "520 kcal", "mass_g": 400, '
                       '"confidence": 70}\n```')
    assert p["ok"] and p["kcal"] == 520 and p["confidence"] == 0.7


def test_broken_json_regex_fallback():
    p = parse_estimate('{"kcal": 300, "mass_g": 250, "confidence": 0.5')
    assert p["ok"] and p["kcal"] == 300 and p["mass_g"] == 250


def test_clamp_and_failure():
    p = parse_estimate('{"kcal": 99999}', kcal_range=(5, 4000))
    assert p["kcal"] == 4000 and p["clamped"] and math.isnan(p["mass_g"])
    assert not parse_estimate("I cannot tell.")["ok"]


def test_choice():
    assert parse_choice('{"question": "q3"}', ["Q1", "Q3"]) == "Q3"
    assert parse_choice("I would ask Q2 first", ["Q1", "Q3"]) is None
    assert parse_choice("I would ask Q3", ["Q1", "Q3"]) == "Q3"


def test_prompt_contains_facts():
    t = estimation_text(["The user reports 300 g."], cot=False)
    assert "300 g" in t and "JSON" in t and "reasoning" not in t
    assert "reasoning" in estimation_text([], cot=True)
