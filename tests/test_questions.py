from pbad.data.build import size_category, true_answers
from pbad.questions import QIDS, format_fact, parse_key, subset_key, subsets

QCFG = {"q3_min_grams": 1, "q4_min_grams": 1, "q7_min_grams": 10, "q6_min_grams": 2}


def ing(name, g):
    return {"name": name, "grams": g, "kcal": 0, "fat": 0}


def test_subsets_count():
    s = subsets(QIDS, 3)
    assert len(s) == 64 and len(set(map(subset_key, s))) == 64


def test_key_roundtrip():
    assert subset_key(("Q3", "Q1")) == "Q1+Q3"
    assert parse_key("Q1+Q3") == ("Q1", "Q3") and parse_key("") == ()
    assert parse_key("ALL") == QIDS


def test_true_answers_mapping():
    a = true_answers([ing("brown rice", 100), ing("olive oil", 5), ing("salt", 0.5), ing("soy sauce", 3),
                      ing("chicken", 120), ing("plate only", 0)], 228.5, QCFG)
    assert a["q3_fat"] and a["q4_sauce"]
    assert a["q5_main"] == "chicken" and a["q5_second"] == "brown rice"
    assert a["q6_ingredients"] == ["brown rice", "chicken", "olive oil", "soy sauce"]
    assert a["q7_count"] == 2


def test_fried_food_counts_as_fat():
    a = true_answers([ing("french fries", 150)], 150, QCFG)
    assert a["q3_fat"] and not a["q4_sauce"] and a["q7_count"] == 1


def test_size_category():
    assert list(size_category([50, 120, 200, 400], (120, 240))) == ["small", "medium", "medium", "large"]


def test_facts():
    thr = (120, 240)
    assert format_fact("Q1", None, thr) is None
    assert "300 g" in format_fact("Q1", 300, thr)
    assert "large" in format_fact("Q2", "large", thr) and "240" in format_fact("Q2", "large", thr)
    assert "not fried" in format_fact("Q3", False, thr)
    assert "chicken, rice" in format_fact("Q6", ["chicken", "rice"], thr)
