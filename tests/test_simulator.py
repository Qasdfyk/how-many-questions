import numpy as np

from pbad.simulator import simulate
from tests.conftest import THR


def test_u0_is_truth(dish):
    a = simulate(dish, "U0", {}, THR, 42)
    assert a["Q1"] == 300 and a["Q2"] == "large" and a["Q3"] is True and a["Q4"] is False
    assert a["Q5"] == "chicken" and a["Q6"] == ["chicken", "olive oil", "rice"] and a["Q7"] == 3


def test_deterministic(dish):
    p = {"mass_sigma": 0.33, "cat_p": 0.2, "dont_know_q": 0.2}
    assert simulate(dish, "V", p, THR, 1) == simulate(dish, "V", p, THR, 1)


def test_u1_calibration(dish):
    """sigma=0.33 log-normal -> roughly half within +-25% (Lucassen 2021).

    Analytically P(|exp(eps)-1|<=0.25) = 0.56 for sigma=0.33 (exactly 50% would need sigma~0.375);
    rounding to 5 g adds a little.
    """
    rel = []
    for i in range(4000):
        d = dish.copy()
        d["dish_id"] = f"d{i}"
        rel.append(simulate(d, "U1", {"mass_sigma": 0.33}, THR, 0)["Q1"] / 300)
    within = np.mean(np.abs(np.array(rel) - 1) <= 0.25)
    assert 0.52 <= within <= 0.62, within
    assert abs(np.median(np.log(rel))) < 0.03


def test_u1_bias(dish):
    rel = [simulate({**dish, "dish_id": f"d{i}"}, "B", {"mass_sigma": 0.33, "mass_mu": 0.2}, THR, 0)["Q1"] / 300
           for i in range(3000)]
    assert 0.15 < np.median(np.log(rel)) < 0.25


def test_u2_flip_rate_and_adjacency(dish):
    flips, sizes = 0, set()
    n = 4000
    for i in range(n):
        a = simulate({**dish, "dish_id": f"d{i}"}, "U2", {"cat_p": 0.2}, THR, 0)
        flips += a["Q3"] is False
        sizes.add(a["Q2"])
    assert abs(flips / n - 0.2) < 0.03
    assert sizes == {"large", "medium"}  # only adjacent category from 'large'


def test_u3_dont_know_rate(dish):
    n, none = 3000, 0
    for i in range(n):
        a = simulate({**dish, "dish_id": f"d{i}"}, "U3", {"dont_know_q": 0.2}, THR, 0)
        none += sum(v is None for v in a.values())
    assert abs(none / (7 * n) - 0.2) < 0.02


def test_variants_independent_of_other_params(dish):
    """Adding categorical noise must not change the perceived-mass draw."""
    a = simulate(dish, "X", {"mass_sigma": 0.33}, THR, 0)
    b = simulate(dish, "X", {"mass_sigma": 0.33, "cat_p": 0.1}, THR, 0)
    assert a["Q1"] == b["Q1"]
