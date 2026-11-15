"""Simulated users U0-U3.

Variant parameters (all optional, combinable):
  mass_sigma, mass_mu : log-normal multiplicative error on perceived mass, m' = m * exp(eps),
                        eps ~ N(mu, sigma). Drives Q1 and (consistently, same eps) the Q2 category.
  cat_p               : probability of a categorical mistake: Q2 -> adjacent size, Q3/Q4 flipped,
                        Q5 -> second-largest ingredient, Q6 -> each ingredient forgotten,
                        Q7 -> off by one.
  dont_know_q         : probability the user answers "I don't know" (budget still spent).

Answers are drawn once per (seed, dish, variant) so a given user is consistent across subsets.
"""
from __future__ import annotations

import zlib

import numpy as np

from pbad.data.build import size_category
from pbad.questions import QIDS

SIZES = ["small", "medium", "large"]


def _rng(seed: int, dish_id: str, variant: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(dish_id.encode()), zlib.crc32(variant.encode())])


def simulate(dish, variant: str, params: dict, q2_thresholds, seed: int) -> dict[str, object]:
    """Return {qid: answer or None} for one dish (a row/mapping with true answers)."""
    rng = _rng(seed, dish["dish_id"], variant)
    # Draw all random numbers up-front in a fixed order so answers don't depend on params presence.
    z = rng.standard_normal()
    u_cat = rng.random(len(QIDS))
    u_side = rng.random()
    u_q6 = rng.random(64)
    u_dk = rng.random(len(QIDS))

    sigma = params.get("mass_sigma", 0.0)
    mu = params.get("mass_mu", 0.0)
    p = params.get("cat_p", 0.0)
    q = params.get("dont_know_q", 0.0)
    eps = mu + sigma * z

    mass = float(dish["q1_mass"])
    perceived = mass * float(np.exp(eps))
    ans: dict[str, object] = {}

    ans["Q1"] = max(5.0, 5 * round(perceived / 5)) if (sigma > 0 or mu != 0) else round(mass)
    size = str(size_category(np.array([perceived]), q2_thresholds)[0])
    if u_cat[1] < p:
        i = SIZES.index(size)
        i = 1 if i != 1 else (0 if u_side < 0.5 else 2)
        size = SIZES[i]
    ans["Q2"] = size
    ans["Q3"] = bool(dish["q3_fat"]) ^ bool(u_cat[2] < p)
    ans["Q4"] = bool(dish["q4_sauce"]) ^ bool(u_cat[3] < p)
    ans["Q5"] = dish["q5_second"] if u_cat[4] < p else dish["q5_main"]
    ingr = list(dish["q6_ingredients"])
    kept = [x for x, u in zip(ingr, u_q6) if u >= p] or ingr[:1]
    ans["Q6"] = kept
    n = int(dish["q7_count"])
    if u_cat[6] < p:
        n = n + 1 if (u_side < 0.5 or n == 1) else n - 1
    ans["Q7"] = n

    for i, qid in enumerate(QIDS):
        if u_dk[i] < q:
            ans[qid] = None
    return ans
