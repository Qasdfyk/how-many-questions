import numpy as np
import pandas as pd
import pytest

from pbad.config import load_config


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def dish():
    return pd.Series({"dish_id": "dish_test", "q1_mass": 300.0, "q3_fat": True, "q4_sauce": False,
                      "q5_main": "chicken", "q5_second": "rice",
                      "q6_ingredients": ["chicken", "olive oil", "rice"], "q7_count": 3})


THR = (120.0, 240.0)


def synthetic_dishes(n=40, seed=0):
    rng = np.random.default_rng(seed)
    mass = rng.uniform(50, 500, n)
    return pd.DataFrame({
        "dish_id": [f"dish_{i}" for i in range(n)], "kcal": mass * 1.5, "mass_g": mass,
        "q1_mass": mass, "q3_fat": rng.random(n) < 0.4, "q4_sauce": rng.random(n) < 0.3,
        "q5_main": "chicken", "q5_second": "rice", "q6_ingredients": [["chicken", "rice"]] * n,
        "q7_count": 2, "fat_kcal_share": rng.uniform(0, 0.6, n), "n_ingredients": rng.integers(1, 10, n),
        "q2_size": np.where(mass < 120, "small", np.where(mass < 240, "medium", "large")),
    })
