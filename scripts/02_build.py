"""Build data/dishes.parquet (GT + true answers Q1-Q7, val/test split)."""
import argparse

from pbad.config import load_config
from pbad.data.build import build, save

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
cfg = load_config(ap.parse_args().config)
df = build(cfg)
save(df, cfg["paths"]["dishes"])
print(df.groupby("split").agg(n=("dish_id", "size"), kcal=("kcal", "median"), mass=("mass_g", "median"),
                              fat=("q3_fat", "mean"), sauce=("q4_sauce", "mean"), q7=("q7_count", "median")))
print("Q2 thresholds (g):", df.attrs["q2_thresholds"])
print(df.q2_size.value_counts().to_dict())
print("top main ingredients:", df.q5_main.value_counts().head(10).to_dict())
print("fat share > 10%:", (df.fat_kcal_share > 0.1).mean().round(3))
