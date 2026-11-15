# Implementation plan — "How Many Questions Are Enough?"

## Context

The topic, research gap, hypotheses H1–H4 and the experiment plan E0–E6 are in `docs/01`–`04` (plus `references.bib` and the 09.11 presentation). This plan covers the **implementation** of phases F2–F4: Nutrition5k → simulated user → VLM → question-selection strategies → evaluation and figures. Long GPU runs are started by the authors in the background.

Environment:
- WSL Ubuntu, **RTX 3070 Laptop 8 GB** GPU, 15 GB RAM, Python 3.12 venv via `uv`, no `gsutil` (data over HTTP).
- **8 GB VRAM ≠ Qwen2.5-VL-7B bf16** from the original plan. The model is chosen in the config; the pilot compared **Qwen2.5-VL-3B-Instruct-AWQ** and **Qwen2.5-VL-7B-Instruct-AWQ** in vLLM. Second VLM (E5): a model of another family, if a larger GPU becomes available. Same code, different config.

Status: F2 done (15.11), F3 pilot done with decisions recorded in `docs/04`; F4 precompute pending.

## Repository structure

```
docs/                01..04_*.md, references.bib, presentation (Polish)
PLAN.md, CLAUDE.md, README.md, pyproject.toml (uv, Python 3.12)
configs/             default.yaml (model, max_pixels, question pool, noise grid, paths, seed)
src/pbad/
  data/download.py   N5k download over HTTP from the public GCS bucket
  data/build.py      dishes.parquet: GT (kcal, mass, macros, ingredients with grams) + Q1–Q7 answers
  data/lexicon.py    hand-made lists: fats (oil, butter…), sauces/dressings
  questions.py       Q1–Q7 pool: question text, type, fact formatting for the prompt
  simulator.py       users U0–U3 (log-normal σ/μ, category mistakes p, "don't know" q), seeded
  vlm/prompt.py      prompt (image + list of facts), JSON output {kcal, mass_g, confidence}, parsers
  vlm/client.py      vLLM client (OpenAI-compatible server), retries, resumable cache
  precompute.py      every dish × subset ≤ 3 (64) × noise variant → prediction, with prompt deduplication
  uncertainty.py     K = 5 S0 samples (temperature > 0) → variance, stated confidence
  strategies.py      S0 none, S1 random, S2 static greedy (validation), S4 uncertainty selector (LightGBM), oracle
  agent.py           S3: online loop, the VLM picks the next question from the pool under budget B
  evaluate.py        MAE, MAPE, acc@20%, share of oracle gain, error vs. B curves, hypothesis checks
  stats.py           paired Wilcoxon + bootstrap 95% CI + Holm correction
  plotstyle.py       project-wide chart style
  plots.py           main figure (error vs. budget per strategy/noise), question ranking, subgroups
scripts/             thin CLIs: 01_download.sh, 02_build.py, 03_pilot.py, 04_precompute.py, 05_eval.py, 06_agent.py, serve_vllm.sh
tests/               pytest: simulator calibration, Q1–Q7 mapping, JSON parser, strategies on synthetic data
outputs/             results, tables, figures (cache not in git)
```

## Steps (following the phases in `docs/04`)

**F2 — data and tools** ✓
1. Environment: `uv venv -p 3.12`, dependencies (pandas, pyarrow, numpy, scipy, scikit-learn, lightgbm, matplotlib, pyyaml, openai, requests, pytest; vllm as an extra).
2. Download: only `metadata/*`, `dish_ids/splits/rgb_{train,test}_ids.txt` and `imagery/realsense_overhead/dish_*/rgb.png` (no depth/video; 1.46 GB). List files and sizes first, then download in parallel with resume.
3. `dishes.parquet`: parse the N5k CSVs (variable number of ingredient columns), only dishes with an overhead image; split: 300 validation from train (seeded), test = official test.
4. Q1–Q7 + fat/sauce lexicon + mass terciles (from validation).
5. Simulator U0–U3 + tests.
6. vLLM in WSL (OpenAI-compatible server), prompt, parser; **end-to-end demo on 10 dishes** (S0 vs. full information).

**F3 — pilot / go-no-go** ✓ (decisions in `docs/04`)
7. `03_pilot.py`: 50 validation dishes, with / without CoT; report: parsing error rate (< 5%), S0 MAPE (30–60%), full-information gain, time per call → precompute cost extrapolation.
8. Cost reduction: identical prompts computed once; full σ/p grid only for B ≤ 2; image prefix caching.

**F4 — main experiments**
9. Precompute validation + test (resumable, run in the background with `nohup`/`tmux`).
10. S2 (greedy order from validation), S4 (features: variance of K samples, confidence, mass_g from S0, answers so far; LightGBM predicts per-question error reduction; greedy step by step conditioned on the answers so far), oracle — all offline from the cache.
11. S3 online agent (B = 1..3, U0 and the main noise variant).
12. Evaluation E1–E4, statistical tests, figures and tables in `outputs/` → resolving H1–H4.
(E5 second VLM = the same pipeline with a different config; E6 mini-pilot with people is outside the code.)

## Verification

- `pytest` green (simulator, mapping, parser, strategies on synthetic data with a known optimum).
- Smoke: `scripts/03_pilot.py --n 10` produces predictions for all 10 dishes, S0 and full information, with a sensible MAPE.
- Offline evaluation on a small cache (about 15 dishes) produces curves and tables without errors before the full precompute.
