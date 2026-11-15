# How Many Questions Are Enough?

**Uncertainty-driven, budget-constrained clarification with noisy user answers for image-based calorie estimation.**

A vision-language model (VLM) estimates the calories of a meal from a single photo. It may ask the user
at most 0–3 questions (e.g. "how much did it weigh?", "was it fried?"), and the user answers imperfectly.
We study how much each question helps, which questions to ask under a budget, and whether choosing them
per photo from the model's uncertainty beats a fixed order, random choice and a VLM agent picking on its own.

Data: [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) (overhead RGB photos with weighed
ingredients). Model: Qwen2.5-VL-7B-Instruct-AWQ served with vLLM on an 8 GB GPU.

Authors: Adam Sokołowski, Antoni Grajek (MSc research project).

| Document | Contents |
|---|---|
| [`docs/01_gap_and_topic.md`](docs/01_gap_and_topic.md) | problem, research gap, RQ1–RQ3, contributions |
| [`docs/02_literature_review.md`](docs/02_literature_review.md) | literature review ([`references.bib`](docs/references.bib)) |
| [`docs/03_brainstorm_and_hypotheses.md`](docs/03_brainstorm_and_hypotheses.md) | questions Q1–Q7, strategies S0–S4, simulated users U0–U3, hypotheses H1–H4 |
| [`docs/04_experiment_plan.md`](docs/04_experiment_plan.md) | experiments, metrics, phases, decisions after the pilot |
| [`PLAN.md`](PLAN.md) | implementation plan |

## Setup (Linux / WSL, CUDA GPU)

```bash
uv sync --extra vlm      # Python 3.12 venv + vLLM
uv run pytest -q         # tests: simulator, Q1–Q7 mapping, parser, strategies
```

## Pipeline

| Step | Command | Time (RTX 3070 Laptop) |
|---|---|---|
| 1. Data (1.46 GB, resumable) | `scripts/01_download.sh` | ~3 min |
| 2. `data/dishes.parquet` | `uv run python scripts/02_build.py` | seconds |
| 3. VLM server (separate terminal/tmux) | `scripts/serve_vllm.sh Qwen/Qwen2.5-VL-7B-Instruct-AWQ` | ~2 min to start |
| 4. Demo / pilot | `uv run python scripts/03_pilot.py --n 50 --cot-mode both` | ~40 min |
| 5. Precompute (background) | `nohup uv run python scripts/04_precompute.py > outputs/precompute.log 2>&1 &` | ~36 h |
| 6. S3 agent (background) | `nohup uv run python scripts/06_agent.py --split test > outputs/agent.log 2>&1 &` | ~1–2 h |
| 7. Evaluation + figures | `uv run python scripts/05_eval.py` | minutes |

Everything reads `configs/default.yaml`; another config: `--config configs/x.yaml` (merged on top of the default),
another model: `--model ...` (must match the served one). `--n N` limits the number of dishes (smoke tests).
`04_precompute.py --dry-run` counts unique prompts without calling the VLM.

**Resuming:** every VLM call goes into `outputs/cache/vlm_<model>.jsonl` (key = hash of the prompt, image and
temperature). An interrupted precompute or agent run just needs to be started again. Identical prompts
(e.g. noise that did not change an answer, or a "don't know" that reduces to a smaller subset) are computed
once: the full protocol is ~107k unique prompts instead of ~275k cells.

## How it works

1. **Precompute.** Answers enter the prompt as an unordered list of facts, so the VLM prediction is computed
   once for every subset of at most 3 of the 7 questions (64 subsets) and every simulated user.
2. **Offline strategies.** S0 (photo only), S1 (random), S2 (fixed greedy order learned on validation),
   S4 (LightGBM selector using the model's uncertainty) and the oracle are evaluated by looking up the subset
   each would ask.
3. **Online agent.** S3 lets the VLM choose its own questions; its final estimate is a cache hit.
4. **Statistics.** Paired Wilcoxon tests with bootstrap CIs and Holm correction, plus automatic checks of H1–H4.

## Results

`outputs/results/<model>/`: `summary_test.csv` (MAE, MAPE, acc@20%, share of the oracle's gain per strategy ×
user × budget), `tests_test.csv` (Wilcoxon + bootstrap CI + Holm, H2), `question_ranking_test.csv` (E3/E4),
`hypotheses_test.md` (automatic check of H1–H4), figures `fig*.png/.pdf` with their data as `.csv`.
Pilot: `outputs/pilot/*.json`.

## Code (`src/pbad/`)

`data/` download and dish table · `questions.py` question pool Q1–Q7 · `simulator.py` users U0–U3 ·
`vlm/` prompt, parser, cached client · `precompute.py` · `uncertainty.py` · `strategies.py` S0–S2, S4,
oracle · `agent.py` S3 · `evaluate.py`, `stats.py` · `plotstyle.py` (chart style), `plots.py`.
