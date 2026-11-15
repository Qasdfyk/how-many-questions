# How Many Questions Are Enough? (PBAD project)

Master's research project (semester 2, PBAD course). Authors: Adam Sokołowski, Antoni Grajek.
Topic: uncertainty-driven selection of questions to the user, with a budget of 0–3 questions and noisy answers, for VLM calorie estimation from a meal photo (Nutrition5k).

## Where things are

- `PLAN.md` — the approved implementation plan (repo structure, steps F2–F4, verification). **We follow it step by step.**
- `docs/01_gap_and_topic.md` — topic, gap, RQ1–RQ3.
- `docs/02_literature_review.md` — literature review; keys `[...]` → `docs/references.bib`.
- `docs/03_brainstorm_and_hypotheses.md` — question pool Q1–Q7, strategies S0–S4 + oracle, users U0–U3, hypotheses H1–H4.
- `docs/04_experiment_plan.md` — data, models, precompute of 64 subsets, experiments E0–E6, metrics, phases F1–F6, changes after F2 and decisions after the F3 pilot.
- `docs/presentation_2026-11-09_pl.pptx` — the 09.11 presentation (in Polish).

## Environment

- WSL Ubuntu, RTX 3070 Laptop GPU **8 GB VRAM**, 15 GB RAM. No `gsutil` (data is downloaded over HTTP from the public `nutrition5k_dataset` bucket).
- Python: venv via `uv` with Python 3.12 (the system one is 3.14 — do not use it). vLLM as an OpenAI-compatible server.
- Long runs (precompute, S3 agent) are started by the user in the background (`tmux`/`nohup`); code must be resumable (cache).

## Conventions

- **Everything in the repository is in English** (docs, README, code, comments, figures). Conversation with the user is in Polish.
- Everything is seeded and driven by config (`configs/*.yaml`).
- `data/` and `outputs/cache/` are not committed.
- Commits: **never** add Claude as a co-author or "Generated with Claude Code" lines.
- The repository is **public** on GitHub (`Qasdfyk/how-many-questions`); the local folder and Python package stay `pbad`. Nothing private (data, keys, emails) goes into git.

## Charts — project standard

Every chart (paper, slides, reports) goes through `src/pbad/plotstyle.py` (`ps.apply()`, `ps.header()`,
`ps.save()`), in a research-report style: warm off-white surface, no frame, only a hairline baseline
and a hairline horizontal grid (solid, never dashed), thin 2 px lines, markers with a surface-colored ring.
- Left-aligned title + a one-sentence takeaway subtitle. Text always in ink colors, never in the series color.
- At most 3 colored series per panel (palette validated for color blindness), the rest gray; color belongs
  to the entity (S4 always blue, S2 orange, S3 aqua, S1 gray, oracle dashed gray).
- Legend for ≥ 2 series, direct labels used sparingly (line end, the series the story is about).
- One y-axis; small multiples instead of many series. Before/after comparisons as dumbbells.
- `ps.save()` writes PNG (slides) + PDF (paper) + CSV with the plotted data. Look at the PNG after
  generating it (label collisions, clipping).

## Status and commands (after the 15.11 session)

- F2 done: code for the whole F2–F4 pipeline in `src/pbad/`, scripts `scripts/01–06`, `pytest` (23 tests).
  Details and commands: `README.md`.
- Model: **Qwen2.5-VL-7B-Instruct-AWQ** with `--cpu-offload-gb 3` (preset in `scripts/serve_vllm.sh`), about 1.2 s per call.
  3B-AWQ rejected in the demo. vLLM 0.33 needs `VLLM_USE_FLASHINFER_SAMPLER=0` (no nvcc in WSL; set in the script).
- F3 pilot decisions (min_kcal 50, no CoT, H1 against the oracle, H4 30% threshold): `docs/04`, "Decisions after the F3 pilot".
- Stopping the server: `pkill -f "[v]llm serve"` (the brackets stop pkill from killing its own shell).
- VLM cache: `outputs/cache/vlm_<model>.jsonl` — do not delete, everything resumes from it.
