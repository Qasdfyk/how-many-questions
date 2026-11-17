# 04 — Hypothesis verification and experiment plan

## Data

- **Nutrition5k** [thames2021nutrition5k]: overhead photos (`realsense_overhead`, RGB) + `metadata` (kcal, mass, macros, ingredients with grams) + the official `dish_ids/splits`.
  - We download **only** the overhead photos and metadata, without videos (the full dataset is about 180 GB; the subset is 1.46 GB).
  - The official train split is divided into **validation** (for tuning the S4 selector, 300 dishes) and the rest, which we do not use because the VLM works zero-shot. The test set is the official test split.
- **Kaggle "Recipe images with nutritional information"** (the dataset from the topic description): **rejected as the main dataset**, because it describes recipes per serving, without the true mass in the photo and without ingredient grams, so user answers cannot be simulated from it. At most a qualitative test or an example in the discussion.
- (Optional) **MetaFood3D**: a generalisation test if time allows.

## Models

- Main: **Qwen2.5-VL-7B-Instruct** (originally planned in bf16; see "Changes after F2" for the 8 GB GPU version).
- Second (robustness of conclusions): **InternVL2.5-8B** or **LLaVA-OneVision-7B**, or a smaller model of another family.
- Serving: **vLLM** (batching, prefix caching of the same photo across many question variants).
- Model output: JSON `{"kcal": number, "mass_g": number, "confidence": 0-1}` in a fixed prompt (short CoT or none, decided after the pilot: none).

## Key computational trick: precomputing subsets

Answers enter the prompt as a list of facts, so their order does not matter. **We compute the VLM prediction once for every question subset of size ≤ 3** (from 7 questions: 1 + 7 + 21 + 35 = 64 subsets) and for every noise model. Then **all strategies (S0–S2, S4, oracle) are evaluated offline** by only choosing which subset a strategy "would ask". The S3 agent needs a separate online run.

Cost estimate: about 500 test + 300 validation dishes, 64 subsets, about 5 noise variants gives about 250k short generations, plus K = 5 S0 samples (uncertainty). Further reduction: identical prompts are computed only once (see "Changes after F2").

## Experiments

| ID | Goal | Configuration | Hypothesis |
|---|---|---|---|
| E0 | Pilot | 50 validation dishes, S0 + full information, prompt choice | — |
| E1 | Budget curve | S0–S4 + oracle, B = 0..3, U0 | H1, H2 |
| E2 | Realistic user | E1 for U1 (σ grid, bias), U2 (p grid), U3 | H3 |
| E3 | Question ranking | single-question gain (B = 1) for each Qi, U0 vs U1/U2 | H3, H4 |
| E4 | Subgroup analysis | dishes with fat > 30% kcal, large/small portions, number of ingredients | H4 |
| E5 | Second VLM | repeat E1–E2 (B ≤ 2) | robustness of conclusions |
| E6 (opt.) | Mini-pilot with people | 10–20 people, several N5k photos, real answers vs. simulation | validation of the noise model |

## Metrics and statistics

- MAE [kcal], MAPE [%], share of predictions within ±20% (*accuracy@20%*), median APE.
- **Error vs. budget B** curve for each strategy and noise model (the paper's main figure).
- Share of the oracle's gain = (error S0 − error strategy) / (error S0 − error oracle) at the same budget. "Recovered gain" relative to full information is reported as an auxiliary metric.
- Significance: paired Wilcoxon test on per-dish absolute errors + bootstrap 95% CI. Holm correction for multiple comparisons.

## Tools

| Element | Tool |
|---|---|
| Downloading and preparing N5k | HTTP from the public bucket, Python, pandas |
| Mapping ingredients to Q1–Q7 answers (fat and sauce lists) | Python + hand-made lists |
| Simulated user (U0–U3) | Python, numpy |
| VLM serving, prompt, JSON parsing | vLLM, Qwen2.5-VL |
| Subset precompute | batch script, resumable cache |
| S4 selector, offline strategy evaluation | LightGBM |
| S3 agent | dialogue loop with vLLM |
| Analysis, plots, tests | pandas, scipy, matplotlib (project plot style) |

## Project phases

Each phase has its products and an explicit completion criterion. The next phase starts only after the previous criterion is met. After F3 there is a go/no-go decision point.

### F1 — Problem definition (12.10–09.11) ✓
- **Goal:** choose a research gap and plan its verification.
- **Tasks:** literature review, gap identification, RQ1–RQ3, hypotheses H1–H4, experiment plan.
- **Products:** files `01`–`04`, `references.bib`, presentation on 09.11.
- **Completion criterion:** the instructor accepts the topic and plan on 09.11.

### F2 — Data and tools (10.11–23.11) ✓
- **Goal:** a working pipeline from photo to calorie estimate with answers.
- **Tasks:** download Nutrition5k, validation/test split; map ingredients to Q1–Q7 answers; simulated user U0–U3 with unit tests; run the VLM in vLLM, prompt and JSON parser; paper skeleton (IMRaD).
- **Products:** code repository, `dishes.parquet` (GT + answers), simulator module, paper skeleton.
- **Completion criterion:** the pipeline works end-to-end on 10 dishes (photo → questions → answers → kcal); demo in class on 23.11. **Met on 15.11.**

### F3 — Pilot and calibration (24.11–30.11)
- **Goal:** check whether the method works before spending GPU time on full experiments.
- **Tasks:** E0 on 50 validation dishes: prompt choice (with/without CoT), per-dish time and real GPU cost, H1 and H3 thresholds, freezing the protocol (prompt, question pool, noise grid, metrics).
- **Products:** short pilot report, frozen experiment protocol.
- **Completion criterion / go-no-go:**
  - < 5% VLM answer parsing errors;
  - S0 MAPE consistent with the literature (about 30–60%);
  - full information lowers MAPE relative to S0, i.e. there is an effect to study;
  - full precompute cost ≤ 2 GPU days.
- **If no-go:** change the prompt, a 3B or 4-bit model, a smaller question pool (5 instead of 7) or a smaller noise grid.

### F4 — Main experiments (01.12–20.12)
- **Goal:** answer RQ1–RQ3 (hypotheses H1–H3).
- **Tasks:** precompute validation + test (64 subsets × noise variants), train the S4 selector on validation, online run of the S3 agent, evaluation E1–E3, statistical tests.
- **Products:** result tables, error vs. budget curves (main figure), Wilcoxon results with CIs, the *Results* section in the first draft.
- **Completion criterion:** each of H1–H3 is resolved (confirmed / rejected) on the test set; draft with results ready for our first discussion.

### F5 — Extended analysis and robustness (21.12–11.01)
- **Goal:** check whether the conclusions generalise.
- **Tasks:** E4 subgroups (H4: fatty dishes, small/large portions), E5 repeating E1–E2 on a second VLM, optionally E6 mini-pilot with 10–20 people.
- **Products:** subgroup analysis, model comparison, possibly a comparison of real human answers with the simulation.
- **Completion criterion:** conclusions confirmed on the second model or differences described in the *Discussion*; draft 2 ready for the second discussion.

### F6 — Paper and presentation (12.01–25.01)
- **Goal:** finish the publication.
- **Tasks:** revisions after the second discussion, *Discussion* and *Limitations*, abstract, repository cleanup (code + reproducible results), final slides.
- **Products:** finished paper, repository with reproduction instructions, presentation.
- **Completion criterion:** paper submitted and presentation given on 25.01.

### Phase summary

| Phase | Dates | Milestone |
|---|---|---|
| F1 Definition | 12.10–09.11 | plan presentation 09.11 ✓ |
| F2 Data and tools | 10.11–23.11 | end-to-end pipeline on 10 dishes ✓ |
| F3 Pilot | 24.11–30.11 | **go/no-go** + frozen protocol |
| F4 Experiments | 01.12–20.12 | H1–H3 resolved, draft 1 |
| F5 Robustness | 21.12–11.01 | second VLM, H4, draft 2 |
| F6 Publication | 12.01–25.01 | paper + presentation 25.01 |

## Risks

- **The VLM does not keep the format / returns absurd values** → JSON validation, retries, clamping to a sensible range, reporting the error rate.
- **Not enough GPU** → 3B or 4-bit model, smaller noise grid, a subset of the test set.
- **The literature noise model does not match reality** → sensitivity analysis over the σ and p grids (conclusions as a function of noise), optional mini-pilot E6.
- **S4 does not beat S2** → that is also a result: "a static order is enough", which simplifies the app. Reported honestly.

## Changes after F2 (15.11)

- **Main model: Qwen2.5-VL-7B-Instruct-AWQ** instead of 7B bf16, because we have an 8 GB GPU (RTX 3070 Laptop, WSL).
  The AWQ weights take 6.7 GiB, so 3 GB of weights are kept in RAM (`--cpu-offload-gb 3`, no CUDA graphs).
  The cost is throughput (about 1.2 s per call at concurrency 8). On a larger GPU the same code
  works with a different config.
- **Qwen2.5-VL-3B-AWQ rejected** already in the 10-dish demo: it anchors on fixed values
  (35/350 kcal), and with full information it is worse than without (MAPE 60% → 82%).
- **Prompt deduplication** instead of a manually reduced grid: identical prompts are computed once.
- **Q2 under U1 noise:** the portion category follows from the same perturbed mass perception as Q1
  (a consistent user). Without this, H3 would be tested in Q2's favour from the start.
- **U1 calibration:** σ = 0.33 gives analytically 56% of answers within ±25% (exactly 50% would require σ ≈ 0.375).
- **Q2** shows the user mass ranges (validation terciles: < 153 g, 153–288 g, > 288 g).

## Decisions after the F3 pilot (15.11)

Pilot: 7B-AWQ, 50 validation dishes. Without CoT: 0% parsing errors, S0 MAPE 85%, full information 51%,
about 1.2 s per call. With CoT: MAPE 89% / 68%, about 2.8 s per call. After dropping dishes < 50 kcal:
S0 39%, full information 43% (median APE 35% in both cases).

- **No CoT**: slower and worse with it.
- **Dishes < 50 kcal are dropped** (`data.min_kcal: 50`). At a few kcal the percentage error
  explodes (10 kcal vs. 35 kcal = 250%) and a single such dish dominates the mean. Dataset: 300 validation, 424 test.
- **"Full information does not help" is treated as a finding**, without forcing a prompt fix. Therefore H1
  is measured against the **oracle** at the same budget (`share_of_oracle`), and the recovered gain relative to
  all 7 questions is reported as an auxiliary metric.
- **H4:** 82% of dishes get > 10% of kcal from fat, so the 10% threshold does not split anything. **We use a 30% threshold.**
- **σ = 0.33 stays** (as in the plan and presentation). The σ ∈ {0.1, 0.33, 0.5} grid shows
  how the conclusions depend on noise anyway, and 0.375 (exactly 50% within ±25%) lies inside this grid.
- **A compute budget of up to 48 h** is acceptable (full protocol: about 107k unique calls, about 36 h).
- **Second VLM (E5)**: later, if a larger GPU becomes available.

## F4 results (17.11, Qwen2.5-VL-7B-AWQ, test n = 424, fit on validation n = 300)

Generated by `scripts/05_eval.py`; tables and figures in `outputs/results/Qwen2.5-VL-7B-Instruct-AWQ/`
(`summary_test.csv`, `tests_test.csv`, `hypotheses_test.md`, `fig1`–`fig3`, `figS1`). 0% parse failures,
S3 invalid-choice rate 1.6%.

MAPE [%], photo only (S0) = 52.8:

| user | B | S1 random | S2 fixed | S3 agent | S4 ours | oracle | all 7 |
|---|---|---|---|---|---|---|---|
| U0 | 1 | 51.4 | 46.1 | 53.5 | 47.7 | 31.1 | 42.2 |
| U0 | 2 | 49.2 | 44.1 | 47.5 | 43.9 | 24.6 | |
| U0 | 3 | 47.8 | 43.1 | 46.5 | 42.7 | 21.7 | |
| U1 σ=0.33 | 1 | 51.8 | 46.3 | 54.5 | 48.9 | 31.6 | 44.4 |
| U1 σ=0.33 | 2 | 49.6 | 43.5 | 47.3 | 43.8 | 24.2 | |
| U1 σ=0.33 | 3 | 48.4 | 42.6 | 46.8 | 43.6 | 22.0 | |

- **H1: depends on the reference.** Against all 7 questions, S4 at B = 2 recovers 84% of the gain (S2 82%), above
  the 70% threshold, so on the full test set all 7 questions do help (52.8 → 42.2), unlike in the 50-dish pilot.
  Against the per-dish oracle at B = 2, S4 reaches only 32% of the gain: H1 is **not** supported under the
  post-pilot definition. The oracle is a genuine, unreachable post-hoc bound (sampling noise is negligible: in 95% of
  dishes all K = 5 samples are identical, and min-of-K gives only 52.5 → 51.5), so the gap is real headroom for
  better per-dish selection.
- **H2: partially supported.** S4 < S1 significantly for every user and B ∈ {1, 2} (Wilcoxon + Holm, p_holm ≤ 0.03);
  S4 < S3 significantly for U0 and U1 at B = 1 and B = 2. S4 vs S2: no significant difference anywhere
  (differences −2.5 to +2.6 pp, CIs contain 0). The learned selector does not beat a good fixed order. A likely cause is that
  the uncertainty features are almost uninformative (identical K samples); the selector's top features are the
  candidate question and the current mass/shift, not the variance.
- **H3: partially supported.** The Q1 (grams) gain drops by 37% under U1 σ=0.33 (5.9 → 3.7 pp, threshold 30%), and
  Q2 (size category) then gives a larger gain than Q1 (6.5 vs 3.7 pp). But Q2 is already the best single question for U0, so
  the ranking does **not** change (Q2 > Q1 > Q4 > Q7 > Q5 > Q3 > Q6 in both). Q3 gives no gain (−0.4 pp).
- **H4: not supported.** For fat-heavy dishes (> 30% kcal from fat, n = 297), Q3 *increases* the error (−3.2 pp) and
  Q2 is best (5.6 pp); Q3 helps only lean dishes (+6.2 pp). Telling the VLM "fried / oil" seems to push it to
  over-correct.
- **Other findings:** Q6 (list ingredients) hurts in every variant (≈ −7 pp); the S3 VLM agent at B = 1 is worse than
  photo only (it often picks Q6/Q7); Q2 helps most for small portions (+20.6 pp) and not at all for large ones.
