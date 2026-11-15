# 03 — Brainstorm and hypotheses

## Approaches considered

| # | Idea | Pros | Cons | Decision |
|---|---|---|---|---|
| 1 | Fine-tune a regressor (ViT/CNN) on Nutrition5k only | simple | without JFT-300M we cannot catch SOTA [Andrade 2025]; nothing new | optional baseline |
| 2 | VLM + a fixed list of questions (all of them) | easy | already done [Bhatambarekar 2025/26] | baseline ("full information") |
| 3 | VLM + an agent asking freely | natural | done before, no cost control | baseline |
| 4 | **VLM + budgeted, uncertainty-driven question selection** | new, measurable, computationally cheap | uncertainty needs a definition | **chosen** |
| 5 | **Simulated user with realistic noise** | new, easy to do offline | noise calibrated from literature, not our own study | **chosen** |
| 6 | Study with real users | strongest validation | time, consent, small sample | optional mini-pilot with 10–20 dishes in January |
| 7 | Two photos / reference object (coin, card) | proven effect | a different direction than questions | out of scope, mentioned in the discussion |

## Question pool (candidates)

| ID | Question | Answer type | Answer source in Nutrition5k |
|---|---|---|---|
| Q1 | How much did the meal weigh (g)? | number | `total_mass` |
| Q2 | Was the portion small / medium / large? | 3 categories | `total_mass` terciles |
| Q3 | Was the dish fried or did it have oil/butter added? | yes/no | fat ingredients in the list (oil, butter…) |
| Q4 | Was there a sauce or dressing? | yes/no | sauce ingredients in the list |
| Q5 | What is the main ingredient? | text | ingredient with the largest mass |
| Q6 | List the ingredients | list | ingredient list |
| Q7 | How many separate items are on the plate? | number | number of ingredients (rounded/noisy) |

## Question-selection strategies (budget B ∈ {0, 1, 2, 3})

- **S0 — no questions** (image only).
- **S1 — random.**
- **S2 — fixed global order (greedy on validation):** the order of questions that on average reduce error the most. The equivalent of the "static" strategy in AFABench.
- **S3 — VLM agent:** the model sees the question pool and picks the next question itself [as in Bhatambarekar].
- **S4 — uncertainty-driven (ours):** for a given photo we predict which question will lower the error the most. Uncertainty is measured by the variance of K VLM samples and/or the model's stated confidence. A small selector (e.g. gradient boosting or regression on features: variance, dish type from the VLM, stated portion) trained on the validation set predicts the error reduction of each question. Greedy selection, step by step, conditioned on the answers so far.
- **Oracle:** the best question subset for a given dish, chosen using GT. An upper bound.

## Simulated user models

- **U0 ideal:** true values (as in prior literature).
- **U1 log-normal Gaussian for mass:** `m' = m · exp(ε)`, `ε ~ N(μ, σ)`. Calibrated on [Lucassen 2021]: about 50% of answers within ±25%, so σ ≈ 0.33. Grid σ ∈ {0.1, 0.33, 0.5}. A variant with a shift μ, e.g. +0.2 (grams over-estimated by about 22%).
- **U2 categorical:** small/medium/large and yes/no questions answered wrongly with probability p ∈ {0, 0.1, 0.2}. For small/medium/large a mistake only goes to the adjacent category.
- **U3 "don't know":** with probability q the user does not answer, and the budget is still spent.

## Hypotheses

- **H1 (budget, diminishing returns):** 2 well-chosen questions give ≥ 70% of the error (MAPE) reduction achieved with full information (all questions, U0).
- **H2 (strategy):** at budgets B = 1 and B = 2 strategy S4 has significantly lower MAPE than S1 and S2 (Wilcoxon test, α = 0.05) and is not worse than S3.
- **H3 (realistic answers):** with realistic noise (U1, σ ≈ 0.33) the gain from the grams question (Q1) drops by ≥ 30% relative to U0. Categorical and binary questions (Q2, Q3) at a budget of 1–2 then give a gain comparable to or larger than Q1, i.e. the question ranking changes relative to the ideal user.
- **H4 (hidden fats, auxiliary):** the fat question Q3 reduces error the most for dishes where fat provides > 10% of calories.

The thresholds (70%, 30%) are preliminary. They were refined after the pilot: see `04_experiment_plan.md`, "Decisions after the F3 pilot" (H1 is measured against the oracle, H4 uses a 30% fat threshold).
