# 01 — Research gap and topic

## Topic (working title)

*How Many Questions Are Enough? Uncertainty-Driven, Budget-Constrained Clarification with Noisy User Answers for Image-Based Calorie Estimation*

## Problem

Estimating calories from a single photo carries error that the image alone cannot remove:
- **Portion (mass/volume).** A 2D photo does not capture the third dimension. This is the main source of error: [Fridolfsson 2025], [Nutrients 2026], [Vinod & Zhu 2026].
- **Hidden ingredients**: oil, butter, sauces and dressings. They add many calories and are often invisible [Bhatambarekar 2025].
- The best RGB models on Nutrition5k reach about 14–26% calorie error (PMAE) [Thames 2021; Han 2023]. General-purpose VLMs in zero-shot mode are off by 35–55% (MAPE) [Fridolfsson 2025; Bhatambarekar 2025].

Additional information from the user clearly helps. In Bhatambarekar's work, providing ingredient masses lowered GPT-4o's MAPE from about 51% to about 29%. Contextual metadata also reduces error [Coburn 2025]. The dish name alone, however, gives little: −1 kcal MAE [Narang 2025].

## Prior work

| Work | Interaction | Number of questions | How answers are modelled |
|---|---|---|---|
| Bhatambarekar 2025 (VT thesis) | "cue ladder": ingredients → total mass → ingredient masses | fixed ladder | **exact dataset values** (ground truth) |
| Bhatambarekar et al., CVPRW 2026 | fixed sequence of 6 questions vs. an agent asking its own questions | 6 / unlimited | answers from GT |
| Coburn et al. 2025 (ACETADA) | metadata: location, meal time, list of dishes | no dialogue | true metadata |
| CaLoRAify 2024, Food-R1 2026 | conversation with a VLM | no budget | — |
| Narang 2025 | dish name | 1 cue | true |

## Research gap

1. **No question budget.** Earlier work either asks everything (a ladder, 6 questions) or lets an agent ask without limit. Nobody measures **how accuracy depends on the number of questions (0–3)**, or which question to ask **when only one or two are allowed**. In an app every question adds to user burden, and lower burden usually means lower accuracy [Höchsmann & Martin 2020].
2. **Unrealistic answers.** Interactive VLMs are given the *true* mass in grams in experiments. People estimate portions poorly: only about 35–50% of answers fall within ±25% of the true mass [Lucassen 2021], and grams are over-estimated by about 22% on average (small sample, thesis). It is unknown **how much of the reported gain survives realistic, noisy or coarse answers**, and **whether the ranking of the most valuable questions changes**.
3. **Active feature acquisition** (AFA) and **question selection maximising expected information gain** (EIG) for LLMs [Aronsson 2025; Kobalczyk 2025] **have not been applied** to image-based calorie estimation.

We searched for work combining a question budget, noisy answers and VLM calorie estimation and found none (as of 28.10.2026).

## Research questions

- **RQ1:** How does calorie-estimation error depend on the question budget (0, 1, 2, 3)? How much of the "full information" gain do 1–2 questions deliver?
- **RQ2:** Does uncertainty-driven question selection (adaptive, image-dependent) achieve lower error than a fixed order, random choice and a free-asking VLM agent at the same budget?
- **RQ3:** How do noise and coarseness of user answers (grams with error, small/medium/large categories, yes/no questions) change the gain from interaction and the ranking of the most valuable questions?

## Contributions

1. An evaluation protocol for interactive calorie estimation **with a question budget and a simulated user**, with noise models calibrated on data about human portion estimation.
2. **Error vs. number of questions** curves for an open VLM on Nutrition5k and a comparison of question-selection strategies.
3. An analysis of **which questions are worth asking a real user** (e.g. "was there oil?" instead of "how many grams?"), with design recommendations for apps.

## Authors

Adam Sokołowski, Antoni Grajek

## Base paper

Lo, Sun, Qiu & Lo (2020), *Image-Based Food Classification and Volume Estimation for Dietary Assessment: A Review*, IEEE JBHI [lo2020review]. A review of food classification and volume estimation showing that volume/portion is the hardest stage. This directly motivates our portion questions.
