# 02 — Literature review (synthesis)

Keys in brackets `[...]` correspond to entries in `references.bib`. **[to verify]** means we did not reach the full text and the numbers come from the abstract or from citations in other papers. They must be checked against the original before going into the paper.

## A. Classic image-based calorie estimation (CNN / ViT)

- **Im2Calories** [meyers2015im2calories]: the first full pipeline recognition → segmentation → volume from depth → calories. Showed that the problem is multi-stage and that portion is the hardest stage.
- **Review by Lo et al. 2020** [lo2020review] (IEEE JBHI, the group's base paper): a review of food classification methods (hand-crafted features vs. deep learning) and image-based volume estimation for dietary assessment. Volume/portion estimation is the weakest link of the whole pipeline.
- **Nutrition5k** [thames2021nutrition5k]: about 5k real cafeteria dishes with weighed ingredients (dish mass and the mass of each ingredient) and overhead RGB-D images. The RGB baseline reaches about 26.1% calorie MAE (70.6 kcal). With depth the error drops to 18.8%, and a portion→density pipeline reaches 16.5%. **This is our main dataset**, because it contains everything a simulated user needs: total mass, the ingredient list with grams, and the presence of fat.
- **DPF-Nutrition** [han2023dpf]: predicts depth from a single photo and fuses it with RGB. 14.7% calorie PMAE on Nutrition5k from a monocular image only.
- **OmniFood8K** [yu2026omnifood8k] (CVPR 2026): a new dataset of Chinese dishes plus NutritionSynth-115K. The model with depth prediction and frequency-domain fusion is the current single-image SOTA.
- **Pre-training and estimation** [andrade2025pretraining]: ViTs pre-trained on public datasets (ImageNet, COYO) clearly lose to the Nutrition5k baseline pre-trained on the private JFT-300M (ResNet-50, about 117 kcal MAE). Takeaway for us: fine-tuning our own regressor will not reach SOTA, so it is better to use a VLM and interaction.
- **ViT for nutrient values** [banerjee2024vit] (literature from the instructor): ViT fine-tuning with optional fusion with recipe data. According to the abstract, 92.3% classification accuracy on Food-101 and an MAE of 8.5 kcal **[to verify]**. This MAE is not comparable with Nutrition5k (different task definition).
- **Review of portion estimation** [vinod2026portion]: methods using depth, multiple views and 3D templates, and deep learning from a single image with auxiliary inputs. Portion estimation remains an open problem.

## B. Large multimodal models (VLM/LMM) for calorie estimation

- **CalorieLLaVA** [tanabe2024calorielava] **[to verify]**: LLaVA fine-tuned on Nutrition5k. According to citations about 64 kcal MAE vs. about 107 kcal (54.8% MAPE) for zero-shot GPT-4V.
- **CalorieVoL** [tanabe2025calorievol] **[to verify]**: adds volume information to an MLLM. The authors show that VLMs recognise volume poorly, which translates directly into calorie error.
- **CaLoRAify** [yao2024caloraify]: the CalData dataset (330k image–text pairs), a VLM with LoRA and RAG that keeps conversational ability. A design paper, without a question budget.
- **Food-R1** [zhu2026foodr1]: a VLM trained with chain-of-thought and RL (GRPO) on CalorieBench-80K. Shows the "reasoning" trend in calorie estimation.
- **ACETADA benchmark** [coburn2025lmm] (IEEE BHI 2025): 8 LMMs (4 open, 4 closed). **Contextual metadata** (location from GPS, meal time, list of dishes) significantly lowers MAE/MAPE and strengthens the effect of prompting techniques (CoT, few-shot, scale hint). The closest evidence that *additional textual information helps VLMs*.
- **Evaluation of GPT-4o / Claude / Gemini** [fridolfsson2025llm]: 52 standardised photos. Energy MAPE about 36% for GPT-4o and Claude, Gemini much worse. **All models under-estimate**, the more so the larger the portion. The authors conclude that the models are not yet fit for precise dietary assessment.
- **Prompt engineering and model choice** [nutrients2026prompt] **[to verify: authors]**: the main source of error is portion, not composition. A stronger model more often admits uncertainty (returns `null`), a weaker one gives confident numbers. An argument for asking the user *when the model is uncertain*.

## C. User interaction and textual information

- **Bhatambarekar, MSc thesis (VT 2025)** [bhatambarekar2025thesis]: 8 zero-shot MLLMs on Nutrition5k and MetaFood3D, with a "cue ladder" photo → ingredient list → total mass → ingredient masses. GPT-4o calorie MAPE: about 51% → about 29%. **Total mass is the strongest single cue.** A second view adds 3–7 pp. An agent asking its own questions beats a fixed sequence, especially with a small context (> 8 pp for some models). Errors remain for hidden ingredients (oil, dressing).
- **Bhatambarekar et al., CVPRW 2026** [bhatambarekar2026agentic]: the conference version. Structured CoT with a fixed chain of 6 questions vs. an agent asking questions based on its own uncertainty. Interaction significantly reduces error. **The work closest to ours. We differ in the question budget, realistic answer noise and an explicit question-selection criterion.**
- **Narang 2025** [narang2025multimodal]: image CNN + dish name on Nutrition5k. MAE drops only from 84.76 to 83.70 kcal. *Not every piece of textual information is worth a question*, which supports the need for question selection.

## D. Question selection: active feature acquisition and clarifying questions in LLMs

- **AFA survey** [aronsson2025afa]: AFA is sequential per-instance feature acquisition under cost, formulated as a POMDP. Taxonomy: cost-aware predictors, model-based planning, RL, hybrid methods. **Our formal framework:** a question is a feature with a cost, the answer is a noisy observation.
- **AFABench** [schutz2025afabench]: an AFA benchmark with static, myopic (greedy) and RL strategies. Gives us ready baselines (static order, greedy).
- **Zero-shot AFA with LLMs** [perets2026zeroshot]: the LLM as a source of domain knowledge rather than a decision-maker, because LLMs plan sequentially poorly. Justifies separating the question-selection module from the VLM.
- **Active Task Disambiguation** [kobalczyk2025active] (ICLR 2025): question selection by an LLM as Bayesian experimental design, where the question maximises expected information gain (EIG). Beats implicit reasoning.
- **Clarification with EIG in agents** [infogain2026clarification] **[to verify: authors]**: the agent asks only when the expected uncertainty reduction justifies it.
- **Modelling future turns** [zhang2024clarifying]: LLMs rarely ask clarifying questions on their own. Training on simulated consequences of asking improves this.

## E. The human as an information source: portion-estimation error

- **Lucassen et al. 2021** [lucassen2021portion]: 40 people, one lunch. 50% of answers fall within ±25% of the true mass with textual portion descriptions and only 35% with portion photos. Estimating in grams is usually inaccurate. **We calibrate the noise model for the "mass" answer on this work.**
- **Höchsmann & Martin 2020** [hochsmann2020review]: a review of image-based dietary assessment. Under-reporting persists, and less burdensome methods are less accurate. "No method is adequate under all conditions". The accuracy–burden trade-off is the **motivation for a question budget**.
- Additionally: people tend to over-estimate low-calorie foods and under-estimate high-calorie ones (2024 study, 35 people) **[full reference to find]**.

## Comparison table

| Work | Data | Model | Input | User questions | Budget | Answer noise | Calorie result |
|---|---|---|---|---|---|---|---|
| Thames 2021 | N5k | CNN (JFT) | RGB(-D) | — | — | — | 26.1% / 16.5% |
| Han 2023 (DPF) | N5k | CNN + depth pred. | RGB | — | — | — | 14.7% |
| Tanabe 2024 | N5k | LLaVA FT | RGB | — | — | — | about 64 kcal* |
| Coburn 2025 | ACETADA | 8 LMMs | RGB + metadata | — (metadata) | — | none | ↓ MAE |
| Fridolfsson 2025 | 52 photos | GPT-4o/Claude/Gemini | RGB | — | — | — | about 36% MAPE |
| Narang 2025 | N5k | CNN | RGB + name | 1 fixed | — | none | 83.7 kcal |
| Bhatambarekar 2025/26 | N5k, MetaFood3D | 8 MLLMs | RGB + cues | ladder / 6 / agent | **none** | **none (GT)** | 51% → 29% MAPE |
| **Our work** | N5k | Qwen2.5-VL (+ second VLM) | RGB + answers | pool of 7 questions | **0–3** | **ideal / log-normal / categorical / "don't know"** | error vs. budget curve |

\* [to verify]

## Takeaways from the review (where the hypotheses come from)

1. Portion and hidden fats account for most of the error, and the image alone will not solve it.
2. Information from the user helps, but only if well chosen (mass yes, dish name almost nothing).
3. Interactive work assumes an ideal user and an unlimited number of questions. The dietetics literature says people answer with large error, and the user-burden literature says every question has a cost.
4. AFA/EIG is a ready formal apparatus for question selection under cost that has not yet been used in this domain.
