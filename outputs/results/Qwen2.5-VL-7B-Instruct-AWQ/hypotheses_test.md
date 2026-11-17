# Automatic hypothesis check (thresholds from docs/03)

## H1 — 2 well-chosen questions capture most of the achievable gain (U0)
- photo only 52.8% vs all 7 questions 42.2% (all questions help)
- S2, B=2: MAPE 44.1%, share of oracle gain 0.31, recovered vs all-7 0.82
- S3, B=2: MAPE 47.5%, share of oracle gain 0.19, recovered vs all-7 0.50
- S4, B=2: MAPE 43.9%, share of oracle gain 0.32, recovered vs all-7 0.84
- oracle, B=2: MAPE 24.6%, share of oracle gain 1.00, recovered vs all-7 2.66

## H2 — see tests_*.csv (S4 vs S1/S2/S3, Wilcoxon + Holm)

## H3 — Q1 gain drop under U1_s033 vs U0 (threshold 30%)
- Q1 gain: U0 5.9 pp, U1_s033 3.7 pp, drop 37%
- Q2 gain under U1_s033: 6.5 pp (vs Q1 3.7 pp)
- Q3 gain under U1_s033: -0.4 pp (vs Q1 3.7 pp)
- ranking U0: ['Q2', 'Q1', 'Q4', 'Q7', 'Q5', 'Q3', 'Q6']
- ranking U1_s033: ['Q2', 'Q1', 'Q4', 'Q7', 'Q5', 'Q3', 'Q6']

## H4 — Q3 is the best single question for fat-heavy dishes (>30% kcal from fat)
- fat_share<=30% (n=127): best Q2 (9.3 pp), Q3 6.2 pp
- fat_share>30% (n=297): best Q2 (5.6 pp), Q3 -3.2 pp
