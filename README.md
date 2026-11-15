# PBAD — „Ile pytań wystarczy?”

Wybór pytań do użytkownika sterowany niepewnością (budżet 0–3 pytań, zaszumione odpowiedzi)
w estymacji kalorii posiłku ze zdjęcia przez VLM, na Nutrition5k. Kontekst: `docs/01`–`04`, plan: `PLAN.md`.

## Instalacja (WSL Ubuntu, GPU 8 GB)

```bash
uv sync --extra vlm      # Python 3.12 venv + vLLM
uv run pytest -q         # testy: symulator, mapowanie Q1–Q7, parser, strategie
```

## Pipeline

| Krok | Komenda | Czas |
|---|---|---|
| 1. Dane (1,46 GB, wznawialne) | `scripts/01_download.sh` | ~3 min |
| 2. `data/dishes.parquet` | `uv run python scripts/02_build.py` | sekundy |
| 3. Serwer VLM (osobny terminal/tmux) | `scripts/serve_vllm.sh Qwen/Qwen2.5-VL-7B-Instruct-AWQ` | ~2 min startu |
| 4. Demo / pilot F3 | `uv run python scripts/03_pilot.py --n 50 --cot-mode both` | ~15 min |
| 5. Precompute (w tle) | `nohup uv run python scripts/04_precompute.py > outputs/precompute.log 2>&1 &` | patrz pilot |
| 6. Agent S3 (w tle) | `nohup uv run python scripts/06_agent.py --split test > outputs/agent.log 2>&1 &` | ~1–2 h |
| 7. Ewaluacja + wykresy | `uv run python scripts/05_eval.py` | minuty |

Wszystko czyta `configs/default.yaml`; inny config: `--config configs/x.yaml` (nakładany na default),
inny model: `--model ...` (musi zgadzać się z serwowanym). `--n N` ogranicza liczbę dań (smoke testy).
`04_precompute.py --dry-run` liczy unikalne prompty bez wywołań VLM.

**Wznawianie:** każde wywołanie VLM trafia do `outputs/cache/vlm_<model>.jsonl` (klucz = hash promptu,
obrazu, temperatury). Przerwany precompute/agent wystarczy uruchomić ponownie. Identyczne prompty
(np. szum, który nie zmienił odpowiedzi, albo „nie wiem” redukujące się do mniejszego podzbioru)
liczone są raz — pełny protokół to ~116 tys. unikalnych promptów zamiast ~301 tys. komórek.

## Wyniki

`outputs/results/<model>/`: `summary_test.csv` (MAE, MAPE, acc@20%, odzyskany zysk per strategia ×
szum × B), `tests_test.csv` (Wilcoxon + bootstrap CI + Holm, H2), `question_ranking_test.csv` (E3/E4),
`hypotheses_test.md` (automatyczne sprawdzenie H1–H4), `fig_*.png`, `per_dish_test.parquet`.
Pilot: `outputs/pilot/*.json`.

## Kod (`src/pbad/`)

`data/` pobieranie i budowa tabeli dań · `questions.py` pula Q1–Q7 · `simulator.py` użytkownicy U0–U3 ·
`vlm/` prompt, parser, klient z cache · `precompute.py` · `uncertainty.py` · `strategies.py` S0–S2, S4,
oracle · `agent.py` S3 · `evaluate.py`, `stats.py`, `plots.py`.
