# Plan realizacji projektu PBAD — „Ile pytań wystarczy?”

## Kontekst

Temat, luka, hipotezy H1–H4 i plan eksperymentów E0–E6 są już gotowe (`01`–`04_*.md`, `references.bib`, prezentacja 09.11). Teraz (15.11, faza F2) trzeba to **zaimplementować**: pipeline Nutrition5k → symulowany użytkownik → VLM → strategie wyboru pytań → ewaluacja i wykresy. Zakres tej realizacji: **cały kod F2–F4**; długie obliczenia GPU odpalasz Ty.

Ustalenia ze środowiska:
- Projekt przenosimy do **WSL Ubuntu: `~/studia/pbad`** (`~/studia` istnieje, obok `amo`).
- WSL: GPU **RTX 3070 Laptop 8 GB** widoczne, 15 GB RAM, 945 GB wolnego, `uv` i `git` są, systemowy Python 3.14 (użyjemy venv z Pythonem 3.12 przez `uv`), brak `gsutil`, brak Claude Code CLI.
- **8 GB VRAM ≠ Qwen2.5-VL-7B bf16** z planu. Zmiana: model wybierany w configu, w pilocie testujemy **Qwen2.5-VL-3B-Instruct-AWQ** i **Qwen2.5-VL-7B-Instruct-AWQ** w vLLM z ograniczoną rozdzielczością obrazu (`max_pixels`), zostaje ten, który się mieści i ma sensowny MAPE. Drugi VLM (E5): mały model innej rodziny (np. InternVL-2B/4B). Jeśli kiedyś będzie dostęp do większego GPU — ten sam kod, inny config. Zmianę opisujemy w `04` i w Ograniczeniach.

## Krok 0 — migracja do WSL (pierwsza rzecz po akceptacji)

1. `cp -r "/mnt/c/Users/adams/OneDrive/Pulpit/STUDIA/mgr/mgr sem2/pbad" ~/studia/pbad` (oryginał na Windowsie zostaje jako kopia).
2. `git init` w `~/studia/pbad`, `.gitignore` (`data/`, `outputs/cache/`, `.venv/`, `*.pptx~`).
3. Skopiować ten plan do repo jako `PLAN.md` + krótki `CLAUDE.md` (kontekst projektu, struktura, komendy, zasada: bez co-autora Claude w commitach).
4. Dać Ci instrukcje uruchomienia kolejnej sesji (sekcja na końcu).

## Struktura repo

```
pbad/
  docs/                01..04_*.md, references.bib, prezentacja (przeniesione)
  PLAN.md, CLAUDE.md, README.md, pyproject.toml (uv, Python 3.12)
  configs/             default.yaml (model, max_pixels, pula pytań, siatka szumu, ścieżki, seed)
  src/pbad/
    data/download.py   pobieranie N5k przez HTTP z publicznego bucketu GCS
    data/build.py      dishes.parquet: GT (kcal, masa, makro, składniki z gramaturą) + odpowiedzi Q1–Q7
    data/lexicon.py    ręczne listy: tłuszcze (olej, masło…), sosy/dressingi
    questions.py       pula Q1–Q7: tekst pytania, typ, funkcja GT→odpowiedź, formatowanie faktu do promptu
    simulator.py       użytkownicy U0–U3 (log-normal σ/μ, pomyłki kategorii p, „nie wiem” q), seedowane
    vlm/prompt.py      prompt (obraz + lista faktów), wyjście JSON {kcal, mass_g, confidence}
    vlm/client.py      klient vLLM (OpenAI-compatible server), retry, parser JSON, clamp zakresu
    precompute.py      dla każdego dania × podzbiór ≤3 (64) × wariant szumu → predykcja; cache wznawialny (parquet/JSONL)
    uncertainty.py     K=5 próbek S0 (temperature>0) → wariancja, deklarowana pewność
    strategies.py      S0 brak, S1 losowa, S2 statyczna greedy (walidacja), S4 selektor niepewności (LightGBM), oracle
    agent.py           S3: online pętla — VLM sam wybiera kolejne pytanie z puli przy budżecie B
    evaluate.py        MAE, MAPE, acc@20%, „odzyskany zysk”, krzywe błąd vs B
    stats.py           Wilcoxon par + bootstrap 95% CI + korekta Holma
    plots.py           główna figura (błąd vs budżet per strategia/szum), ranking pytań, podgrupy
  scripts/             cienkie CLI: 01_download.sh, 02_build.py, 03_pilot.py, 04_precompute.py, 05_eval.py, 06_agent.py
  tests/               pytest: symulator (U1 σ=0.33 → ~50% w ±25%), mapowanie Q1–Q7, parser JSON, strategie na sztucznych danych
  outputs/             wyniki, tabele, figury
```

## Etapy (zgodnie z fazami z `04`)

**F2 — dane i narzędzia (do 23.11)**
1. Środowisko: `uv venv -p 3.12`, zależności (pandas, pyarrow, numpy, scipy, scikit-learn, lightgbm, matplotlib, pyyaml, openai, requests, pytest; vllm osobno).
2. Download: tylko `metadata/*` (dish_metadata_cafe1/2, ingredients_metadata), `dish_ids/splits/rgb_{train,test}_ids.txt` i `imagery/realsense_overhead/dish_*/rgb.png` (bez depth/wideo; szac. ~2 GB). Najpierw lista plików + rozmiar, potem pobranie równoległe ze wznawianiem.
3. `dishes.parquet`: parsowanie CSV N5k (zmienna liczba kolumn składników), tylko dania z obrazem overhead; podział: ~300 walidacja z train (seed), test = oficjalny test.
4. Q1–Q7 + leksykon tłuszczy/sosów + tercyle masy (z walidacji).
5. Symulator U0–U3 + testy.
6. vLLM w WSL (serwer OpenAI-compatible), prompt, parser; **demo end-to-end na 10 daniach** (S0 vs pełna informacja).

**F3 — pilot / go-no-go**
7. `03_pilot.py`: 50 dań walidacji, 2 modele × (z CoT / bez CoT); raport: % błędów parsowania (<5%), MAPE S0 (30–60%), zysk pełnej informacji, czas/danie → ekstrapolacja kosztu precompute. Zamrożenie protokołu w `configs/frozen.yaml`.
8. Redukcja kosztu, jeśli trzeba: szumimy tylko pytania obecne w podzbiorze; pełna siatka σ/p tylko dla B≤2; prefix caching obrazu.

**F4 — eksperymenty główne**
9. Precompute walidacja + test (wznawialny, odpalany przez Ciebie w tle, np. `nohup`/`tmux`).
10. S2 (kolejność greedy z walidacji), S4 (cechy: wariancja K próbek, confidence, mass_g z S0, typ dania; LightGBM przewiduje redukcję błędu per pytanie; greedy krok po kroku z warunkowaniem na dotychczasowych odpowiedziach), oracle — wszystkie offline z cache.
11. S3 agent online (B=1..3, U0 i główny wariant szumu).
12. Ewaluacja E1–E4, testy statystyczne, figury i tabele w `outputs/` → rozstrzygnięcie H1–H4.
(E5 drugi VLM = ten sam pipeline z innym configiem; E6 mini-pilot z ludźmi poza zakresem kodu.)

## Weryfikacja

- `pytest` zielony (symulator, mapowanie, parser, strategie na syntetycznych danych z znanym optimum).
- Smoke: `scripts/03_pilot.py --n 10` daje predykcje dla wszystkich 10 dań, S0 i pełnej informacji, z sensownym MAPE.
- Ewaluacja offline na małym cache (np. 20 dań) produkuje krzywe i tabelę bez błędów, zanim odpalisz pełny precompute.

## Instrukcje na następną sesję (podam też na końcu wykonania)

W terminalu WSL (Ubuntu):
```bash
curl -fsSL https://claude.ai/install.sh | bash
cd ~/studia/pbad && claude
```
Na start sesji: „Realizujemy PLAN.md, zacznij od F2”. `CLAUDE.md` w repo da jej pełny kontekst.
