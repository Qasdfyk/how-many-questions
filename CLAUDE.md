# PBAD — „Ile pytań wystarczy?”

Projekt badawczy (mgr, sem. 2, przedmiot PBAD). Autorzy: Adam Sokołowski, Antoni Grajek.
Temat: wybór pytań do użytkownika sterowany niepewnością, przy budżecie 0–3 pytań i zaszumionych odpowiedziach, w estymacji kalorii posiłku ze zdjęcia przez VLM (Nutrition5k).

## Gdzie co jest

- `PLAN.md` — zatwierdzony plan realizacji (struktura repo, etapy F2–F4, weryfikacja). **Realizujemy go krok po kroku.**
- `docs/01_luka_i_temat.md` — temat, luka, RQ1–RQ3.
- `docs/02_przeglad_literatury.md` — przegląd literatury; klucze `[...]` → `docs/references.bib`.
- `docs/03_burza_mozgow_hipotezy.md` — pula pytań Q1–Q7, strategie S0–S4 + oracle, użytkownicy U0–U3, hipotezy H1–H4.
- `docs/04_plan_eksperymentow.md` — dane, modele, precompute 64 podzbiorów, eksperymenty E0–E6, metryki, fazy F1–F6 i terminy.

## Środowisko

- WSL Ubuntu, GPU RTX 3070 Laptop **8 GB VRAM**, 15 GB RAM. Brak `gsutil` (dane pobieramy przez HTTP z publicznego bucketu `nutrition5k_dataset`).
- Python: venv przez `uv` z Pythonem 3.12 (systemowy to 3.14 — nie używać). vLLM jako serwer OpenAI-compatible.
- Przez 8 GB VRAM modele: Qwen2.5-VL-3B/7B w wersji AWQ, ograniczone `max_pixels`; wybór w pilocie (F3), model w configu.
- Długie obliczenia (precompute, agent S3) odpala użytkownik w tle (`tmux`/`nohup`); kod musi być wznawialny (cache).

## Konwencje

- Dokumentacja i komunikacja po polsku; kod, nazwy i komentarze po angielsku.
- Wszystko seedowane i sterowane configiem (`configs/*.yaml`).
- `data/` i `outputs/cache/` nie trafiają do gita.
- Commity: **nigdy** nie dodawać Claude jako współautora ani linijek „Generated with Claude Code”.
