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
- Repo jest **publiczne** na GitHubie — nic prywatnego (dane, klucze, maile) do gita.

## Wykresy — standard projektu

Każdy wykres (artykuł, slajdy, raporty) robimy przez `src/pbad/plotstyle.py` (`ps.apply()`, `ps.header()`,
`ps.save()`), w stylu raportów badawczych: ciepłe złamane białe tło, bez ramki, tylko cienka linia bazowa
i cienka pozioma siatka (ciągła, nigdy przerywana), cienkie linie 2 px, markery z obwódką w kolorze tła.
- Tytuł do lewej + jednozdaniowy podtytuł z wnioskiem. Tekst zawsze w kolorach atramentu, nigdy w kolorze serii.
- Max 3 kolorowe serie na panel (paleta zwalidowana pod daltonizm), reszta w szarości; kolor należy do bytu
  (S4 zawsze niebieski, S2 pomarańczowy, S3 turkusowy, S1 szary, oracle szary przerywany).
- Legenda przy ≥ 2 seriach, etykiety bezpośrednie oszczędnie (koniec linii, seria, o której jest historia).
- Jedna oś Y; zamiast wielu serii — small multiples. Porównania przed/po — dumbbell.
- `ps.save()` zapisuje PNG (slajdy) + PDF (artykuł) + CSV z danymi wykresu. Po wygenerowaniu obejrzeć PNG
  (kolizje etykiet, ucięcia).

## Stan i komendy (po sesji 15.11)

- F2 zrobione: kod całego pipeline'u F2–F4 w `src/pbad/`, skrypty `scripts/01–06`, `pytest` (23 testy).
  Szczegóły i komendy: `README.md`. Zmiany względem planu: sekcja „Zmiany po F2” w `docs/04`.
- Model: **Qwen2.5-VL-7B-Instruct-AWQ** z `--cpu-offload-gb 3` (preset w `scripts/serve_vllm.sh`), ~1,2 s/wywołanie.
  3B-AWQ odrzucony w demo. vLLM 0.33 wymaga `VLLM_USE_FLASHINFER_SAMPLER=0` (brak nvcc w WSL; ustawione w skrypcie).
- Decyzje po pilocie F3 (min_kcal 50, bez CoT, H1 względem oracle, H4 próg 30%): `docs/04`, sekcja „Decyzje po pilocie F3”.
- Zatrzymywanie serwera: `pkill -f "[v]llm serve"` (nawiasy, żeby pkill nie zabił własnej powłoki).
- Cache VLM: `outputs/cache/vlm_<model>.jsonl` — nie kasować, wszystko jest z niego wznawiane.
