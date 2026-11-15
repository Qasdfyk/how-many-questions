# 02 — Przegląd literatury (synteza)

Klucze w nawiasach `[...]` odpowiadają wpisom w `references.bib`. Oznaczenie **[do weryfikacji]** znaczy, że nie dotarliśmy do pełnego tekstu i liczby pochodzą z abstraktu albo z cytowań w innych pracach. Przed wpisaniem ich do artykułu trzeba sprawdzić oryginał.

## A. Klasyczna estymacja kalorii ze zdjęcia (CNN / ViT)

- **Im2Calories** [meyers2015im2calories]: pierwszy pełny pipeline rozpoznawanie → segmentacja → objętość z głębi → kalorie. Pokazał, że problem jest wieloetapowy, a porcja to najtrudniejszy etap.
- **Przegląd Lo et al. 2020** [lo2020review] (IEEE JBHI, praca bazowa grupy): przegląd metod klasyfikacji żywności (cechy ręczne vs. deep learning) i estymacji objętości ze zdjęć na potrzeby oceny diety. Estymacja objętości/porcji jest najsłabszym ogniwem całego pipeline'u.
- **Nutrition5k** [thames2021nutrition5k]: zbiór ok. 5 tys. realnych dań z kafeterii z ważonymi składnikami (masa dania i masa każdego składnika) oraz zdjęciami RGB-D z góry. Bazowy model RGB osiąga ok. 26,1% MAE kalorii (70,6 kcal). Z głębią błąd spada do 18,8%, a pipeline porcja→gęstość do 16,5%. **To nasz zbiór główny**, bo zawiera wszystko, czego potrzebuje symulowany użytkownik: masę całkowitą, listę składników z gramaturą i obecność tłuszczu.
- **DPF-Nutrition** [han2023dpf]: przewiduje głębię z pojedynczego zdjęcia i łączy ją z RGB. Na Nutrition5k ma 14,7% PMAE kalorii przy samym obrazie monokularnym.
- **OmniFood8K** [yu2026omnifood8k] (CVPR 2026): nowy zbiór dań chińskich oraz NutritionSynth-115K. Model z predykcją głębi i fuzją w dziedzinie częstotliwości to obecny SOTA dla pojedynczego zdjęcia.
- **Pre-training a estymacja** [andrade2025pretraining]: ViT pretrenowane na publicznych zbiorach (ImageNet, COYO) wyraźnie przegrywają z bazą Nutrition5k pretrenowaną na prywatnym JFT-300M (ResNet-50 ok. 117 kcal MAE). Wniosek dla nas: samodzielny fine-tune regresora nie osiągnie SOTA, więc lepiej wykorzystać VLM i interakcję.
- **ViT do wartości odżywczych** [banerjee2024vit] (literatura od prowadzącego): fine-tune ViT z opcjonalną fuzją z danymi przepisu. Wg streszczenia 92,3% trafności klasyfikacji na Food-101 i MAE 8,5 kcal **[do weryfikacji]**. Ta MAE jest nieporównywalna z Nutrition5k (inna definicja zadania).
- **Przegląd estymacji porcji** [vinod2026portion]: metody z głębią, wieloma widokami i szablonami 3D oraz uczenie głębokie z pojedynczego obrazu z wejściami pomocniczymi. Porcja pozostaje otwartym problemem.

## B. Duże modele multimodalne (VLM/LMM) w estymacji kalorii

- **CalorieLLaVA** [tanabe2024calorielava] **[do weryfikacji]**: LLaVA fine-tunowana na Nutrition5k. Wg cytowań ok. 64 kcal MAE wobec ok. 107 kcal (54,8% MAPE) dla zero-shot GPT-4V.
- **CalorieVoL** [tanabe2025calorievol] **[do weryfikacji]**: dodaje do MLLM informację o objętości. Autorzy pokazują, że VLM słabo rozpoznają objętość, co bezpośrednio przekłada się na błąd kalorii.
- **CaLoRAify** [yao2024caloraify]: zbiór CalData (330 tys. par obraz–tekst), VLM z LoRA i RAG, zachowuje możliwość rozmowy. Praca projektowa, bez budżetu pytań.
- **Food-R1** [zhu2026foodr1]: VLM trenowany z chain-of-thought i RL (GRPO) na CalorieBench-80K. Pokazuje trend „rozumowania” w estymacji kalorii.
- **Benchmark ACETADA** [coburn2025lmm] (IEEE BHI 2025): 8 LMM (4 otwarte, 4 zamknięte). **Metadane kontekstowe** (miejsce z GPS, pora posiłku, lista potraw) istotnie obniżają MAE/MAPE i wzmacniają efekt technik promptowania (CoT, few-shot, scale hint). Najbliższy dowód, że *dodatkowa informacja tekstowa pomaga VLM*.
- **Ocena GPT-4o / Claude / Gemini** [fridolfsson2025llm]: 52 standaryzowane zdjęcia. MAPE energii ok. 36% dla GPT-4o i Claude, Gemini znacznie gorzej. **Wszystkie modele zaniżają** wartości, i to tym mocniej, im większa porcja. Wniosek autorów: modele nie nadają się jeszcze do precyzyjnej oceny diety.
- **Prompt engineering i wybór modelu** [nutrients2026prompt] **[do weryfikacji: autorzy]**: główne źródło błędu to porcja, nie skład. Mocniejszy model częściej przyznaje się do niepewności (zwraca `null`), słabszy podaje pewne liczby. To argument za pytaniem użytkownika *wtedy, gdy model jest niepewny*.

## C. Interakcja z użytkownikiem i informacja tekstowa

- **Bhatambarekar, praca mgr (VT 2025)** [bhatambarekar2025thesis]: 8 MLLM zero-shot na Nutrition5k i MetaFood3D, z „drabinką wskazówek” zdjęcie → lista składników → masa całkowita → masa składników. MAPE kalorii GPT-4o: ok. 51% → ok. 29%. **Masa całkowita to najsilniejsza pojedyncza wskazówka.** Drugi widok daje +3–7 pp. Agent, który sam zadaje pytania, wygrywa ze stałą sekwencją, zwłaszcza przy małym kontekście (> 8 pp dla części modeli). Błędy zostają przy ukrytych składnikach (olej, dressing).
- **Bhatambarekar et al., CVPRW 2026** [bhatambarekar2026agentic]: wersja konferencyjna. Strukturalny CoT ze stałym łańcuchem 6 pytań kontra agent, który zadaje pytania na podstawie własnej niepewności. Interakcja istotnie zmniejsza błąd. **Praca najbliższa naszej. Różnimy się budżetem pytań, realistycznym szumem odpowiedzi i jawnym kryterium wyboru pytań.**
- **Narang 2025** [narang2025multimodal]: CNN obraz + nazwa dania na Nutrition5k. MAE spada tylko z 84,76 do 83,70 kcal. *Nie każda informacja tekstowa jest warta pytania*, co wspiera potrzebę wyboru pytań.

## D. Wybór pytań: active feature acquisition i pytania doprecyzowujące w LLM

- **Survey AFA** [aronsson2025afa]: AFA to sekwencyjne pozyskiwanie cech per instancja przy koszcie, sformułowane jako POMDP. Taksonomia: predyktory kosztowe, planowanie modelowe, RL, metody hybrydowe. **Nasze ramy formalne:** pytanie to cecha z kosztem, odpowiedź to zaszumiona obserwacja.
- **AFABench** [schutz2025afabench]: benchmark AFA. Strategie statyczne, krótkowzroczne (*myopic*, greedy) i RL. Daje nam gotowe baseline'y (statyczna kolejność, greedy).
- **Zero-shot AFA z LLM** [perets2026zeroshot]: LLM jako źródło wiedzy dziedzinowej, nie jako decydent, bo LLM słabo planuje sekwencyjnie. Uzasadnia oddzielenie modułu wyboru pytań od VLM.
- **Active Task Disambiguation** [kobalczyk2025active] (ICLR 2025): wybór pytań przez LLM jako bayesowskie planowanie eksperymentu, gdzie pytanie maksymalizuje oczekiwany zysk informacji (EIG). Wygrywa z niejawnym rozumowaniem.
- **Clarification z EIG w agentach** [infogain2026clarification] **[do weryfikacji: autorzy]**: agent pyta tylko wtedy, gdy oczekiwana redukcja niepewności to uzasadnia.
- **Modelowanie przyszłych tur** [zhang2024clarifying]: LLM rzadko same z siebie zadają pytania doprecyzowujące. Trenowanie na symulowanych skutkach pytania to poprawia.

## E. Człowiek jako źródło informacji: błąd szacowania porcji

- **Lucassen et al. 2021** [lucassen2021portion]: 40 osób, jeden lunch. W ±25% prawdziwej masy mieści się 50% odpowiedzi przy opisach tekstowych porcji i tylko 35% przy wyborze zdjęć porcji. Szacowanie w gramach jest zwykle niedokładne. **Na tej pracy skalibrujemy model szumu odpowiedzi „masa”.**
- **Höchsmann & Martin 2020** [hochsmann2020review]: przegląd metod oceny diety ze zdjęć. Zaniżanie (*underreporting*) się utrzymuje, a mniej uciążliwe metody są mniej dokładne. „Żadna metoda nie jest adekwatna we wszystkich warunkach”. Kompromis dokładność–obciążenie to **motywacja budżetu pytań**.
- Uzupełniająco: ludzie mają tendencję do przeszacowania produktów niskokalorycznych i niedoszacowania wysokokalorycznych (badanie 2024, 35 osób) **[do znalezienia pełna referencja]**.

## Tabela porównawcza

| Praca | Dane | Model | Wejście | Pytania do użytkownika | Budżet | Szum odpowiedzi | Wynik kcal |
|---|---|---|---|---|---|---|---|
| Thames 2021 | N5k | CNN (JFT) | RGB(-D) | — | — | — | 26,1% / 16,5% |
| Han 2023 (DPF) | N5k | CNN + depth pred. | RGB | — | — | — | 14,7% |
| Tanabe 2024 | N5k | LLaVA FT | RGB | — | — | — | ok. 64 kcal* |
| Coburn 2025 | ACETADA | 8 LMM | RGB + metadane | — (metadane) | — | brak | ↓ MAE |
| Fridolfsson 2025 | 52 zdj. | GPT-4o/Claude/Gemini | RGB | — | — | — | ok. 36% MAPE |
| Narang 2025 | N5k | CNN | RGB + nazwa | 1 stała | — | brak | 83,7 kcal |
| Bhatambarekar 2025/26 | N5k, MetaFood3D | 8 MLLM | RGB + wskazówki | drabinka / 6 / agent | **brak** | **brak (GT)** | 51% → 29% MAPE |
| **Nasza praca** | N5k | Qwen2.5-VL (+drugi VLM) | RGB + odpowiedzi | pula 7 pytań | **0–3** | **idealne / gaussowski / kategoryczny / „nie wiem”** | krzywa błąd vs. budżet |

\* [do weryfikacji]

## Wnioski z przeglądu, czyli skąd hipotezy

1. Porcja i ukryte tłuszcze odpowiadają za większość błędu, a sam obraz tego nie rozwiąże.
2. Informacja od użytkownika pomaga, ale tylko odpowiednio dobrana (masa tak, nazwa dania prawie nic).
3. Prace interaktywne zakładają idealnego użytkownika i nieograniczoną liczbę pytań. Literatura dietetyczna mówi, że ludzie odpowiadają z dużym błędem, a literatura o obciążeniu użytkownika, że każde pytanie kosztuje.
4. AFA/EIG to gotowy aparat formalny do wyboru pytań przy koszcie, którego w tej domenie jeszcze nie użyto.
