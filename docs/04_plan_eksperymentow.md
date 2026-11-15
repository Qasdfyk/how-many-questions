# 04 — Plan weryfikacji hipotez i eksperymentów

## Dane

- **Nutrition5k** [thames2021nutrition5k]: zdjęcia z góry (`realsense_overhead`, RGB) + `metadata` (kcal, masa, makro, składniki z gramaturą) + oficjalne podziały `dish_ids/splits`.
  - Pobieramy **tylko** zdjęcia z góry i metadane, bez filmów (cały zbiór ma ok. 180 GB; rozmiar podzbioru trzeba sprawdzić przed pobraniem).
  - Oficjalny train dzielimy na **walidację** (do strojenia selektora S4, ok. 300 dań) i resztę, której nie używamy, bo VLM działa zero-shot. Test to oficjalny podział testowy.
- **Kaggle „Recipe images with nutritional information”** (zbiór z opisu tematu): **odrzucony jako zbiór główny**, bo opisuje przepisy na porcję, bez prawdziwej masy na zdjęciu i bez gramatury składników, więc nie da się z niego symulować odpowiedzi użytkownika. Najwyżej jako test jakościowy lub przykład w dyskusji.
- (Opcjonalnie) **MetaFood3D**: test uogólnienia, jeśli wystarczy czasu.

## Modele

- Główny: **Qwen2.5-VL-7B-Instruct** (lokalne GPU, ok. 16–20 GB VRAM w bf16; przy mniejszym GPU wersja 4-bit lub 3B).
- Drugi (odporność wniosków): **InternVL2.5-8B** lub **LLaVA-OneVision-7B**.
- Serwowanie: **vLLM** (batching, prefix caching tego samego zdjęcia przy wielu wariantach pytań).
- Wyjście modelu: JSON `{"kcal": liczba, "mass_g": liczba, "confidence": 0-1}` w stałym promptcie (CoT krótki albo wyłączony, decyzja po pilocie).

## Kluczowy trik obliczeniowy: precompute podzbiorów

Odpowiedzi trafiają do promptu jako lista faktów, więc ich kolejność nie ma znaczenia. **Liczymy predykcję VLM raz dla każdego podzbioru pytań o rozmiarze ≤ 3** (z 7 pytań to 1 + 7 + 21 + 35 = 64 podzbiory) i dla każdego modelu szumu. Wtedy **wszystkie strategie (S0–S2, S4, oracle) ewaluujemy offline**, wybierając tylko, który podzbiór „zadałaby” dana strategia. Agent S3 wymaga osobnego przebiegu online.

Szacunek kosztu: ok. 500 dań testowych + 300 walidacyjnych, 64 podzbiory, ok. 5 wariantów szumu (U0, U1 σ = 0,33, U1 σ = 0,5, U2 p = 0,1, U3 q = 0,2) daje ok. 250 tys. krótkich generacji. Do tego K = 5 próbek dla S0 (niepewność). Przy vLLM i krótkim wyjściu to **rzędu 10–20 h GPU** [do sprawdzenia w pilocie]. Można zredukować: szumić tylko pytania zawarte w podzbiorze, a pełną siatkę σ i p liczyć tylko dla B ≤ 2.

## Eksperymenty

| ID | Cel | Konfiguracja | Hipoteza |
|---|---|---|---|
| E0 | Pilot | 50 dań walidacyjnych, S0 + pełna informacja, dobór promptu | — |
| E1 | Krzywa budżetu | S0–S4 + oracle, B = 0..3, U0 | H1, H2 |
| E2 | Realistyczny użytkownik | E1 dla U1 (siatka σ, bias), U2 (siatka p), U3 | H3 |
| E3 | Ranking pytań | zysk pojedynczego pytania (B = 1) dla każdego Qi, U0 vs U1/U2 | H3, H4 |
| E4 | Analiza podgrup | dania z tłuszczem > 10% kcal, duże/małe porcje, liczba składników | H4 |
| E5 | Drugi VLM | powtórzenie E1–E2 (B ≤ 2) | odporność wniosków |
| E6 (opcj.) | Mini-pilot z ludźmi | 10–20 osób, kilka zdjęć z N5k, prawdziwe odpowiedzi vs. symulacja | walidacja modelu szumu |

## Metryki i statystyka

- MAE [kcal], MAPE [%], odsetek predykcji w ±20% (*accuracy@20%*).
- Krzywa **błąd vs. budżet B** dla każdej strategii i modelu szumu (główna figura artykułu).
- „Odzyskany zysk” = (błąd S0 − błąd strategii) / (błąd S0 − błąd pełnej informacji).
- Istotność: test Wilcoxona par na błędach bezwzględnych per danie + bootstrapowe 95% CI. Korekta Holma przy wielu porównaniach.

## Narzędzia i szacunek czasu (pod zajęcia 23.11)

| Element | Narzędzie | Czas |
|---|---|---|
| Pobranie i przygotowanie N5k | `gsutil`, Python, pandas | 1–2 dni |
| Mapowanie składników na odpowiedzi Q1–Q7 (lista tłuszczy i sosów) | Python + ręczna lista | 1 dzień |
| Symulator użytkownika (U0–U3) | Python, numpy | 1 dzień |
| Serwowanie VLM, prompt, parsowanie JSON | vLLM, Qwen2.5-VL | 2–3 dni |
| Precompute podzbiorów | skrypt batch | 1–2 dni GPU |
| Selektor S4, ewaluacja strategii offline | scikit-learn / LightGBM | 2 dni |
| Agent S3 | pętla dialogowa z vLLM | 1–2 dni |
| Analiza, wykresy, testy | pandas, scipy, matplotlib | 2 dni |

## Fazy projektu

Każda faza ma swoje produkty i jawne kryterium zakończenia. Kolejna faza zaczyna się dopiero po spełnieniu kryterium poprzedniej. Po F3 jest punkt decyzyjny go/no-go.

### F1 — Definicja problemu (12.10–09.11) ✓
- **Cel:** wybrać lukę badawczą i zaplanować jej weryfikację.
- **Zadania:** przegląd literatury, identyfikacja luki, pytania RQ1–RQ3, hipotezy H1–H4, plan eksperymentów.
- **Produkty:** pliki `01`–`04`, `references.bib`, prezentacja 09.11.
- **Kryterium zakończenia:** prowadzący akceptuje temat i plan na zajęciach 09.11.

### F2 — Dane i narzędzia (10.11–23.11)
- **Cel:** działający pipeline od zdjęcia do szacunku kcal z odpowiedziami.
- **Zadania:**
  - pobranie Nutrition5k (`realsense_overhead` + `metadata`), podział walidacja (ok. 300 dań) / test;
  - mapowanie składników na odpowiedzi Q1–Q7 (lista tłuszczy, sosów, tercyle masy);
  - symulator użytkownika U0–U3 z testami jednostkowymi (np. czy rozkład błędu U1 daje ok. 50% w ±25%);
  - uruchomienie Qwen2.5-VL-7B w vLLM, prompt i parser JSON;
  - szkielet artykułu (IMRaD) na 16.11.
- **Produkty:** repozytorium z kodem, tabela dań `dishes.parquet` (GT + odpowiedzi), moduł symulatora, szkielet artykułu.
- **Kryterium zakończenia:** pipeline działa end-to-end na 10 daniach (zdjęcie → pytania → odpowiedzi → kcal); demo na zajęciach 23.11.

### F3 — Pilot i kalibracja (24.11–30.11)
- **Cel:** sprawdzić, czy metoda działa, zanim wydamy czas GPU na pełne eksperymenty.
- **Zadania:** E0 na 50 daniach walidacyjnych: wybór promptu (z CoT / bez CoT), pomiar czasu na danie i realnego kosztu GPU, ustalenie progów w H1 i H3, zamrożenie protokołu (prompt, pula pytań, siatka szumu, metryki).
- **Produkty:** krótki raport z pilota (1–2 strony), zamrożony protokół eksperymentów.
- **Kryterium zakończenia / go-no-go:**
  - < 5% błędów parsowania odpowiedzi VLM;
  - MAPE S0 w zakresie zgodnym z literaturą (ok. 30–60%);
  - pełna informacja (wszystkie pytania, U0) obniża MAPE względem S0, czyli jest efekt do badania;
  - koszt pełnego precompute ≤ 2 doby GPU.
- **Jeśli no-go:** zmiana promptu, model 3B lub 4-bit, mniejsza pula pytań (5 zamiast 7) albo mniejsza siatka szumu.

### F4 — Eksperymenty główne (01.12–20.12)
- **Cel:** odpowiedzieć na RQ1–RQ3 (hipotezy H1–H3).
- **Zadania:** precompute walidacja + test (64 podzbiory × warianty szumu), trening selektora S4 na walidacji, przebieg online agenta S3, ewaluacja E1–E3, testy statystyczne.
- **Produkty:** tabele wyników, krzywe błąd vs. budżet (główna figura), wyniki testów Wilcoxona z CI, sekcja *Wyniki* w pierwszym drafcie.
- **Kryterium zakończenia:** każda z H1–H3 ma rozstrzygnięcie (potwierdzona / odrzucona) na zbiorze testowym; draft z wynikami gotowy na naszą pierwszą dyskusję (30.11 / 07.12 / 14.12 — termin do ustalenia).

### F5 — Analiza rozszerzona i odporność (21.12–11.01)
- **Cel:** sprawdzić, czy wnioski są ogólne.
- **Zadania:** E4 podgrupy (H4: dania tłuste, małe/duże porcje), E5 powtórzenie E1–E2 na drugim VLM, opcjonalnie E6 mini-pilot z 10–20 osobami.
- **Produkty:** analiza podgrup, porównanie modeli, ew. porównanie prawdziwych odpowiedzi ludzi z symulacją.
- **Kryterium zakończenia:** wnioski potwierdzone na drugim modelu albo różnice opisane w sekcji *Dyskusja*; draft 2 gotowy na drugą dyskusję (21.12 / 11.01 / 18.01 — termin do ustalenia).

### F6 — Artykuł i prezentacja (12.01–25.01)
- **Cel:** domknąć publikację.
- **Zadania:** poprawki po drugiej dyskusji, *Dyskusja* i *Ograniczenia*, abstrakt, porządek w repozytorium (kod + wyniki do odtworzenia), slajdy końcowe.
- **Produkty:** gotowy artykuł, repozytorium z instrukcją odtworzenia wyników, prezentacja.
- **Kryterium zakończenia:** artykuł oddany i prezentacja wygłoszona 25.01.

### Podsumowanie faz

| Faza | Termin | Kamień milowy |
|---|---|---|
| F1 Definicja | 12.10–09.11 | prezentacja planu 09.11 ✓ |
| F2 Dane i narzędzia | 10.11–23.11 | pipeline end-to-end na 10 daniach |
| F3 Pilot | 24.11–30.11 | **go/no-go** + zamrożony protokół |
| F4 Eksperymenty | 01.12–20.12 | rozstrzygnięcie H1–H3, draft 1 |
| F5 Odporność | 21.12–11.01 | drugi VLM, H4, draft 2 |
| F6 Publikacja | 12.01–25.01 | artykuł + prezentacja 25.01 |

## Ryzyka

- **VLM nie trzyma formatu / zwraca absurdy** → walidacja JSON, retry, przycięcie do sensownego zakresu, zgłoszenie odsetka błędów.
- **Za mało GPU** → model 3B lub 4-bit, mniejsza siatka szumu, podzbiór testu.
- **Model szumu z literatury nie odpowiada rzeczywistości** → analiza wrażliwości na siatce σ i p (wnioski jako funkcja szumu), opcjonalnie mini-pilot E6.
- **S4 nie bije S2** → to też wynik: „statyczna kolejność wystarcza”, co upraszcza aplikację. Raportujemy uczciwie.

## Zmiany po F2 (15.11)

- **Model główny: Qwen2.5-VL-7B-Instruct-AWQ** zamiast 7B bf16 — mamy GPU 8 GB (RTX 3070 Laptop, WSL).
  Wagi AWQ zajmują 6,7 GiB, więc 3 GB wag trzymamy w RAM (`--cpu-offload-gb 3`, bez CUDA graphs).
  Kosztem jest przepustowość (~1 s/wywołanie przy współbieżności 8). Na większym GPU ten sam kod
  działa z innym configiem.
- **Qwen2.5-VL-3B-AWQ odrzucony** już w demo na 10 daniach: kotwiczy się na stałych wartościach
  (35/350 kcal), a z pełną informacją jest gorszy niż bez niej (MAPE 60% → 82%).
- **Deduplikacja promptów** zamiast ręcznej redukcji siatki: identyczne prompty liczymy raz.
  Pełny protokół (300 walidacja + 492 test, 8 wariantów szumu) to ~116 tys. unikalnych wywołań.
- **Q2 pod szumem U1:** kategoria porcji wynika z tej samej zaburzonej percepcji masy co Q1
  (spójny użytkownik). Bez tego H3 byłaby sprawdzana na korzyść Q2 z góry.
- **Kalibracja U1:** σ = 0,33 daje analitycznie 56% odpowiedzi w ±25% (dokładnie 50% wymagałoby σ ≈ 0,375).
- **Q2** pokazuje użytkownikowi przedziały masy (tercyle z walidacji: < 153 g, 153–288 g, > 288 g).

## Decyzje po pilocie F3 (15.11)

Pilot: 7B-AWQ, 50 dań walidacji. Bez CoT: 0% błędów parsowania, MAPE S0 85%, pełna informacja 51%,
~1,2 s/wywołanie. Z CoT: MAPE 89% / 68%, ~2,8 s/wywołanie. Po odrzuceniu dań < 50 kcal:
S0 39%, pełna informacja 43% (mediana APE 35% w obu przypadkach).

- **Bez CoT** — z CoT wolniej i gorzej.
- **Odrzucamy dania < 50 kcal** (`data.min_kcal: 50`). Przy kilku–kilkunastu kcal błąd procentowy
  eksploduje (10 kcal vs 35 kcal = 250%) i jedno takie danie dominuje średnią. Zbiór: 300 walidacja, 424 test.
- **„Pełna informacja nie pomaga” traktujemy jako wynik**, bez dostrajania promptu na siłę. Dlatego H1
  mierzymy względem **oracle** przy tym samym budżecie (`share_of_oracle`), a odzyskany zysk względem
  wszystkich 7 pytań raportujemy pomocniczo.
- **H4:** 82% dań ma > 10% kcal z tłuszczu, więc próg 10% nic nie dzieli. **Przyjmujemy próg 30%.**
- **σ = 0,33 zostaje** (jak w planie i prezentacji). Siatka σ ∈ {0,1; 0,33; 0,5} i tak pokazuje,
  jak wnioski zależą od szumu, a 0,375 (dokładne 50% w ±25%) leży wewnątrz tej siatki.
- **Budżet obliczeń do 48 h** jest akceptowalny.
- **Drugi VLM (E5)** — później, jeśli będzie dostęp do większego GPU.
