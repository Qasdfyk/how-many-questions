# 03 — Burza mózgów i hipotezy

## Rozważane podejścia

| # | Pomysł | Plusy | Minusy | Decyzja |
|---|---|---|---|---|
| 1 | Sam fine-tune regresora (ViT/CNN) na Nutrition5k | prosty | bez JFT-300M nie dogonimy SOTA [Andrade 2025]; nic nowego | baseline opcjonalny |
| 2 | VLM + stała lista pytań (wszystkie) | łatwe | to już zrobiono [Bhatambarekar 2025/26] | baseline („pełna informacja”) |
| 3 | VLM + agent pytający swobodnie | naturalne | zrobione, brak kontroli kosztu | baseline |
| 4 | **VLM + wybór pytań przy budżecie, sterowany niepewnością** | nowe, mierzalne, tanie obliczeniowo | trzeba zdefiniować niepewność | **wybrane** |
| 5 | **Symulowany użytkownik z realistycznym szumem** | nowe, łatwe do zrobienia offline | kalibracja szumu z literatury, nie z badań własnych | **wybrane** |
| 6 | Badanie z prawdziwymi użytkownikami | najmocniejsza walidacja | czas, zgody, mała próba | opcjonalnie, mini-pilot 10–20 dań w styczniu |
| 7 | Dwa zdjęcia / obiekt referencyjny (moneta, karta) | sprawdzone efekty | to inny kierunek niż pytania | poza zakresem, wzmianka w dyskusji |

## Pula pytań (kandydaci)

| ID | Pytanie | Typ odpowiedzi | Źródło odpowiedzi w Nutrition5k |
|---|---|---|---|
| Q1 | Ile ważył posiłek (g)? | liczba | `total_mass` |
| Q2 | Porcja: mała / średnia / duża? | 3 kategorie | tercyle `total_mass` |
| Q3 | Czy danie było smażone lub polane olejem/masłem? | tak/nie | składniki tłuszczowe w liście (olej, masło…) |
| Q4 | Czy był sos lub dressing? | tak/nie | składniki-sosy w liście |
| Q5 | Co jest głównym składnikiem? | tekst | składnik o największej masie |
| Q6 | Wymień składniki | lista | lista składników |
| Q7 | Ile jest osobnych elementów na talerzu? | liczba | liczba składników (zaokrąglona/zaszumiona) |

## Strategie wyboru pytań (przy budżecie B ∈ {0,1,2,3})

- **S0 — brak pytań** (sam obraz).
- **S1 — losowa.**
- **S2 — stała kolejność globalna (greedy na walidacji):** kolejność pytań średnio najbardziej redukujących błąd. Odpowiednik strategii „statycznej” z AFABench.
- **S3 — agent VLM:** model widzi pulę pytań i sam wybiera następne [jak Bhatambarekar].
- **S4 — sterowana niepewnością (nasza):** dla danego zdjęcia przewidujemy, które pytanie najbardziej obniży błąd. Niepewność mierzymy wariancją K próbek VLM i/lub deklarowaną pewnością modelu. Mały selektor (np. gradient boosting albo regresja na cechach: wariancja, typ dania z VLM, deklarowana porcja) uczony na zbiorze walidacyjnym przewiduje redukcję błędu dla każdego pytania. Wybór greedy, krok po kroku, z uwzględnieniem dotychczasowych odpowiedzi.
- **Wyrocznia (oracle):** najlepszy podzbiór pytań dla danego dania, wybrany na podstawie GT. Górna granica.

## Modele odpowiedzi użytkownika (symulowany użytkownik)

- **U0 idealny:** prawdziwe wartości (jak w dotychczasowej literaturze).
- **U1 gaussowski log-normalny dla masy:** `m' = m · exp(ε)`, `ε ~ N(μ, σ)`. Kalibracja na [Lucassen 2021]: ok. 50% odpowiedzi w ±25%, więc σ ≈ 0,33. Siatka σ ∈ {0,1; 0,33; 0,5}. Wariant z przesunięciem μ, np. +0,2 (przeszacowanie gramów ok. 22%).
- **U2 kategoryczny:** pytania S/M/L i tak/nie z prawdopodobieństwem pomyłki p ∈ {0; 0,1; 0,2}. Przy S/M/L pomyłka tylko do sąsiedniej kategorii.
- **U3 „nie wiem”:** z prawdopodobieństwem q użytkownik nie odpowiada, a budżet i tak jest zużyty.

## Hipotezy

- **H1 (budżet, malejące zwroty):** 2 dobrze wybrane pytania dają ≥ 70% redukcji błędu (MAPE) osiąganej przy pełnej informacji (wszystkie pytania, U0).
- **H2 (strategia):** przy budżecie B = 1 i B = 2 strategia S4 ma statystycznie istotnie niższy MAPE niż S1 i S2 (test Wilcoxona, α = 0,05) oraz nie gorszy niż S3.
- **H3 (realistyczne odpowiedzi):** przy realistycznym szumie (U1, σ ≈ 0,33) zysk z pytania o gramy (Q1) spada o ≥ 30% względem U0. Pytania kategoryczne i binarne (Q2, Q3) przy budżecie 1–2 dają wtedy zysk porównywalny z Q1 lub większy, czyli ranking pytań zmienia się względem idealnego użytkownika.
- **H4 (ukryte tłuszcze, pomocnicza):** pytanie Q3 o tłuszcz najbardziej redukuje błąd dla dań, w których tłuszcz stanowi > 10% kalorii.

Progi (70%, 30%) są wstępne. Na 09.11 możemy je zostawić jako „kierunkowe” albo doprecyzować po pilocie na 50 daniach.
