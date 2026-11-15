# 01 — Luka badawcza i temat pracy

## Temat (tytuł roboczy)

**PL:** Ile pytań wystarczy? Wybór pytań sterowany niepewnością przy realistycznych odpowiedziach użytkownika w interaktywnej estymacji kaloryczności posiłku ze zdjęcia

**EN:** *How Many Questions Are Enough? Uncertainty-Driven, Budget-Constrained Clarification with Noisy User Answers for Image-Based Calorie Estimation*

## Problem

Szacowanie kalorii z jednego zdjęcia ma błąd, którego sam obraz nie usunie:
- **Porcja (masa/objętość).** Zdjęcie 2D nie oddaje trzeciego wymiaru. To główne źródło błędu: [Fridolfsson 2025], [Nutrients 2026], [Vinod & Zhu 2026].
- **Ukryte składniki**, czyli olej, masło, sosy i dressingi. Wnoszą dużo kalorii, a często ich nie widać [Bhatambarekar 2025].
- Najlepsze modele RGB na Nutrition5k mają ok. 14–26% błędu kalorii (PMAE) [Thames 2021; Han 2023]. Ogólne VLM w trybie zero-shot mylą się o 35–55% (MAPE) [Fridolfsson 2025; Bhatambarekar 2025].

Dodatkowa informacja od użytkownika wyraźnie pomaga. U Bhatambarekara podanie masy składników obniżyło MAPE GPT-4o z ok. 51% do ok. 29%. Metadane kontekstowe też zmniejszają błąd [Coburn 2025]. Sama nazwa dania daje jednak niewiele: −1 kcal MAE [Narang 2025].

## Co już zrobiono (stan wiedzy)

| Praca | Interakcja | Ile pytań | Jak modelowane odpowiedzi |
|---|---|---|---|
| Bhatambarekar 2025 (thesis VT) | „cue ladder”: składniki → masa całkowita → masa składników | stała drabinka | **dokładne wartości z datasetu** (ground truth) |
| Bhatambarekar et al., CVPRW 2026 | stała sekwencja 6 pytań vs. agent sam zadaje pytania | 6 / dowolnie | odpowiedzi z GT |
| Coburn et al. 2025 (ACETADA) | metadane: miejsce, pora posiłku, lista potraw | brak dialogu | metadane prawdziwe |
| CaLoRAify 2024, Food-R1 2026 | rozmowa z VLM | brak budżetu | — |
| Narang 2025 | nazwa dania | 1 wskazówka | prawdziwa |

## Luka badawcza

1. **Brak budżetu pytań.** Wcześniejsze prace pytają o wszystko (drabinka, 6 pytań) albo pozwalają agentowi pytać bez limitu. Nikt nie mierzy, **jak dokładność zależy od liczby pytań (0–3)**, ani nie sprawdza, **które pytanie zadać, gdy wolno zadać tylko jedno lub dwa**. W aplikacji każde pytanie zwiększa obciążenie użytkownika (*user burden*), a mniejsze obciążenie zwykle oznacza mniejszą dokładność [Höchsmann & Martin 2020].
2. **Nierealistyczne odpowiedzi.** Interaktywne VLM dostają w eksperymentach *prawdziwą* masę w gramach. Ludzie szacują porcje słabo: tylko ok. 35–50% odpowiedzi mieści się w ±25% prawdziwej masy [Lucassen 2021], a gramy są przeszacowywane średnio o ok. 22% (mała próba, praca dyplomowa). Nie wiadomo, **ile z deklarowanego zysku zostaje przy realistycznych, zaszumionych lub zgrubnych odpowiedziach**, ani **czy wtedy zmienia się ranking najbardziej wartościowych pytań**.
3. **Metody aktywnego pozyskiwania cech** (*active feature acquisition*, AFA) i **wyboru pytań maksymalizującego zysk informacji** (EIG) dla LLM [Aronsson 2025; Kobalczyk 2025] **nie były stosowane** do estymacji kaloryczności ze zdjęcia.

Szukaliśmy prac łączących budżet pytań, zaszumione odpowiedzi i estymację kalorii przez VLM. Nie znaleźliśmy żadnej (stan na 28.10.2026).

## Pytania badawcze

- **RQ1:** Jak błąd estymacji kalorii zależy od budżetu pytań (0, 1, 2, 3)? Ile zysku z „pełnej informacji” dają 1–2 pytania?
- **RQ2:** Czy dobór pytań sterowany niepewnością (adaptacyjny, zależny od zdjęcia) daje mniejszy błąd niż stała kolejność, losowy wybór i agent VLM pytający swobodnie, przy tym samym budżecie?
- **RQ3:** Jak szum i zgrubność odpowiedzi użytkownika (gramy z błędem, kategorie mała/średnia/duża, pytania tak/nie) zmieniają zysk z interakcji i ranking najcenniejszych pytań?

## Wkład (contributions)

1. Protokół ewaluacji interaktywnej estymacji kalorii **z budżetem pytań i symulowanym użytkownikiem**. Modele szumu skalibrujemy na danych o ludzkim szacowaniu porcji.
2. Krzywe **błąd vs. liczba pytań** dla otwartego VLM na Nutrition5k oraz porównanie strategii wyboru pytań.
3. Analiza, **jakie pytania warto zadawać realnemu użytkownikowi** (np. „czy był olej?” zamiast „ile gramów?”), z rekomendacjami projektowymi dla aplikacji.

## Autorzy

Adam Sokołowski, Antoni Grajek

## Praca bazowa

Lo, Sun, Qiu & Lo (2020), *Image-Based Food Classification and Volume Estimation for Dietary Assessment: A Review*, IEEE JBHI [lo2020review]. To przegląd klasyfikacji żywności i estymacji objętości: pokazuje, że objętość/porcja to najtrudniejszy etap. To bezpośrednio motywuje nasze pytania o porcję.
