# Analýza časového radu legislatívnych zmien

*Report vygenerovaný: 2025-03-29*

## 1. Prehľad dát

- **Dátový súbor:** unique_dates_counts.csv
- **Časový rozsah:** 2023-04-14 to 2024-03-31
- **Počet pozorovaní:** 353
- **Frekvencia dát:** D
- **Stacionarita:** Stacionárny

## 2. Najlepší model Holt-Winters

### 2.1 Parametre modelu

| Parameter | Hodnota |
|-----------|--------|
| Sezónna perióda | 7 |
| Typ trendu | add |
| Typ sezónnosti | add |
| Smoothing Level | 0.5 |
| Smoothing Trend | 0.01 |
| Smoothing Seasonal | 0.1 |

### 2.2 Metriky výkonu modelu

| Metrika | Hodnota |
|---------|--------|
| MSE | 37.3442 |
| RMSE | 6.1110 |
| MAE | 3.4800 |
| R² | -0.1207 |
| MASE | 1.6828 |

## 3. Predikcie

Model predikuje nasledujúce hodnoty pre budúce obdobia:

| Dátum | Predikcia |
|--------|----------|
| 2024-04-01 | 14.40 |
| 2024-04-02 | 5.99 |
| 2024-04-03 | 5.92 |
| 2024-04-04 | 6.00 |
| 2024-04-05 | 6.15 |
| 2024-04-06 | 6.34 |
| 2024-04-07 | 14.49 |
| 2024-04-08 | 14.82 |
| 2024-04-09 | 6.41 |
| 2024-04-10 | 6.35 |
| 2024-04-11 | 6.42 |
| 2024-04-12 | 6.58 |
| 2024-04-13 | 6.76 |
| 2024-04-14 | 14.92 |
| 2024-04-15 | 15.24 |
| 2024-04-16 | 6.83 |
| 2024-04-17 | 6.77 |
| 2024-04-18 | 6.85 |
| 2024-04-19 | 7.00 |
| 2024-04-20 | 7.19 |
| 2024-04-21 | 15.34 |
| 2024-04-22 | 15.66 |
| 2024-04-23 | 7.25 |
| 2024-04-24 | 7.19 |
| 2024-04-25 | 7.27 |
| 2024-04-26 | 7.42 |
| 2024-04-27 | 7.61 |
| 2024-04-28 | 15.76 |
| 2024-04-29 | 16.09 |
| 2024-04-30 | 7.68 |


## 4. Záver

Na základe analýzy časového radu legislatívnych zmien bol identifikovaný najlepší model s 7-dňovou sezónnou periódou. 

Model dosahuje koeficient determinácie (R²) -0.1207, čo znamená, že 12.1% variability v dátach je vysvetlenej modelom.

Hodnota MASE 1.6828 indikuje, že model je horší ako naivný model.

*Poznámka: Tento report bol automaticky vygenerovaný pomocou skriptu holt_winters_analysis.py*
