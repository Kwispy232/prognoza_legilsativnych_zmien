# Analýza časového radu legislatívnych zmien

*Report vygenerovaný: 2025-05-08*

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
| Sezónna perióda | 70 |
| Typ trendu | add |
| Typ sezónnosti | add |
| Smoothing Level | 0.1 |
| Smoothing Trend | 0.01 |
| Smoothing Seasonal | 0.01 |

### 2.2 Metriky výkonu modelu

| Metrika | Hodnota |
|---------|--------|
| MSE | 20.2526 |
| RMSE | 4.5003 |
| MAE | 2.3939 |
| R² | 0.3922 |
| MASE | 1.1576 |

## 3. Predikcie

Model predikuje nasledujúce hodnoty pre budúce obdobia:

| Dátum | Predikcia |
|--------|----------|
| 2024-04-01 | 5.56 |
| 2024-04-02 | 1.96 |
| 2024-04-03 | 1.96 |
| 2024-04-04 | 1.97 |
| 2024-04-05 | 1.98 |
| 2024-04-06 | 1.99 |
| 2024-04-07 | 8.80 |
| 2024-04-08 | 12.81 |
| 2024-04-09 | 3.83 |
| 2024-04-10 | 2.04 |
| 2024-04-11 | 2.05 |
| 2024-04-12 | 2.06 |
| 2024-04-13 | 2.07 |
| 2024-04-14 | 9.88 |
| 2024-04-15 | 3.27 |
| 2024-04-16 | 2.88 |
| 2024-04-17 | 2.10 |
| 2024-04-18 | 2.11 |
| 2024-04-19 | 2.11 |
| 2024-04-20 | 2.12 |
| 2024-04-21 | 4.53 |
| 2024-04-22 | 13.53 |
| 2024-04-23 | 2.14 |
| 2024-04-24 | 2.15 |
| 2024-04-25 | 2.16 |
| 2024-04-26 | 3.77 |
| 2024-04-27 | 2.18 |
| 2024-04-28 | 15.38 |
| 2024-04-29 | 6.37 |
| 2024-04-30 | 2.19 |


## 4. Záver

Na základe analýzy časového radu legislatívnych zmien bol identifikovaný najlepší model s 70-dňovou sezónnou periódou. 

Model dosahuje koeficient determinácie (R²) 0.3922, čo znamená, že 39.2% variability v dátach je vysvetlenej modelom.

Hodnota MASE 1.1576 indikuje, že model je horší ako naivný model.

*Poznámka: Tento report bol automaticky vygenerovaný pomocou skriptu holt_winters_analysis.py*
