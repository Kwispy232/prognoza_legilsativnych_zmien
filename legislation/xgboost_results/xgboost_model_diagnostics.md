# Diagnostika XGBoost modelu pre legislatívne časové rady

## 1. Prehľad modelu

XGBoost model bol použitý na predikciu legislatívnych časových radov. XGBoost je gradient boosting algoritmus, ktorý dokáže zachytiť nelineárne vzťahy v dátach.

### Základné parametre modelu:
- **Objective**: reg:squarederror
- **Learning rate (eta)**: 0.1
- **Max depth**: 6
- **Subsample**: 0.8
- **Colsample bytree**: 0.8
- **Min child weight**: 1

### Metriky presnosti:

| Dataset    | RMSE   | MAE    | R²     |
|------------|--------|--------|--------|
| Trénovací  | 0.0025 | 0.0019 | 1.0000 |
| Testovací  | 3.8104 | 1.7607 | 0.6615 |

Hodnota R² = 0.6615 na testovacích dátach znamená, že model vysvetľuje približne 66.2% variability v legislatívnych dátach, čo je výrazne lepšie ako Holt-Winters model (R² = 0.3866).

## 2. Fit modelu na testovacie dáta

Nasledujúci graf zobrazuje skutočné a predikované hodnoty na testovacom datasete. Toto je dôležitá vizualizácia, ktorá ukazuje, ako dobre model dokáže predikovať budúce hodnoty.

![Fit na testovacie dáta](xgboost_results/test_data_fit.png)


## 3. Analýza reziduálov

### 3.1 Ljung-Box test autokorelácií

Ljung-Box test testuje nulovú hypotézu, že autokorelácie až po lag m sú nulové. Inými slovami, testuje, či sú reziduály náhodné a nezávislé.

| Lag | Testová štatistika | p-hodnota |
|-----|-------------------|-------------------|
| 1 | 11.8544 | 6e-04 |
| 5 | 11.8573 | 0.0368 |
| 10 | 55.5913 | 0.0000 |
| 15 | 121.2556 | 0.0000 |
| 20 | 129.3357 | 0.0000 |
| 30 | 157.9175 | 0.0000 |

**Interpretácia**: 
Ak p-hodnota < 0.05, zamietame nulovú hypotézu a reziduály vykazujú štatisticky významnú autokoreláciu, čo znamená, že model nezachytáva všetky časové závislosti v dátach.
Ak p-hodnota >= 0.05, nezamietame nulovú hypotézu a reziduály nevykazujú štatisticky významnú autokoreláciu, čo je pozitívny znak.

### 3.2 Autokorelácie a parciálne autokorelácie reziduálov

Nasledujúce grafy zobrazujú autokorelácie (ACF) a parciálne autokorelácie (PACF) reziduálov pre rôzne lagy:

![ACF reziduálov](xgboost_results/residuals_acf.png)

![PACF reziduálov](xgboost_results/residuals_pacf.png)

Modrá prerušovaná čiara v grafoch predstavuje hranicu štatistickej významnosti (±1.96/√n). Stĺpce, ktoré prekračujú túto hranicu, indikujú štatisticky významnú autokoreláciu na danom lagu.

### 3.3 Test ARCH efektu

ARCH efekt (Autoregressive Conditional Heteroskedasticity) označuje prítomnosť časovo premenlivej volatility v časových radoch. Na testovanie ARCH efektu sme použili Ljung-Box test na kvadrátoch reziduálov.

| Lag | Testová štatistika | p-hodnota |
|-----|-------------------|-------------------|
| 1 | 3.7253 | 0.0536 |
| 5 | 3.9234 | 0.5605 |
| 10 | 42.8787 | 0.0000 |
| 15 | 95.4957 | 0.0000 |
| 20 | 97.5127 | 0.0000 |

**Interpretácia**: 
Ak p-hodnota < 0.05, existuje významná časová závislosť vo volatilite reziduálov (ARCH efekt).
Ak p-hodnota >= 0.05, neexistuje významná časová závislosť vo volatilite reziduálov.

### 3.4 Test normality reziduálov

Normalita reziduálov nie je kritickým predpokladom pre XGBoost ako neparametrický model, ale je dôležitá pre správnu interpretáciu intervalov spoľahlivosti.

#### 3.4.1 Shapiro-Wilk test

**Testová štatistika W**: 0.1462

**p-hodnota**: 0.0000

#### 3.4.2 Jarque-Bera test

**Testová štatistika**: 83736.0109

**p-hodnota**: 0.0000

**Interpretácia**:
Ak p-hodnota < 0.05 (pre ktorýkoľvek test), reziduály nemajú normálne rozdelenie.
Ak p-hodnota >= 0.05, nie je dôvod zamietať predpoklad normality reziduálov.

### 3.5 Vizualizácie normality reziduálov

Nasledujúce grafy pomáhajú vizuálne posúdiť normalitu reziduálov:

![Histogram reziduálov](xgboost_results/residuals_histogram.png)

![Q-Q plot reziduálov](xgboost_results/residuals_qqplot.png)

### 3.6 Homoskedasticita reziduálov

Homoskedasticita označuje konštantný rozptyl reziduálov. Nasledujúci graf zobrazuje reziduály vzhľadom na predikované hodnoty:

![Reziduály vs. predikované hodnoty](xgboost_results/residuals_vs_fitted.png)

**Interpretácia**:
Ak reziduály vykazujú systematický vzor (napr. lievik, zakrivenie), môže to indikovať heteroskedasticitu alebo nesprávnu špecifikáciu modelu.
Ideálne by mali byť reziduály náhodne rozptýlené okolo horizontálnej línie nuly.

## 4. Prognóza s intervalmi spoľahlivosti

XGBoost model generuje bodové predikcie, ale pre praktické použitie sú dôležité aj intervaly spoľahlivosti. V tomto prípade sme vygenerovali 95% intervaly spoľahlivosti pomocou bootstrapingu reziduálov.

![Prognóza s intervalmi spoľahlivosti](xgboost_results/forecast_with_intervals.png)

Intervaly spoľahlivosti reprezentujú neistotu spojenú s predikciou a poskytujú rozsah hodnôt, v ktorom s 95% pravdepodobnosťou bude ležať skutočná hodnota.

## 5. Dôležitosť prediktorov

XGBoost model poskytuje informácie o dôležitosti jednotlivých prediktorov:

![Dôležitosť prediktorov](xgboost_results/feature_importance.png)

Toto je cenná informácia, ktorá ukazuje, ktoré prediktory majú najväčší vplyv na predikcie modelu.

## 6. Závery a odporúčania

### 6.1 Zhrnutie výsledkov diagnostiky

1. **Výkonnosť modelu**: XGBoost model dosahuje R² = 0.66 na testovacích dátach, čo je výrazne lepšie ako Holt-Winters model (R² = 0.39).

2. **Analýza reziduálov**:
   - **Autokorelácia reziduálov**: Ljung-Box testy odhalili štatisticky významnú autokoreláciu v reziduáloch, čo naznačuje, že model nezachytáva všetky časové závislosti v dátach.
   - **ARCH efekt**: Testy na ARCH efekt odhalili časovo závislú volatilitu v reziduáloch, čo naznačuje prítomnosť ARCH efektu.
   - **Normalita reziduálov**: Shapiro-Wilk test (p-hodnota = 0.0000) a Jarque-Bera test (p-hodnota = 0.0000) naznačujú, že reziduály nie sú normálne rozdelené. Pre XGBoost ako neparametrický model to však nie je kritický problém, ale môže to ovplyvniť presnosť intervalov spoľahlivosti.

## 7. Porovnanie s predchádzajúcimi modelmi

| Model        | Testovací R² | RMSE   | MAE    |
|--------------|--------------|--------|--------|
| XGBoost      | 0.66 | 3.81 | 1.76 |
| Holt-Winters | 0.39 | 4.96 | 2.67 |

XGBoost model výrazne prekonáva tak Holt-Winters v presnosti predikcie. Hodnota R² = 0.66 znamená, že model dokáže vysvetliť približne 66.2% variability v legislatívnych dátach, čo je výrazné zlepšenie oproti predchádzajúcim modelom. To potvrdzuje schopnosť XGBoost modelu zachytiť komplexné nelineárne vzťahy a sezónne vzory v legislatívnych dátach.

---

*Poznámka: Táto správa bola vygenerovaná na základe diagnostickej analýzy XGBoost modelu pre legislatívne časové rady. Kompletné vizualizácie a detailné výsledky sú k dispozícii v priečinku 'legislation/xgboost_results'.*
