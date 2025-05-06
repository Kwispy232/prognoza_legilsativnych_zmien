#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Holt-Winters analýza časových radov pre legislatívne zmeny

Tento skript implementuje Holt-Winters exponenciálne vyhladzovanie na analýzu 
časových radov legislatívnych zmien. Umožňuje testovanie stacionarity dát, 
analýzu sezónnosti a optimalizáciu parametrov modelu Holt-Winters.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import os
from datetime import datetime, timedelta

# Nastavenie štýlu grafov
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)  # Veľkosť grafov
plt.rcParams['font.size'] = 12  # Veľkosť písma v grafoch

# Vytvorenie výstupného adresára pre vizualizácie a výsledky
output_dir = 'legislation/hw_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_data(filename='legislation/unique_dates_counts.csv'):
    """Načítanie a príprava údajov časového radu
    
    Táto funkcia načíta údaje zo súboru CSV, skonvertuje stĺpec s dátumami
    na formát datetime, nastaví dátum ako index a zoradí dáta podľa dátumu.
    
    Parametre:
    ----------
    filename : str, voliteľný
        Cesta k súboru CSV s údajmi (predvolene 'unique_dates_counts.csv')
        
    Návratová hodnota:
    -----------------
    pandas.DataFrame
        Dataframe s načítanými a pripravenými údajmi
    """
    print(f"Loading dataset: {filename}")
    df = pd.read_csv(filename)
    
    # Konverzia dátumu na formát datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Nastavenie dátumu ako indexu
    df = df.set_index('Date')
    
    # Zoradenie podľa dátumu
    df = df.sort_index()
    
    # Zobrazenie základných informácií o dátach
    print("\nData Overview:")
    print(f"Date Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"Number of observations: {len(df)}")
    print(f"Data frequency: {pd.infer_freq(df.index) or 'D'}")
    
    return df

def test_stationarity(series):
    """Testovanie stacionarity časového radu pomocou ADF testu
    
    Táto funkcia vykoná Augmented Dickey-Fuller test na overenie stacionarity
    časového radu. Stacionarita je dôležitá pre správne fungovanie modelov
    časových radov.
    
    Parametre:
    ----------
    series : pandas.Series
        Časový rad na testovanie
        
    Návratová hodnota:
    -----------------
    bool
        True ak je časový rad stacionárny podľa ADF testu, inak False
    """
    print("\n=== Stationarity Tests ===")
    
    # ADF Test (Augmented Dickey-Fuller)
    print("\nAugmented Dickey-Fuller Test:")
    adf_result = adfuller(series.dropna())
    adf_output = pd.Series(
        [adf_result[0], adf_result[1], adf_result[4]['1%'], adf_result[4]['5%'], adf_result[4]['10%']],
        index=['Test Statistic', 'p-value', '1% Critical Value', '5% Critical Value', '10% Critical Value']
    )
    print(adf_output)
    is_adf_stationary = adf_result[1] < 0.05  # p-hodnota < 0.05 znamená, že môžeme zamietnuť nulovú hypotézu o nestacionarite
    print(f"Conclusion: Series is {'stationary' if is_adf_stationary else 'non-stationary'} according to ADF test")
    
    return is_adf_stationary

def analyze_seasonality(series):
    """Analýza sezónnych vzorov v časovom rade
    
    Táto funkcia analyzuje sezónnosť v časovom rade pomocou autokorelačnej funkcie (ACF)
    a parciálnej autokorelačnej funkcie (PACF). Vytvára a ukladá grafy týchto funkcií.
    
    Parametre:
    ----------
    series : pandas.Series
        Časový rad na analýzu
    """
    print("\n=== Seasonality Analysis ===")
    
    # Vykreslenie ACF a PACF grafov
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plot_acf(series.dropna(), lags=40, ax=ax1)  # Autokorelačná funkcia
    ax1.set_title('Autocorrelation Function')
    plot_pacf(series.dropna(), lags=40, ax=ax2)  # Parciálna autokorelačná funkcia
    ax2.set_title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/acf_pacf.png")
    plt.close()

def evaluate_model(predictions, actual):
    """Výpočet metrík pre vyhodnotenie modelu
    
    Táto funkcia počíta rôzne metriky chýb na vyhodnotenie presnosti modelu.
    Používa metriky, ktoré dobre fungujú aj s časovými radmi obsahujúcimi nuly.
    
    Parametre:
    ----------
    predictions : array-like
        Predikované hodnoty modelu
    actual : array-like
        Skutočné hodnoty
        
    Návratová hodnota:
    -----------------
    dict
        Slovník obsahujúci vypočítané metriky:
        - MSE: Stredná kvadratická chyba (Mean Squared Error)
        - RMSE: Odmocnina strednej kvadratickej chyby (Root Mean Squared Error)
        - MAE: Stredná absolútna chyba (Mean Absolute Error)
        - R²: Koeficient determinácie (R-squared)
        - MASE: Stredná absolútna škálovaná chyba (Mean Absolute Scaled Error)
    """
    # Výpočet základných metrík chýb
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predictions)
    
    # Výpočet koeficientu determinácie (R²)
    # R² meria, ako dobre model vysvetľuje variabilitu v dátach
    # Funguje dobre aj s dátami obsahujúcimi nuly
    ss_total = np.sum((actual - np.mean(actual))**2)  # Celková suma štvorcov
    ss_residual = np.sum((actual - predictions)**2)  # Reziduálna suma štvorcov
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    
    # Výpočet MASE (Mean Absolute Scaled Error)
    # MASE je nezávislá od mierky a dobre zvláda nuly v dátach
    # Porovnáva MAE s MAE naivnej predikcie
    if len(actual) > 1:
        # Použitie jednokrokovej naivnej predikcie (predchádzajúca hodnota) ako základu
        naive_errors = np.abs(actual[1:] - actual[:-1])
        
        # Ak nie sú žiadne nenulové chyby v naivnej predikcii, použitie iného základu
        if np.sum(naive_errors) == 0 or len(naive_errors) == 0:
            # Použitie priemeru časového radu ako jednoduchého základu
            mean_actual = np.mean(actual)
            if mean_actual == 0:
                # Ak je priemer tiež nula, použitie malej konštanty
                mase = mae / 0.1 if mae != 0 else 0.0
            else:
                # Porovnanie MAE s priemerom skutočných hodnôt
                mase = mae / mean_actual
        else:
            # Štandardný výpočet MASE
            naive_mae = np.mean(naive_errors)
            mase = mae / naive_mae
    else:
        # Pre jednobodový časový rad použitie rozumnej predvolenej hodnoty
        mase = 1.0 if mae == 0 else float('inf')
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r_squared,
        'MASE': mase
    }

def optimize_holt_winters(series, seasonal_periods=None):
    """Optimalizácia modelu Holt-Winters s rôznymi parametrami
    
    Táto funkcia spúšťa viacero modelov Holt-Winters s rôznymi kombináciami
    parametrov, aby našla optimálnu konfiguráciu. Testuje rôzne typy trendov,
    sezónnosti a vyhladzovacie parametre.
    
    Parametre:
    ----------
    train : pandas.Series
        Trénovacia množina časového radu
    test : pandas.Series
        Testovacia množina časového radu
    seasonal_periods : int, voliteľný
        Počet období v sezónnom cykle (napr. 7 pre týždennú sezónnosť)
        
    Návratová hodnota:
    -----------------
    dict alebo None
        Parametre najlepšieho modelu alebo None, ak sa nenašiel žiadny platný model
    """
    print("\n=== Holt-Winters Model Optimization ===")
    
    # If seasonal_periods is None, try to infer from data
    if seasonal_periods is None:
        if pd.infer_freq(series.index) == 'D':
            seasonal_periods = 7  # Weekly seasonality for daily data
        elif pd.infer_freq(series.index) == 'M':
            seasonal_periods = 12  # Yearly seasonality for monthly data
        else:
            seasonal_periods = 7  # Default to weekly
    
    print(f"Using seasonal period: {seasonal_periods}")
    
    # Parameter grid
    trend_types = ['add', 'mul', None]
    seasonal_types = ['add', 'mul', None]
    smoothing_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    smoothing_slopes = [0.01, 0.1, 0.3, 0.5, 0.9]
    smoothing_seasonals = [0.01, 0.1, 0.3, 0.5, 0.9]
    
    # Reduced parameter grid for faster execution
    param_grid = list(itertools.product(
        trend_types,
        seasonal_types,
        smoothing_levels,  # Reduced smoothing_level options
        smoothing_slopes,       # Reduced smoothing_slope options
        smoothing_seasonals        # Reduced smoothing_seasonal options
    ))
    
    # Filter out invalid combinations
    valid_params = []
    for params in param_grid:
        trend, seasonal, sl, ss, ssa = params
        # Skip if trend is None but smoothing_slope is specified
        if trend is None and ss is not None:
            continue
        # Skip if seasonal is None but smoothing_seasonal is specified
        if seasonal is None and ssa is not None:
            continue
        valid_params.append(params)
    
    print(f"Testing {len(valid_params)} parameter combinations...")
    
    # Store results
    results = []
    
    # Run models with different parameters
    for i, (trend, seasonal, sl, ss, ssa) in enumerate(valid_params):
        try:
            # Skip invalid combinations
            if trend is None and seasonal is None:
                continue
                
            # Create model
            model_params = {
                'trend': trend,
                'seasonal': seasonal,
                'seasonal_periods': seasonal_periods if seasonal else None
            }
            
            # Add smoothing parameters if applicable
            fit_params = {}
            if sl is not None:
                fit_params['smoothing_level'] = sl
            if trend is not None and ss is not None:
                fit_params['smoothing_trend'] = ss
            if seasonal is not None and ssa is not None:
                fit_params['smoothing_seasonal'] = ssa
            
            # Print progress only occasionally to reduce verbosity
            if i % 50 == 0:
                print(f"Testing combination {i+1}/{len(valid_params)}...")
            
            # Fit model
            model = ExponentialSmoothing(series, **model_params)
            fit = model.fit(**fit_params)
            
            # Make predictions
            predictions = fit.fittedvalues
            
            # Evaluate
            metrics = evaluate_model(predictions, series)
            
            # Store results
            results.append({
                'trend': trend,
                'seasonal': seasonal,
                'seasonal_periods': seasonal_periods if seasonal else None,
                'smoothing_level': sl,
                'smoothing_trend': ss if trend else None,
                'smoothing_seasonal': ssa if seasonal else None,
                'MSE': metrics['MSE'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R²': metrics['R²'],
                'MASE': metrics['MASE'],
                'AIC': fit.aic,
                'BIC': fit.bic,
                'model': fit
            })
            
        except Exception as e:
            # Zobrazenie chyby len pre multiplikativne modely (najcastejsia chyba)
            if trend == 'mul' or seasonal == 'mul':
                if i % 50 == 0:  # Obmedzenie poctu vypisov chyb
                    print(f"Error with multiplicative model: {str(e)}")
    
    # Sort results by MSE
    results.sort(key=lambda x: x['MSE'])
    
    # Display top 5 models v tabulkovej forme pre lepsiu citatelnost
    print("\nTop 5 Models by MSE:")
    print(f"{'Rank':<5}{'Trend':<8}{'Seasonal':<10}{'Period':<8}{'MSE':<10}{'R²':<10}{'MASE':<10}")
    print("-" * 60)
    
    for i, result in enumerate(results[:5]):
        print(f"{i+1:<5}{result['trend'] or 'None':<8}{result['seasonal'] or 'None':<10}{result['seasonal_periods'] or 'N/A':<8}{result['MSE']:<10.4f}{result['R²']:<10.4f}{result['MASE']:<10.4f}")
    
    # Return the best model
    return results[0] if results else None

def generate_markdown_report(best_params, forecast_series, in_sample_metrics, data_info):
    """Generovanie markdown reportu s výsledkami analýzy
    
    Táto funkcia vytvára podrobný markdown report s výsledkami analýzy časového radu,
    vrátane informácií o dátach, najlepšom modeli, metrikách výkonu a predikciách.
    
    Parametre:
    ----------
    best_params : dict
        Parametre najlepšieho modelu z optimalizácie
    forecast_series : pandas.Series
        Časový rad s predikciami
    in_sample_metrics : dict
        Metriky výkonu modelu na trénovacích dátach
    data_info : dict
        Informácie o analyzovaných dátach
        
    Návratová hodnota:
    -----------------
    str
        Obsah markdown reportu
    """
    # Základné informácie o reporte
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Vytvorenie reportu
    report = f"""# Analýza časového radu legislatívnych zmien

*Report vygenerovaný: {current_date}*

## 1. Prehľad dát

- **Dátový súbor:** {data_info['filename']}
- **Časový rozsah:** {data_info['date_range']}
- **Počet pozorovaní:** {data_info['observations']}
- **Frekvencia dát:** {data_info['frequency']}
- **Stacionarita:** {data_info['stationarity']}

## 2. Najlepší model Holt-Winters

### 2.1 Parametre modelu

| Parameter | Hodnota |
|-----------|--------|
| Sezónna perióda | {best_params['seasonal_periods']} |
| Typ trendu | {best_params['trend']} |
| Typ sezónnosti | {best_params['seasonal']} |
| Smoothing Level | {best_params['smoothing_level']} |
| Smoothing Trend | {best_params['smoothing_trend']} |
| Smoothing Seasonal | {best_params['smoothing_seasonal']} |

### 2.2 Metriky výkonu modelu

| Metrika | Hodnota |
|---------|--------|
| MSE | {in_sample_metrics['MSE']:.4f} |
| RMSE | {in_sample_metrics['RMSE']:.4f} |
| MAE | {in_sample_metrics['MAE']:.4f} |
| R² | {in_sample_metrics['R²']:.4f} |
| MASE | {in_sample_metrics['MASE']:.4f} |

## 3. Predikcie

Model predikuje nasledujúce hodnoty pre budúce obdobia:

| Dátum | Predikcia |
|--------|----------|
"""
    
    # Pridanie predikcií do tabuľky
    for date, value in forecast_series.items():
        report += f"| {date.strftime('%Y-%m-%d')} | {value:.2f} |\n"
    
    # Pridanie záveru
    report += f"""

## 4. Záver

Na základe analýzy časového radu legislatívnych zmien bol identifikovaný najlepší model s {best_params['seasonal_periods']}-dňovou sezónnou periódou. 

Model dosahuje koeficient determinácie (R²) {in_sample_metrics['R²']:.4f}, čo znamená, že {abs(in_sample_metrics['R²']*100):.1f}% variability v dátach je vysvetlenej modelom.

Hodnota MASE {in_sample_metrics['MASE']:.4f} indikuje, že model je {"lepší" if in_sample_metrics['MASE'] < 1 else "horší"} ako naivný model.

*Poznámka: Tento report bol automaticky vygenerovaný pomocou skriptu holt_winters_analysis.py*
"""
    
    return report



def final_model_analysis(best_params, full_series, forecast_periods=30):
    """Trénovanie najlepšieho modelu na celom datasete a generovanie predikcií
    
    Táto funkcia trénuje najlepší model Holt-Winters (s optimálnymi parametrami)
    na celom datasete a generuje predikcie do budúcnosti. Vytvára a ukladá
    vizualizácie predikcií a komponentov modelu.
    
    Parametre:
    ----------
    best_params : dict
        Parametre najlepšieho modelu z optimalizácie
    full_series : pandas.Series
        Celý časový rad na trénovanie
    forecast_periods : int, voliteľný
        Počet období do budúcnosti na predikciu (predvolene 30)
        
    Návratová hodnota:
    -----------------
    dict
        Slovník obsahujúci predikcie a metriky výkonu modelu
    """
    print("\n=== Final Model Analysis ===")
    
    # Extract parameters
    trend = best_params['trend']
    seasonal = best_params['seasonal']
    seasonal_periods = best_params['seasonal_periods']
    smoothing_level = best_params['smoothing_level']
    smoothing_trend = best_params['smoothing_trend']
    smoothing_seasonal = best_params['smoothing_seasonal']
    
    # Print final model configuration
    print("\nFinal Model Configuration:")
    print(f"Trend: {trend}")
    print(f"Seasonal: {seasonal}")
    print(f"Seasonal Periods: {seasonal_periods}")
    print(f"Smoothing Level: {smoothing_level}")
    print(f"Smoothing Trend: {smoothing_trend}")
    print(f"Smoothing Seasonal: {smoothing_seasonal}")
    
    # Create model
    model_params = {
        'trend': trend,
        'seasonal': seasonal,
        'seasonal_periods': seasonal_periods
    }
    
    # Add smoothing parameters
    fit_params = {}
    if smoothing_level is not None:
        fit_params['smoothing_level'] = smoothing_level
    if trend is not None and smoothing_trend is not None:
        fit_params['smoothing_trend'] = smoothing_trend
    if seasonal is not None and smoothing_seasonal is not None:
        fit_params['smoothing_seasonal'] = smoothing_seasonal
    
    # Fit model on full dataset
    model = ExponentialSmoothing(full_series, **model_params)
    fit = model.fit(**fit_params)
    
    # Generate in-sample predictions
    in_sample_predictions = fit.fittedvalues
    
    # Calculate in-sample metrics
    in_sample_metrics = evaluate_model(in_sample_predictions, full_series)
    print("\nIn-Sample Performance:")
    print(f"MSE: {in_sample_metrics['MSE']:.4f}")
    print(f"RMSE: {in_sample_metrics['RMSE']:.4f}")
    print(f"MAE: {in_sample_metrics['MAE']:.4f}")
    print(f"R²: {in_sample_metrics['R²']:.4f}")
    print(f"MASE: {in_sample_metrics['MASE']:.4f}")
    
    # Generate forecast
    last_date = full_series.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods)
    forecast = fit.forecast(forecast_periods)
    forecast_series = pd.Series(forecast, index=forecast_dates)
    
    # Create and save forecast visualization
    plt.figure(figsize=(12, 6))
    plt.plot(full_series, label='Historical Data')
    plt.plot(in_sample_predictions, color='red', linestyle='--', label='Fitted Values')
    plt.plot(forecast_series, color='green', label=f'{forecast_periods}-Day Forecast')
    plt.title('Holt-Winters Time Series Forecast')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/forecast.png")
    plt.close()
    
    # Create and save components visualization
    if hasattr(fit, 'level') and hasattr(fit, 'slope') and hasattr(fit, 'season'):
        plt.figure(figsize=(12, 10))
        
        # Level
        plt.subplot(3, 1, 1)
        plt.plot(fit.level)
        plt.title('Level')
        plt.grid(True)
        
        # Trend
        if trend is not None:
            plt.subplot(3, 1, 2)
            plt.plot(fit.slope)
            plt.title('Trend')
            plt.grid(True)
        
        # Seasonal
        if seasonal is not None:
            plt.subplot(3, 1, 3)
            plt.plot(fit.season)
            plt.title('Seasonal')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/components.png")
        plt.close()
    
    # Save forecast to CSV
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': forecast
    })
    forecast_df.to_csv(f"{output_dir}/forecast.csv", index=False)
    print(f"\nForecast saved to {output_dir}/forecast.csv")
    
    # Vrátenie výsledkov pre použitie v reporte
    return {
        'forecast_series': forecast_series,
        'in_sample_metrics': in_sample_metrics,
        'model_config': best_params
    }

def test_multiple_seasonal_periods(series):
    """Testovanie rôznych sezónnych periód a výber najlepšieho modelu
    
    Táto funkcia testuje rôzne sezónne periódy a vyberá najlepší model
    na základe najnižšej MSE na testovacej množine.
    
    Parametre:
    ----------
    train : pandas.Series
        Trénovacia množina časového radu
    test : pandas.Series
        Testovacia množina časového radu
        
    Návratová hodnota:
    -----------------
    dict
        Parametre najlepšieho modelu
    """
    # Vytvorenie zoznamu všetkých možných periód od 2 do 90 dní
    # Holt-Winters vyžaduje periódu väčšiu ako 1
    seasonal_periods_to_test = list(range(2, 91))  # všetky periódy od 2 do 90 dní
    
    # Filtrovanie periód, ktoré sú príliš dlhé pre dané dáta
    max_period = len(series) // 2  # Maximálna perióda by nemala byť väčšia ako polovica dĺžky trénovacích dát
    seasonal_periods_to_test = [p for p in seasonal_periods_to_test if p <= max_period]
    
    print(f"\n=== Testing Multiple Seasonal Periods ===")
    print(f"Testing seasonal periods: {seasonal_periods_to_test}")
    
    best_model = None
    best_mse = float('inf')
    
    # Vytvorenie zoznamu na uloženie výsledkov pre každú periódu
    period_results = []
    
    # Zobrazenie progress baru pre testovanie periód
    total_periods = len(seasonal_periods_to_test)
    print(f"\nTesting {total_periods} different seasonal periods...")
    
    # Testovanie každej sezónnej periódy
    for i, period in enumerate(seasonal_periods_to_test):
        # Zobrazenie progress baru
        progress = (i + 1) / total_periods * 100
        if i % 5 == 0 or i == total_periods - 1:  # Zobrazenie len obcas
            print(f"Progress: {progress:.1f}% - Testing period: {period}")
            
        model_params = optimize_holt_winters(series, seasonal_periods=period)
        
        if model_params:
            period_results.append({
                'period': period,
                'MSE': model_params['MSE'],
                'params': model_params
            })
            
            if model_params['MSE'] < best_mse:
                best_mse = model_params['MSE']
                best_model = model_params
                print(f"New best model found with period {period}, MSE: {best_mse:.4f}")
    
    if period_results:
        # Zoradenie výsledkov podľa MSE
        period_results.sort(key=lambda x: x['MSE'])
        
        # Zobrazenie top 10 najlepších periód v kompaktnej tabuľke
        print(f"\n=== Top 10 Best Seasonal Periods ===")
        print(f"{'Rank':<5}{'Period':<8}{'MSE':<10}{'R²':<10}{'MASE':<10}{'Trend':<8}{'Seasonal':<8}")
        print("-" * 65)
        
        for i, result in enumerate(period_results[:10]):
            params = result['params']
            print(f"{i+1:<5}{params['seasonal_periods']:<8}{params['MSE']:<10.4f}{params['R²']:<10.4f}{params['MASE']:<10.4f}{params['trend']:<8}{params['seasonal']:<8}")
        
        # Zobrazenie najlepšieho modelu v kompaktnom formáte
        print(f"\n=== Best Model (Period: {best_model['seasonal_periods']}) ===")
        print(f"Configuration: {best_model['trend']} trend, {best_model['seasonal']} seasonal")
        print(f"Smoothing parameters: level={best_model['smoothing_level']}, trend={best_model['smoothing_trend']}, seasonal={best_model['smoothing_seasonal']}")
        print(f"Performance: MSE={best_model['MSE']:.4f}, R²={best_model['R²']:.4f}, MASE={best_model['MASE']:.4f}")
    
    return best_model

def main():
    """Hlavná funkcia na spustenie analýzy Holt-Winters
    
    Táto funkcia koordinuje celý proces analýzy časového radu:
    1. Načítanie a prípravu dát
    2. Testovanie stacionarity
    3. Analýzu sezónnosti
    4. Rozdelenie dát na trénovaciu a testovaciu množinu
    5. Testovanie rôznych sezónnych periód
    6. Optimalizáciu modelu Holt-Winters s najlepšou periódou
    7. Trénovanie finálneho modelu a generovanie predikcií
    8. Vytvorenie markdown reportu s výsledkami
    """
    print("=== Holt-Winters Time Series Analysis ===")
    
    # Načítanie dát
    df = load_data()
    series = df['Count']
    filename = 'unique_dates_counts.csv'
    
    # Informácie o dátach pre report
    data_info = {
        'filename': filename,
        'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
        'observations': len(df),
        'frequency': pd.infer_freq(df.index) or 'D'
    }
    
    # Testovanie stacionarity
    is_adf_stationary = test_stationarity(series)
    data_info['stationarity'] = 'Stacionárny' if is_adf_stationary else 'Nestacionárny'
    
    # Analýza sezónnosti
    analyze_seasonality(series)
    
    # Rozdelenie dát na trénovaciu a testovaciu množinu
    # train, test = split_data(series)
    
    # Testovanie rôznych sezónnych periód a nájdenie najlepšieho modelu
    best_params = test_multiple_seasonal_periods(series)
    
    if best_params:
        # Finálny model a predikcie
        results = final_model_analysis(best_params, series)
        
        # Generovanie markdown reportu
        report = generate_markdown_report(
            best_params, 
            results['forecast_series'], 
            results['in_sample_metrics'], 
            data_info
        )
        
        # Uloženie reportu do súboru
        with open(f"{output_dir}/analysis_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"\nAnalysis complete! Report saved to {output_dir}/analysis_report.md")
        print(f"Check the {output_dir} directory for all output files.")
    else:
        print("\nNo valid model found. Please check your data or try different parameters.")

if __name__ == "__main__":
    main()