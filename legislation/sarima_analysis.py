#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SARIMA analýza časových radov pre legislatívne zmeny

Tento skript implementuje Seasonal ARIMA (SARIMA) model na analýzu 
časových radov legislatívnych zmien. Zahŕňa testovanie stacionarity, 
identifikáciu parametrov modelu, diagnostické testy a prognózy.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from datetime import datetime, timedelta
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose

# Nastavenie štýlu grafov
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Vytvorenie výstupného adresára pre vizualizácie a výsledky
output_dir = 'legislation/sarima_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_data(filename='legislation/unique_dates_counts.csv'):
    """
    Načítanie a príprava údajov časového radu
    
    Táto funkcia načíta údaje zo súboru CSV, skonvertuje stĺpec s dátumami
    na formát datetime, nastaví dátum ako index a zoradí dáta podľa dátumu.
    
    Parametre:
    ----------
    filename : str, voliteľný
        Cesta k súboru CSV s údajmi (predvolene 'legislation/unique_dates_counts.csv')
        
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
    
    # Explicitné nastavenie frekvencie (riešenie pre ValueWarning)
    df = df.asfreq('D', method='pad')  # 'D' pre denné údaje, method='pad' pre vyplnenie chýbajúcich hodnôt
    
    # Zobrazenie základných informácií o dátach
    print("\nData Overview:")
    print(f"Date Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"Number of observations: {len(df)}")
    print(f"Data frequency: {pd.infer_freq(df.index) or 'D'}")
    
    return df

def test_stationarity(series, title=''):
    """
    Testovanie stacionarity časového radu pomocou ADF testu
    
    Táto funkcia vykoná Augmented Dickey-Fuller test na overenie stacionarity
    časového radu a vytvorí vizualizáciu.
    
    Parametre:
    ----------
    series : pandas.Series
        Časový rad na testovanie
    title : str, voliteľný
        Titulok pre vizualizáciu
        
    Návratová hodnota:
    -----------------
    tuple
        (is_stationary, adf_result) - bool indikujúci stacionaritu a výsledky testu
    """
    print(f"\n=== Stationarity Tests {title} ===")
    
    # Vytvorenie vizualizácie časového radu a jeho štatistík
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    
    # Pôvodný časový rad
    ax1.plot(series)
    ax1.set_title(f'Raw Time Series {title}')
    ax1.set_xlabel('')
    
    # Kĺzavý priemer
    ax2.plot(series.rolling(window=7).mean(), label='7-day MA')
    ax2.plot(series.rolling(window=30).mean(), label='30-day MA')
    ax2.set_title('Moving Average')
    ax2.legend()
    
    # Kĺzavý štandardná odchýlka
    ax3.plot(series.rolling(window=7).std(), label='7-day SD')
    ax3.plot(series.rolling(window=30).std(), label='30-day SD')
    ax3.set_title('Rolling Standard Deviation')
    ax3.legend()
    
    # Autokorelačná funkcia
    pd.plotting.autocorrelation_plot(series, ax=ax4)
    ax4.set_title('Autocorrelation Function')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/stationarity_test_{title.replace(' ', '_')}.png")
    
    # ADF Test
    adf_result = adfuller(series.dropna())
    adf_output = pd.Series(
        [adf_result[0], adf_result[1], adf_result[4]['1%'], adf_result[4]['5%'], adf_result[4]['10%']],
        index=['Test Statistic', 'p-value', '1% Critical Value', '5% Critical Value', '10% Critical Value']
    )
    print("\nAugmented Dickey-Fuller Test:")
    print(adf_output)
    is_stationary = adf_result[1] < 0.05
    print(f"Conclusion: Series is {'stationary' if is_stationary else 'non-stationary'} according to ADF test")
    
    return is_stationary, adf_result

def seasonal_decomposition_analysis(series, freq=7):
    """
    Rozklad časového radu na trendovú, sezónnu a reziduálnu zložku
    
    Táto funkcia vykonáva sezónny rozklad časového radu pomocou metódy 
    z balíka statsmodels a vizualizuje výsledky.
    
    Parametre:
    ----------
    series : pandas.Series
        Časový rad na analýzu
    freq : int, voliteľný
        Frekvencia sezónnosti (predvolene 7 pre týždennú sezónnosť)
        
    Návratová hodnota:
    -----------------
    DecomposeResult
        Objekt obsahujúci rozložené komponenty časového radu
    """
    print(f"\n=== Seasonal Decomposition (Frequency={freq}) ===")
    
    # Vykonanie sezónneho rozkladu
    decomposition = seasonal_decompose(series, model='additive', period=freq)
    
    # Vizualizácia výsledkov
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Original Series')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend Component')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal Component')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual Component')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/seasonal_decomposition_freq{freq}.png")
    
    return decomposition

def analyze_acf_pacf(series, lags=40):
    """
    Analýza autokorelačnej a parciálnej autokorelačnej funkcie
    
    Táto funkcia vytvára a vizualizuje ACF a PACF grafy na identifikáciu
    vhodných parametrov pre ARIMA model.
    
    Parametre:
    ----------
    series : pandas.Series
        Časový rad na analýzu
    lags : int, voliteľný
        Počet oneskorení na zobrazenie (predvolene 40)
    """
    print("\n=== ACF and PACF Analysis ===")
    
    # Vytvorenie ACF a PACF grafov
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plot_acf(series.dropna(), lags=lags, ax=ax1)
    ax1.set_title('Autocorrelation Function (ACF)')
    plot_pacf(series.dropna(), lags=lags, ax=ax2)
    ax2.set_title('Partial Autocorrelation Function (PACF)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/acf_pacf.png")

def fit_sarima_model(series, order, seasonal_order):
    """
    Trénovanie SARIMA modelu s danými parametrami
    
    Táto funkcia trénuje SARIMA model na dátach s danými parametrami
    a vracia natrénovaný model spolu s metrikami výkonu a testami významnosti parametrov.
    
    Parametre:
    ----------
    series : pandas.Series
        Časový rad na trénovanie
    order : tuple
        Parametre ARIMA (p, d, q)
    seasonal_order : tuple
        Sezónne parametre SARIMA (P, D, Q, S)
        
    Návratová hodnota:
    -----------------
    tuple
        (model_fit, diagnostics) - natrénovaný model a slovník diagnostických metrík
        vrátane významnosti parametrov
    """
    try:
        # Trénovanie modelu
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)
        
        # Základné metriky
        aic = model_fit.aic
        bic = model_fit.bic
        
        # Residuals analysis
        residuals = model_fit.resid
        residuals_mean = residuals.mean()
        residuals_var = residuals.var()
        
        # Ljung-Box test
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_pvalue = lb_test['lb_pvalue'].values[0]
        
        # Jarque-Bera normality test
        jb_test = stats.jarque_bera(residuals)
        jb_pvalue = jb_test[1]
        
        # Extrakcia informácií o parametroch modelu priamo z objektu modelu
        params = model_fit.params
        conf_int = model_fit.conf_int()
        p_values = model_fit.pvalues
        std_errors = model_fit.bse
        t_values = model_fit.tvalues
        
        # Vytvorenie slovníka s významami parametrov
        param_significance = {
            'parameters': list(params.index),
            'coefficients': list(params.values),
            'std_errors': list(std_errors.values),
            't_values': list(t_values.values),
            'p_values': list(p_values.values),
            'conf_int_lower': list(conf_int.iloc[:, 0].values),
            'conf_int_upper': list(conf_int.iloc[:, 1].values)
        }
        
        # Hodnotenie významnosti parametrov
        significant_params = []
        non_significant_params = []
        for i, param in enumerate(param_significance['parameters']):
            if param_significance['p_values'][i] < 0.05:
                significant_params.append(param)
            else:
                non_significant_params.append(param)
        
        diagnostics = {
            'aic': aic,
            'bic': bic,
            'residuals_mean': residuals_mean,
            'residuals_var': residuals_var,
            'lb_pvalue': lb_pvalue,  # P-hodnota Ljung-Box testu
            'jb_pvalue': jb_pvalue,  # P-hodnota Jarque-Bera testu
            'residuals': residuals,  # pre ďalšiu analýzu
            'param_significance': param_significance,  # informácie o významnosti parametrov
            'significant_params': significant_params,  # štatisticky významné parametre (p < 0.05)
            'non_significant_params': non_significant_params  # štatisticky nevýznamné parametre (p >= 0.05)
        }
        
        return model_fit, diagnostics
    
    except Exception as e:
        print(f"Error fitting model with order={order}, seasonal_order={seasonal_order}: {e}")
        return None, None

def auto_sarima(series, seasonal=True, m=7):
    """
    Automaticky nájsť najlepšie parametre SARIMA modelu
    
    Táto funkcia používa pmdarima (auto_arima) na automatické nájdenie 
    optimálnych parametrov SARIMA modelu.
    
    Parametre:
    ----------
    series : pandas.Series
        Časový rad na analýzu
    seasonal : bool, voliteľný
        Či zahrnúť sezónnu zložku (predvolene True)
    m : int, voliteľný
        Sezónna perióda (predvolene 7 pre týždennú sezónnosť)
        
    Návratová hodnota:
    -----------------
    tuple
        (best_model, best_order, best_seasonal_order) - najlepší model a jeho parametre
    """
    print(f"\n=== Auto SARIMA Analysis (seasonal={seasonal}, m={m}) ===")
    
    # Automatický výber parametrov
    auto_model = pm.auto_arima(
        series,
        start_p=0, start_q=0,
        max_p=5, max_q=5, max_d=2,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2, max_D=1,
        m=m,
        seasonal=seasonal,
        d=None,
        D=None,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    
    print(f"Best SARIMA model: {auto_model.order} x {auto_model.seasonal_order}")
    print(f"AIC: {auto_model.aic()}")
    
    return auto_model, auto_model.order, auto_model.seasonal_order

def split_data(series, test_size=0.2):
    """
    Rozdelenie dát na trénovaciu a testovaciu množinu
    
    Táto funkcia rozdeľuje časový rad na trénovaciu a testovaciu množinu
    podľa zadaného pomeru.
    
    Parametre:
    ----------
    series : pandas.Series
        Časový rad na rozdelenie
    test_size : float, voliteľný
        Podiel dát na testovaciu množinu (predvolene 0.2 = 20%)
        
    Návratová hodnota:
    -----------------
    tuple
        (train, test) - trénovacia a testovacia množina
    """
    n = len(series)
    split_idx = int(n * (1 - test_size))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    
    print(f"\n=== Data Split ===")
    print(f"Training set: {len(train)} observations ({train.index.min().strftime('%Y-%m-%d')} to {train.index.max().strftime('%Y-%m-%d')})")
    print(f"Test set: {len(test)} observations ({test.index.min().strftime('%Y-%m-%d')} to {test.index.max().strftime('%Y-%m-%d')})")
    
    return train, test

def evaluate_model(predictions, actual):
    """
    Výpočet metrík pre vyhodnotenie modelu
    
    Táto funkcia počíta rôzne metriky chýb na vyhodnotenie presnosti modelu.
    
    Parametre:
    ----------
    predictions : array-like
        Predikované hodnoty modelu
    actual : array-like
        Skutočné hodnoty
        
    Návratová hodnota:
    -----------------
    dict
        Slovník obsahujúci vypočítané metriky
    """
    # Výpočet metrík
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predictions)
    
    # Výpočet R²
    ss_total = np.sum((actual - np.mean(actual))**2)
    ss_residual = np.sum((actual - predictions)**2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    
    # Výpočet MASE
    naive_errors = np.abs(np.diff(actual))
    if len(naive_errors) > 0 and np.sum(naive_errors) > 0:
        naive_mae = np.mean(naive_errors)
        mase = mae / naive_mae
    else:
        mase = np.nan
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r_squared,
        'MASE': mase
    }

def visualize_parameter_significance(model_fit, diagnostics):
    """
    Vizualizácia významnosti parametrov modelu
    
    Táto funkcia vizualizuje štatistickú významnosť parametrov modelu 
    vrátane koeficientov, p-hodnôt a intervalov spoľahlivosti.
    
    Parametre:
    ----------
    model_fit : statsmodels.SARIMAXResults
        Natrénovaný SARIMA model
    diagnostics : dict
        Slovník diagnostických metrík vrátane údajov o významnosti parametrov
    """
    print("\n=== Parameter Significance Analysis ===")
    
    # Extrakcia údajov o významnosti parametrov
    param_significance = diagnostics['param_significance']
    significant_params = diagnostics['significant_params']
    non_significant_params = diagnostics['non_significant_params']
    
    # Výpis významnosti parametrov
    print("\nStatistically Significant Parameters (p < 0.05):")
    if significant_params:
        for param in significant_params:
            idx = param_significance['parameters'].index(param)
            coef = param_significance['coefficients'][idx]
            p_val = param_significance['p_values'][idx]
            t_val = param_significance['t_values'][idx]
            print(f"{param:<20}: coef = {coef:.4f}, t = {t_val:.4f}, p-value = {p_val:.4f}")
    else:
        print("None")
    
    print("\nNon-Significant Parameters (p >= 0.05):")
    if non_significant_params:
        for param in non_significant_params:
            idx = param_significance['parameters'].index(param)
            coef = param_significance['coefficients'][idx]
            p_val = param_significance['p_values'][idx]
            t_val = param_significance['t_values'][idx]
            print(f"{param:<20}: coef = {coef:.4f}, t = {t_val:.4f}, p-value = {p_val:.4f}")
    else:
        print("None")
    
    # Vizualizácia koeficientov a ich intervalov spoľahlivosti
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Koeficienty a intervaly spoľahlivosti
    params = param_significance['parameters']
    coefs = param_significance['coefficients']
    lower_ci = param_significance['conf_int_lower']
    upper_ci = param_significance['conf_int_upper']
    p_values = param_significance['p_values']
    
    # Farebné rozlíšenie významných a nevýznamných parametrov
    colors = ['blue' if p < 0.05 else 'red' for p in p_values]
    
    # Graf koeficientov s intervalmi spoľahlivosti
    y_pos = np.arange(len(params))
    
    # Vykreslenie každého parametra samostatne s príslušnou farbou
    for i, (param, coef, p_val) in enumerate(zip(params, coefs, p_values)):
        color = 'blue' if p_val < 0.05 else 'red'
        xerr = np.array([[coef - lower_ci[i]], [upper_ci[i] - coef]])
        ax1.errorbar(coef, y_pos[i], xerr=xerr, fmt='o', capsize=5, color=color, ecolor=color)
    
    # Pridanie vertikálnej čiary na nule
    ax1.axvline(x=0, color='gray', linestyle='--')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(params)
    ax1.set_xlabel('Coefficient Value')
    ax1.set_title('Model Parameters with 95% Confidence Intervals')
    
    # Vysvetlivka farieb
    blue_patch = plt.Rectangle((0, 0), 1, 1, color='blue')
    red_patch = plt.Rectangle((0, 0), 1, 1, color='red')
    ax1.legend([blue_patch, red_patch], ['Significant (p < 0.05)', 'Non-significant (p >= 0.05)'])
    
    # Graf p-hodnôt
    ax2.bar(y_pos, p_values, color=colors)
    ax2.axhline(y=0.05, color='gray', linestyle='--', label='Significance Level (0.05)')
    ax2.set_xticks(y_pos)
    ax2.set_xticklabels(params, rotation=45, ha='right')
    ax2.set_ylabel('p-value')
    ax2.set_title('Parameter p-values (Lower is More Significant)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/parameter_significance.png")
    
    # Výpis tabuľky s úplným súhrnom parametrov
    print("\nFull Parameter Summary:")
    print(model_fit.summary().tables[1])

def analyze_residuals(model_fit, diagnostics=None):
    """
    Analýza reziduálov modelu
    
    Táto funkcia analyzuje a vizualizuje reziduály modelu na overenie 
    predpokladov o chybách modelu a významnosť parametrov modelu.
    
    Parametre:
    ----------
    model_fit : statsmodels.SARIMAXResults
        Natrénovaný SARIMA model
    diagnostics : dict, voliteľný
        Slovník diagnostických metrík vrátane údajov o významnosti parametrov
    """
    print("\n=== Residual Analysis ===")
    
    residuals = model_fit.resid
    
    # Vizualizácia reziduálov
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Časový rad reziduálov
    ax1.plot(residuals)
    ax1.set_title('Residuals Time Series')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Residual Value')
    
    # Histogram a normálna distribúcia
    residuals.plot(kind='hist', density=True, bins=20, ax=ax2)
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    ax2.plot(x, p, 'k', linewidth=2)
    ax2.set_title('Residuals Distribution')
    
    # QQ plot
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot')
    
    # ACF reziduálov
    plot_acf(residuals.dropna(), lags=40, ax=ax4)
    ax4.set_title('Residuals ACF')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/residual_analysis.png")
    
    # Ljung-Box test
    lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
    print("\nLjung-Box Test for Autocorrelation:")
    print(lb_test)
    
    # Jarque-Bera normality test
    jb_test = stats.jarque_bera(residuals)
    print("\nJarque-Bera Normality Test:")
    print(f"Statistic: {jb_test[0]}")
    print(f"P-value: {jb_test[1]}")
    print(f"Residuals are {'normally distributed' if jb_test[1] > 0.05 else 'not normally distributed'}")
    
    # Ak sú k dispozícii diagnostické údaje, analyzujeme aj významnosť parametrov
    if diagnostics and 'param_significance' in diagnostics:
        visualize_parameter_significance(model_fit, diagnostics)

def forecast_future(model_fit, steps=30, original_series=None):
    """
    Predikcia budúcich hodnôt pomocou SARIMA modelu
    
    Táto funkcia predikuje budúce hodnoty pomocou natrénovaného modelu
    a vizualizuje výsledky.
    
    Parametre:
    ----------
    model_fit : statsmodels.SARIMAXResults
        Natrénovaný SARIMA model
    steps : int, voliteľný
        Počet krokov do budúcnosti na predikciu (predvolene 30)
    original_series : pandas.Series, voliteľný
        Pôvodný časový rad na porovnanie s predikciami
        
    Návratová hodnota:
    -----------------
    pandas.Series
        Časový rad s predikovanými hodnotami
    """
    print(f"\n=== Forecasting {steps} Steps Ahead ===")
    
    # Predikcia
    forecast_result = model_fit.get_forecast(steps=steps)
    forecast_index = pd.date_range(
        start=original_series.index[-1] + pd.Timedelta(days=1),
        periods=steps,
        freq='D'
    )
    
    # Hodnoty predikcie a intervaly spoľahlivosti
    forecast_mean = forecast_result.predicted_mean
    forecast_mean.index = forecast_index
    
    conf_int = forecast_result.conf_int()
    conf_int.index = forecast_index
    
    # Vizualizácia predikcie
    plt.figure(figsize=(14, 7))
    if original_series is not None:
        plt.plot(original_series, label='Historical Data')
    
    plt.plot(forecast_mean, color='red', label='Forecast')
    plt.fill_between(
        conf_int.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        color='pink', alpha=0.3
    )
    
    plt.title('SARIMA Forecast with 95% Confidence Interval')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/forecast_{steps}_days.png")
    
    return forecast_mean

def generate_markdown_report(data_info, model_info, metrics, forecast_info):
    """
    Generovanie markdown reportu s výsledkami analýzy
    
    Táto funkcia vytvára podrobný markdown report s výsledkami analýzy,
    vrátane matematického popisu modelu SARIMA.
    
    Parametre:
    ----------
    data_info : dict
        Informácie o analyzovaných dátach
    model_info : dict
        Informácie o modeli (parametre, diagnostika)
    metrics : dict
        Metriky výkonu modelu
    forecast_info : dict
        Informácie o predikcii
        
    Návratová hodnota:
    -----------------
    str
        Obsah markdown reportu
    """
    # Extrakcia parametrov modelu
    p, d, q = model_info['order']
    P, D, Q, m = model_info['seasonal_order']
    
    report = f"""# SARIMA Model Analysis Report

## 1. Overview

**Date Range:** {data_info['date_range']}
**Number of Observations:** {data_info['num_observations']}
**Data Frequency:** {data_info['frequency']}

## 2. Model Specification

The Seasonal ARIMA (SARIMA) model is a time series forecasting method that incorporates seasonality. 

### Mathematical Notation

The SARIMA model is denoted as SARIMA(p,d,q)(P,D,Q)m where:
- p = {p}: Order of the non-seasonal autoregressive (AR) terms
- d = {d}: Order of non-seasonal differencing
- q = {q}: Order of the non-seasonal moving average (MA) terms
- P = {P}: Order of the seasonal autoregressive terms
- D = {D}: Order of seasonal differencing
- Q = {Q}: Order of the seasonal moving average terms
- m = {m}: Seasonal period

The mathematical form of the SARIMA model is:

$$\\Phi_P(B^m)\\phi_p(B)(1-B)^d(1-B^m)^D y_t = \\alpha + \\Theta_Q(B^m)\\theta_q(B)\\varepsilon_t$$

Where:
- $\\phi_p(B)$: Non-seasonal AR operator of order p
- $\\Phi_P(B^m)$: Seasonal AR operator of order P
- $(1-B)^d$: Non-seasonal differencing of order d
- $(1-B^m)^D$: Seasonal differencing of order D
- $\\theta_q(B)$: Non-seasonal MA operator of order q
- $\\Theta_Q(B^m)$: Seasonal MA operator of order Q
- $\\varepsilon_t$: White noise error term
- $B$: Backshift operator
- $\\alpha$: Constant term

## 3. Model Diagnostics

**AIC:** {model_info['aic']:.4f}
**BIC:** {model_info['bic']:.4f}

**Residual Analysis:**
- Mean of Residuals: {model_info['residuals_mean']:.4f}
- Variance of Residuals: {model_info['residuals_var']:.4f}
- Ljung-Box Test p-value: {model_info['lb_pvalue']:.4f} (> 0.05 indicates no significant autocorrelation in residuals)
- Jarque-Bera Test p-value: {model_info['jb_pvalue']:.4f} (> 0.05 indicates residuals are normally distributed)

## 4. Model Performance Metrics

### In-Sample Performance (Training Data)
- MSE: {metrics['train']['MSE']:.4f}
- RMSE: {metrics['train']['RMSE']:.4f}
- MAE: {metrics['train']['MAE']:.4f}
- R²: {metrics['train']['R²']:.4f}
- MASE: {metrics['train']['MASE']:.4f}

### Out-of-Sample Performance (Test Data)
- MSE: {metrics['test']['MSE']:.4f}
- RMSE: {metrics['test']['RMSE']:.4f}
- MAE: {metrics['test']['MAE']:.4f}
- R²: {metrics['test']['R²']:.4f}
- MASE: {metrics['test']['MASE']:.4f}

## 5. Forecasting

Forecast period: {forecast_info['steps']} days
Start date: {forecast_info['start_date']}
End date: {forecast_info['end_date']}

## 6. Conclusion

Based on the model performance metrics and diagnostics, the SARIMA({p},{d},{q})({P},{D},{Q}){m} model {metrics['conclusion']} for forecasting legislative changes. The model captures {metrics['captures']} and can be used {metrics['usage']}.

## 7. Visualizations

Multiple visualizations were generated during this analysis and stored in the '{output_dir}' directory:
- Stationarity tests
- Seasonal decomposition
- ACF and PACF plots
- Residual analysis
- Forecast with confidence intervals
"""
    
    # Save the report to a file
    report_path = f"{output_dir}/sarima_model_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to {report_path}")
    
    return report

def main():
    """
    Hlavná funkcia na spustenie SARIMA analýzy
    
    Táto funkcia koordinuje celý proces analýzy časového radu:
    1. Načítanie a prípravu dát
    2. Testovanie stacionarity a sezónny rozklad
    3. Rozdelenie dát na trénovacie a testovacie množiny
    4. Automatické hľadanie parametrov SARIMA modelu
    5. Trénovanie a vyhodnotenie modelu
    6. Analýzu reziduálov a diagnostiku modelu
    7. Predikcie do budúcnosti
    8. Generovanie markdown reportu
    """
    # 1. Načítanie dát
    df = load_data()
    series = df['Count']
    
    # Základné informácie o dátach pre report
    data_info = {
        'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
        'num_observations': len(df),
        'frequency': 'Daily'
    }
    
    # 2. Analýza dát
    is_stationary, _ = test_stationarity(series, title='Original')
    
    # Skúmanie sezónnosti - použitie týždenného vzoru (7 dní) na základe predchádzajúcej analýzy
    # Výsledky predchádzajúcej analýzy ukázali, že týždenná sezónnosť (m=7) je najlepšia pre tieto dáta
    decomposition = seasonal_decomposition_analysis(series, freq=7)
    
    # ACF a PACF analýza
    analyze_acf_pacf(series)
    
    # 3. Rozdelenie dát na trénovacie a testovacie množiny
    train, test = split_data(series, test_size=0.2)
    
    # 4. Automatické hľadanie parametrov SARIMA modelu
    # Na základe predchádzajúceho výskumu vieme, že týždenná sezónnosť (m=7) je najvhodnejšia
    auto_model, best_order, best_seasonal_order = auto_sarima(train, seasonal=True, m=7)
    
    # 5. Trénovanie a vyhodnotenie modelu
    print(f"\n=== Training SARIMA Model with Best Parameters ===")
    print(f"Order: {best_order}, Seasonal Order: {best_seasonal_order}")
    
    best_model_fit, diagnostics = fit_sarima_model(train, best_order, best_seasonal_order)
    
    if best_model_fit is not None:
        # Predikcia na trénovacích dátach
        train_predictions = best_model_fit.predict()
        train_metrics = evaluate_model(train_predictions, train)
        print("\nTraining Set Metrics:")
        for metric, value in train_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Predikcia na testovacích dátach
        test_predictions = best_model_fit.get_forecast(steps=len(test)).predicted_mean
        test_predictions.index = test.index
        test_metrics = evaluate_model(test_predictions, test)
        print("\nTest Set Metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Vizualizácia predikcií na testovacích dátach
        plt.figure(figsize=(14, 7))
        plt.plot(train.index, train, label='Training Data')
        plt.plot(test.index, test, label='Test Data')
        plt.plot(test.index, test_predictions, color='red', label='Predictions')
        plt.title('SARIMA Model Test Set Predictions')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/test_predictions.png")
        
        # 6. Analýza reziduálov a diagnostika modelu vrátane významnosti parametrov
        analyze_residuals(best_model_fit, diagnostics)
        
        # 7. Trénovanie finálneho modelu na všetkých dátach a predikcia do budúcnosti
        print("\n=== Training Final Model on Full Dataset ===")
        final_model_fit, _ = fit_sarima_model(series, best_order, best_seasonal_order)
        
        if final_model_fit is not None:
            # Predikcia 30 dní do budúcnosti
            forecast_steps = 30
            forecast_series = forecast_future(final_model_fit, steps=forecast_steps, original_series=series)
            
            forecast_info = {
                'steps': forecast_steps,
                'start_date': forecast_series.index.min().strftime('%Y-%m-%d'),
                'end_date': forecast_series.index.max().strftime('%Y-%m-%d')
            }
            
            # Určenie kvality modelu pre záver reportu
            conclusion_quality = 'is suitable'
            if test_metrics['R²'] < 0.3:
                conclusion_quality = 'may have limited predictive power'
            elif test_metrics['R²'] > 0.6:
                conclusion_quality = 'is highly suitable'
            
            metrics_for_report = {
                'train': train_metrics,
                'test': test_metrics,
                'conclusion': conclusion_quality,
                'captures': 'the weekly seasonality pattern in legislative changes',
                'usage': 'for short to medium-term forecasting of legislative activity'
            }
            
            # 8. Generovanie markdown reportu
            model_info = {
                'order': best_order,
                'seasonal_order': best_seasonal_order,
                'aic': final_model_fit.aic,
                'bic': final_model_fit.bic,
                'residuals_mean': final_model_fit.resid.mean(),
                'residuals_var': final_model_fit.resid.var(),
                'lb_pvalue': acorr_ljungbox(final_model_fit.resid, lags=[10], return_df=True)['lb_pvalue'].values[0],
                'jb_pvalue': stats.jarque_bera(final_model_fit.resid)[1]
            }
            
            generate_markdown_report(data_info, model_info, metrics_for_report, forecast_info)
            
            print("\nAnalysis complete!")
            print(f"Results saved in {output_dir} directory")
    else:
        print("Failed to train SARIMA model with selected parameters.")

if __name__ == "__main__":
    main()
