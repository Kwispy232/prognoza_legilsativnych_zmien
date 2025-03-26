#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Time Series Analysis for Legislative Changes
"""
# Komplexná analýza časových radov pre legislatívne zmeny
# Importujeme potrebné knižnice:
# - pandas a numpy pre manipuláciu s dátami
# - matplotlib a seaborn pre vizualizáciu
# - statsmodels pre štatistickú analýzu časových radov
# - scipy pre pokročilé štatistické transformácie
# - os a datetime pre prácu so súbormi a dátumami
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import boxcox
import os
from datetime import datetime, timedelta

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create output directory for visualizations
output_dir = 'visualizations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Funkcia na získanie alebo generovanie dát
# Ak súbor existuje, načíta ho
# Ak neexistuje, vygeneruje vzorové dáta s trendom, sezónnosťou a šumom
# - Trend: lineárny nárast od 10 do 30
# - Mesačná sezónnosť: sínusový vzor s amplitúdou 5
# - Štvrťročná sezónnosť: sínusový vzor s amplitúdou 10
# - Náhodný šum: normálne rozdelenie s priemerom 0 a štandardnou odchýlkou 3
def get_or_generate_data(filename='unique_dates_counts.csv', n_samples=353):
    if os.path.exists(filename):
        print(f"Načítavam existujúci dataset: {filename}")
        df = pd.read_csv(filename)
        # Ensure date column is properly formatted
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    else:
        print(f"Dataset {filename} nenájdený. Generujem vzorové dáta...")
        # Generate sample time series data with trend, seasonality and noise
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_samples)]
        
        # Create trend component
        trend = np.linspace(10, 30, n_samples)
        
        # Create seasonal component (monthly pattern)
        seasonal = 5 * np.sin(np.linspace(0, 2 * np.pi * (n_samples/365) * 12, n_samples))
        
        # Create quarterly stronger seasonality
        quarterly = 10 * np.sin(np.linspace(0, 2 * np.pi * (n_samples/365) * 4, n_samples))
        
        # Add random noise
        noise = np.random.normal(0, 3, n_samples)
        
        # Combine components
        counts = trend + seasonal + quarterly + noise
        
        # Ensure counts are positive integers
        counts = np.round(np.maximum(counts, 1)).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Count': counts
        })
        
        # Save the generated dataset
        df.to_csv(filename, index=False)
        print(f"Sample dataset generated and saved as {filename}")
        return df

# Funkcia na vykonanie ADF testu stacionarity
# Stacionarita znamená, že štatistické vlastnosti časového radu (priemer, rozptyl) 
# sa v čase nemenia
# Nulová hypotéza: časový rad je nestacionárny
# Ak p-hodnota < 0.05, zamietame nulovú hypotézu a považujeme rad za stacionárny
def adf_test(timeseries):
    print("Výsledky Augmented Dickey-Fuller testu:")
    result = adfuller(timeseries.dropna())
    
    # Formátovanie výsledkov
    output = pd.Series(
        [result[0], result[1], result[4]['1%'], result[4]['5%'], result[4]['10%']],
        index=['Testová štatistika', 'p-hodnota', '1% kritická hodnota', '5% kritická hodnota', '10% kritická hodnota']
    )
    
    # Určenie, či je časový rad stacionárny
    is_stationary = result[1] < 0.05
    
    print(output)
    print(f"Záver: Časový rad je {'stacionárny' if is_stationary else 'nestacionárny'}")
    return is_stationary

# Funkcia na výpočet sily sezónnosti
# Sila sezónnosti sa počíta ako pomer rozptylu sezónnej zložky 
# k súčtu rozptylov sezónnej zložky a reziduálnej zložky
def calculate_seasonal_strength(decomposition):
    """Výpočet sily sezónnosti na základe výsledkov dekompozície"""
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    # Výpočet rozptylu
    var_seasonal = np.var(seasonal.dropna())
    var_residual = np.var(residual.dropna())
    
    # Calculate seasonal strength
    if var_seasonal + var_residual > 0:
        seasonal_strength = var_seasonal / (var_seasonal + var_residual)
    else:
        seasonal_strength = 0
    
    return seasonal_strength

# Funkcia na analýzu sezónnosti pre rôzne periódy
# Pre každú periódu vypočítame silu sezónnosti a určíme najsilnejšiu
def analyze_seasonality(timeseries, periods):
    results = {}
    
    for period in periods:
        if len(timeseries) > period * 2:  # Potrebujeme aspoň 2 úplné periódy pre dekompozíciu
            try:
                decomposition = seasonal_decompose(timeseries, model='additive', period=period)
                strength = calculate_seasonal_strength(decomposition)
                results[period] = {
                    'decomposition': decomposition,
                    'strength': strength
                }
                print(f"Perióda {period}: Sila sezónnosti = {strength:.4f}")
            except Exception as e:
                print(f"Nepodarilo sa rozložiť pre periódu {period}: {e}")
    
    # Nájdenie periódy s najsilnejšou sezónnosťou
    if results:
        strongest_period = max(results.keys(), key=lambda k: results[k]['strength'])
        print(f"Zistená najsilnejšia sezónna perióda: {strongest_period} dní")
        return results, strongest_period
    else:
        print("Nebola zistená významná sezónnosť")
        return results, None

# Funkcia na vytvorenie a uloženie vizualizácií
# 1. Pôvodný časový rad - zobrazuje surové počty legislatívnych zmien v čase
# 2. Kĺzavý priemer a štandardná odchýlka - ukazuje trend a volatilitu
# 3. Časový rad po prvej diferencii - odstránenie trendu pre dosiahnutie stacionarity
# 4. Autokorelačná funkcia (ACF) - ukazuje korelácie medzi pozorovaniami v rôznych časových oneskoreniach
# 5. Parciálna autokorelačná funkcia (PACF) - ukazuje priamy vzťah medzi pozorovaním a jeho oneskorením
def create_visualizations(df, output_dir):
    time_series = df.set_index('Date')['Count']
    
    # 1. Pôvodný časový rad
    plt.figure(figsize=(12, 6))
    plt.plot(time_series, color='blue')
    plt.title('Pôvodný časový rad: Legislatívne zmeny')
    plt.xlabel('Dátum')
    plt.ylabel('Počet')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/povodny_casovy_rad.png")
    plt.close()
    
    # 2. Kĺzavý priemer a štandardná odchýlka
    rolling_mean = time_series.rolling(window=30).mean()
    rolling_std = time_series.rolling(window=30).std()
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_series, color='blue', label='Pôvodný')
    plt.plot(rolling_mean, color='red', label='Kĺzavý priemer (30 dní)')
    plt.plot(rolling_std, color='green', label='Kĺzavá št. odchýlka (30 dní)')
    plt.title('Kĺzavý priemer a štandardná odchýlka')
    plt.xlabel('Dátum')
    plt.ylabel('Počet')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/klzave_statistiky.png")
    plt.close()
    
    # 3. Časový rad po prvej diferencii
    diff_series = time_series.diff().dropna()
    
    plt.figure(figsize=(12, 6))
    plt.plot(diff_series, color='purple')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Časový rad po prvej diferencii')
    plt.xlabel('Dátum')
    plt.ylabel('Diferencia')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/diferencovany_rad.png")
    plt.close()
    
    # 4. Autokorelačná funkcia
    plt.figure(figsize=(12, 6))
    plot_acf(time_series.dropna(), lags=50, alpha=0.05)
    plt.title('Autokorelačná funkcia (ACF)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/acf_graf.png")
    plt.close()
    
    # 5. Parciálna autokorelačná funkcia
    plt.figure(figsize=(12, 6))
    plot_pacf(time_series.dropna(), lags=50, alpha=0.05)
    plt.title('Parciálna autokorelačná funkcia (PACF)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pacf_graf.png")
    plt.close()
    
    return rolling_mean, rolling_std, diff_series

# Funkcia na transformáciu časového radu, ak je nestacionárny
# Možnosti transformácie:
# - Diferencia: odčítanie predchádzajúcej hodnoty od aktuálnej
# - Logaritmická transformácia: aplikácia prirodzeného logaritmu
# - Box-Cox transformácia: parametrická mocninná transformácia
def transform_series(time_series, transformation='difference'):
    if transformation == 'difference':
        transformed = time_series.diff().dropna()
        print("Applied first-order differencing")
    elif transformation == 'log':
        # Ensure all values are positive
        min_val = time_series.min()
        if min_val <= 0:
            offset = abs(min_val) + 1
            transformed = np.log(time_series + offset)
            print(f"Applied log transformation with offset {offset}")
        else:
            transformed = np.log(time_series)
            print("Applied log transformation")
    elif transformation == 'boxcox':
        # Ensure all values are positive
        min_val = time_series.min()
        if min_val <= 0:
            offset = abs(min_val) + 1
            transformed, lambda_val = boxcox(time_series + offset)
            transformed = pd.Series(transformed, index=time_series.index)
            print(f"Applied Box-Cox transformation with lambda={lambda_val:.4f} and offset {offset}")
        else:
            transformed, lambda_val = boxcox(time_series)
            transformed = pd.Series(transformed, index=time_series.index)
            print(f"Applied Box-Cox transformation with lambda={lambda_val:.4f}")
    else:
        transformed = time_series
        print("No transformation applied")
    
    return transformed

# Funkcia na odporúčanie metód prognózovania na základe výsledkov analýzy
# Ak je časový rad stacionárny a nemá silnú sezónnosť:
#   - ARIMA (Autoregresívny integrovaný kĺzavý priemer)
#   - Exponenciálne vyrovnávanie
# Ak je stacionárny a má sezónnosť:
#   - SARIMA (Sezónny ARIMA)
#   - Holt-Winters (Exponenciálne vyrovnávanie so sezónnosťou)
# Ak je nestacionárny a nemá silnú sezónnosť:
#   - ARIMA s diferenciáciou
#   - Prophet (Facebook/Meta)
# Ak je nestacionárny a má sezónnosť:
#   - SARIMA s diferenciáciou
#   - Prophet
# Pre komplexné vzory vždy zvážiť:
#   - LSTM neurónové siete
#   - Random Forest alebo XGBoost
def recommend_forecasting_methods(is_stationary, seasonal_periods, seasonal_strength):
    recommendations = []
    
    print("\nOdporúčania pre metódy prognózovania:")
    
    if is_stationary:
        print("✓ Časový rad je stacionárny")
        if not seasonal_periods or all(strength < 0.3 for period, strength in seasonal_strength.items()):
            print("✓ Nezistená výrazná sezónnosť")
            recommendations.append({
                'method': 'ARIMA',
                'reason': 'Vhodná pre stacionárne časové rady bez výraznej sezónnosti',
                'priority': 'Vysoká'
            })
            recommendations.append({
                'method': 'Exponenciálne vyrovnávanie',
                'reason': 'Funguje dobre so stacionárnymi dátami a dokáže sa prispôsobiť menším výkyvom',
                'priority': 'Stredná'
            })
        else:
            print(f"✓ Zistená sezónnosť s periódou/periódami: {list(seasonal_periods.keys())}")
            recommendations.append({
                'method': 'SARIMA',
                'reason': f'Dokáže spracovať stacionárne dáta aj sezónnosť (perióda={max(seasonal_periods.keys())})',
                'priority': 'Vysoká'
            })
            recommendations.append({
                'method': 'Holt-Winters',
                'reason': 'Metóda exponenciálneho vyrovnávania, ktorá zohľadňuje sezónnosť',
                'priority': 'Stredná'
            })
    else:
        print("✓ Časový rad je nestacionárny")
        if not seasonal_periods or all(strength < 0.3 for period, strength in seasonal_strength.items()):
            print("✓ Nezistená výrazná sezónnosť")
            recommendations.append({
                'method': 'ARIMA s diferenciou',
                'reason': 'Dokáže spracovať nestacionárne dáta pomocou diferencovania',
                'priority': 'Vysoká'
            })
            recommendations.append({
                'method': 'Prophet',
                'reason': 'Robustná voči chýbajúcim dátam a posunom v trende, dobre zvláda nestacionárne dáta',
                'priority': 'Stredná'
            })
        else:
            print(f"✓ Zistená sezónnosť s periódou/periódami: {list(seasonal_periods.keys())}")
            recommendations.append({
                'method': 'SARIMA s diferenciou',
                'reason': f'Zvláda nestacionárne dáta aj sezónnosť (perióda={max(seasonal_periods.keys())})',
                'priority': 'Vysoká'
            })
            recommendations.append({
                'method': 'Prophet',
                'reason': 'Automaticky detekuje sezónnosť a zvláda nestacionárne dáta',
                'priority': 'Vysoká'
            })
    
    # Always consider machine learning approaches for complex patterns
    recommendations.append({
        'method': 'LSTM neurónové siete',
        'reason': 'Dokáže zachytiť komplexné nelineárne vzory v dátach',
        'priority': 'Stredná' if len(recommendations) < 2 else 'Nízka'
    })
    recommendations.append({
        'method': 'Random Forest alebo XGBoost',
        'reason': 'Ensemblové metódy, ktoré dokážu zachytiť nelineárne vzťahy',
        'priority': 'Stredná' if len(recommendations) < 2 else 'Nízka'
    })
    
    return recommendations

# Funkcia na generovanie podrobnej správy o analýze
# Obsahuje:
# - Súhrn datasetu (počet pozorovaní, rozsah dátumov, štatistiky)
# - Výsledky testu stacionarity
# - Analýzu sezónnosti a jej silu
# - Interpretáciu vizualizácií
# - Odporúčania pre metódy prognózovania
# - Obmedzenia a ďalšie úvahy
def generate_report(df, is_stationary, seasonal_results, strongest_period, recommendations, transformations_tried=None):
    report = "# Komplexná analýza časového radu\n\n"
    
    # Súhrn dát
    report += "## Súhrn dát\n"
    report += f"- Celkový počet pozorovaní: {len(df)}\n"
    report += f"- Rozsah dátumov: {df['Date'].min().date()} až {df['Date'].max().date()}\n"
    report += f"- Priemerný počet: {df['Count'].mean():.2f}\n"
    report += f"- Minimálny počet: {df['Count'].min()}\n"
    report += f"- Maximálny počet: {df['Count'].max()}\n"
    report += f"- Štandardná odchýlka: {df['Count'].std():.2f}\n\n"
    
    # Výsledky testu stacionarity
    report += "## Analýza stacionarity\n"
    report += f"- **Výsledok ADF testu**: Časový rad je {'stacionárny' if is_stationary else 'nestacionárny'}\n"
    
    if transformations_tried:
        report += "- **Vyskúšané transformácie**:\n"
        for transform, result in transformations_tried.items():
            transform_name = {
                'First-order differencing': 'Prvá diferencia',
                'Log transformation': 'Logaritmická transformácia',
                'Box-Cox transformation': 'Box-Cox transformácia'
            }.get(transform, transform)
            report += f"  - {transform_name}: {'Úspešná' if result else 'Neúspešná'} v dosiahnutí stacionarity\n"
    
    if not is_stationary:
        report += "- **Odporúčané transformácie**:\n"
        report += "  - Prvá diferencia\n"
        report += "  - Logaritmická transformácia\n"
        report += "  - Box-Cox transformácia\n\n"
    else:
        report += "\n"
    
    # Analýza sezónnosti
    report += "## Analýza sezónnosti\n"
    
    if seasonal_results:
        report += "- **Zistené sezónne periódy**:\n"
        for period, result in seasonal_results.items():
            strength = result['strength']
            strength_level = "Silná" if strength > 0.6 else "Stredná" if strength > 0.3 else "Slabá"
            report += f"  - {period} dní: {strength_level} sezónnosť (sila = {strength:.4f})\n"
        
        if strongest_period:
            report += f"- **Najsilnejšia sezónna perióda**: {strongest_period} dní\n\n"
    else:
        report += "- Nebola zistená významná sezónnosť\n\n"
    
    # Interpretácia vizualizácií
    report += "## Interpretácia vizualizácií\n"
    report += "- **Pôvodný časový rad**: "
    report += "Zobrazuje surový počet legislatívnych zmien v čase, odhaľuje celkový vzor a potenciálne odchodýlky.\n"
    
    report += "- **Kĺzavé štatistiky**: "
    report += "Kĺzavý priemer indikuje trendovú zložku, zatáľ čo kĺzavá štandardná odchýlka ukazuje volatilitu v čase.\n"
    
    report += "- **Diferencovaný rad**: "
    report += "Prvá diferencia odstraňuje trendovú zložku, čo pomáha dosiahnuť stacionaritu. "
    report += "Hodnoty kolisajúce okolo nuly indikujú úspešné odstránenie trendu.\n"
    
    report += "- **ACF graf**: "
    report += "Autokorelačná funkcia zobrazuje korelácie medzi pozorovaniami v rôznych časových oneskoreniach. "
    report += "Významné vrcholy v pravidelných intervaloch naznačujú sezónnosť.\n"
    
    report += "- **PACF graf**: "
    report += "Parciálna autokorelačná funkcia pomáha identifikovať rád autoregresného modelu. "
    report += "Zobrazuje priamy vzťah medzi pozorovaním a jeho oneskorením.\n\n"
    
    # Odporúčania pre prognózovanie
    report += "## Odporúčania pre metódy prognózovania\n"
    
    for rec in recommendations:
        report += f"- **{rec['method']}** (Priorita: {rec['priority']})\n"
        report += f"  - Zdôvodnenie: {rec['reason']}\n"
    
    report += "\n"
    
    # Obmedzenia a úvahy
    report += "## Obmedzenia a ďalšie úvahy\n"
    report += "- Analýza predpokladá, že minulé vzory budú pokračovať aj v budúcnosti.\n"
    report += "- Externé faktory ovplyvňujúce legislatívne zmeny (napr. politické udalosti, voľby) nie sú v tejto analýze zohľadnené.\n"
    report += "- Sila sezónnosti sa môže časom meniť, čo si vyžaduje pravidelné prehodnotenie.\n"
    report += "- Pre optimálne prognózovanie by mohli byť zahrnuté dodatočné externé premenné ako príznaky.\n"
    
    return report

# Hlavná funkcia, ktorá spúšťa celú analýzu
# 1. Získa alebo vygeneruje dáta
# 2. Zobrazí základné štatistiky
# 3. Vytvorí vizualizácie
# 4. Testuje stacionaritu
# 5. Ak nie je stacionárny, skúša transformácie
# 6. Analyzuje sezónnosť
# 7. Vytvorí dekompozíciu sezónnosti pre najsilnejšiu periódu
# 8. Odporúča metódy prognózovania
# 9. Generuje a ukladá správu
def main():
    print("Začínam komplexnú analýzu časového radu legislatívnych zmien")
    
    # Získanie alebo generovanie dát
    df = get_or_generate_data()
    print(f"Rozmery datasetu: {df.shape}")
    
    # Zobrazenie základných štatistík
    print("\nZákladné štatistiky:")
    print(df.describe())
    
    # Nastavenie dátumu ako indexu pre analýzu časového radu
    time_series = df.set_index('Date')['Count']
    
    # Vytvorenie vizualizácií
    print("\nGenerujem vizualizácie...")
    rolling_mean, rolling_std, diff_series = create_visualizations(df, output_dir)
    
    # Testovanie stacionarity
    print("\nTestujem stacionaritu...")
    is_stationary = adf_test(time_series)
    
    # Ak nie je stacionárny, skúšame transformácie
    transformations_tried = {}
    if not is_stationary:
        print("\nTrying transformations to achieve stationarity:")
        
        # Try differencing
        print("\nTesting first-order differencing:")
        diff_stationary = adf_test(diff_series)
        transformations_tried['First-order differencing'] = diff_stationary
        
        # Try log transformation
        print("\nTesting log transformation:")
        # Ensure all values are positive for log transform
        min_val = time_series.min()
        if min_val <= 0:
            offset = abs(min_val) + 1
            log_series = np.log(time_series + offset)
        else:
            log_series = np.log(time_series)
        log_stationary = adf_test(log_series)
        transformations_tried['Log transformation'] = log_stationary
        
        # Try Box-Cox transformation
        print("\nTesting Box-Cox transformation:")
        try:
            # Ensure all values are positive for Box-Cox
            if min_val <= 0:
                boxcox_series, _ = boxcox(time_series + offset)
            else:
                boxcox_series, _ = boxcox(time_series)
            boxcox_series = pd.Series(boxcox_series, index=time_series.index)
            boxcox_stationary = adf_test(boxcox_series)
            transformations_tried['Box-Cox transformation'] = boxcox_stationary
        except Exception as e:
            print(f"Box-Cox transformation failed: {e}")
            transformations_tried['Box-Cox transformation'] = False
    
    # Analýza sezónnosti
    print("\nAnalyzujem sezónnosť...")
    # Testovanie rôznych potenciálnych sezónnych periód
    seasonal_periods = [7, 30, 90, 365]  # weekly, monthly, quarterly, yearly
    seasonal_results, strongest_period = analyze_seasonality(time_series, seasonal_periods)
    
    # If we have a strongest period, create seasonal decomposition plot
    if strongest_period and strongest_period in seasonal_results:
        decomposition = seasonal_results[strongest_period]['decomposition']
        
        plt.figure(figsize=(12, 10))
        plt.subplot(411)
        plt.plot(time_series, label='Original')
        plt.legend(loc='best')
        plt.title(f'Seasonal Decomposition (Period={strongest_period})')
        
        plt.subplot(412)
        plt.plot(decomposition.trend, label='Trend')
        plt.legend(loc='best')
        
        plt.subplot(413)
        plt.plot(decomposition.seasonal, label='Seasonality')
        plt.legend(loc='best')
        
        plt.subplot(414)
        plt.plot(decomposition.resid, label='Residuals')
        plt.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/seasonal_decomposition.png")
        plt.close()
    
    # Calculate seasonal strength for each period
    seasonal_strength = {period: result['strength'] for period, result in seasonal_results.items()}
    
    # Odporúčanie metód prognózovania
    recommendations = recommend_forecasting_methods(
        is_stationary or any(transformations_tried.values()),
        seasonal_results,
        seasonal_strength
    )
    
    # Generovanie správy
    print("\nGenerujem komplexnú správu...")
    report = generate_report(
        df, 
        is_stationary, 
        seasonal_results, 
        strongest_period, 
        recommendations,
        transformations_tried
    )
    
    # Uloženie správy do súboru
    with open('time_series_analysis_report.md', 'w') as f:
        f.write(report)
    
    print("\nAnalýza dokončená!")
    print(f"Vizualizácie uložené do adresára '{output_dir}'")
    print("Správa uložená ako 'time_series_analysis_report.md'")

if __name__ == "__main__":
    main()
