#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of Train Scraped Entries Dataset
- Sorts the data by date
- Performs time series analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import boxcox
import os
from datetime import datetime

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create output directory for visualizations
output_dir = 'output/visualizations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_and_prepare_data(filename='train_scraped_entries.csv'):
    """
    Load the train_scraped_entries.csv file, convert dates to datetime format,
    and sort by date.
    """
    print(f"Loading dataset: {filename}")
    df = pd.read_csv(filename)
    
    # Display the first few rows of the original data
    print("\nOriginal data (first 5 rows):")
    print(df.head())
    
    # Convert date from DD.MM.YYYY format to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    
    # Sort by date
    df_sorted = df.sort_values('date')
    
    # Display the first few rows of the sorted data
    print("\nSorted data (first 5 rows):")
    print(df_sorted.head())
    
    # Save the sorted dataset
    sorted_filename = os.path.join('output', 'sorted_train_entries.csv')
    df_sorted.to_csv(sorted_filename, index=False)
    print(f"\nSorted dataset saved as {sorted_filename}")
    
    return df_sorted

def adf_test(timeseries):
    """
    Perform Augmented Dickey-Fuller test for stationarity
    """
    print("\nResults of Augmented Dickey-Fuller test:")
    result = adfuller(timeseries.dropna())
    
    # Format results
    output = pd.Series(
        [result[0], result[1], result[4]['1%'], result[4]['5%'], result[4]['10%']],
        index=['Test statistic', 'p-value', '1% critical value', '5% critical value', '10% critical value']
    )
    
    # Determine if the time series is stationary
    is_stationary = result[1] < 0.05
    
    print(output)
    print(f"Conclusion: The time series is {'stationary' if is_stationary else 'non-stationary'}")
    return is_stationary

def calculate_seasonal_strength(decomposition):
    """Calculate seasonal strength based on decomposition results"""
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    # Calculate variance
    var_seasonal = np.var(seasonal.dropna())
    var_residual = np.var(residual.dropna())
    
    # Calculate seasonal strength
    if var_seasonal + var_residual > 0:
        seasonal_strength = var_seasonal / (var_seasonal + var_residual)
    else:
        seasonal_strength = 0
    
    return seasonal_strength

def analyze_seasonality(timeseries, periods):
    """
    Analyze seasonality for different periods
    """
    results = {}
    
    for period in periods:
        if len(timeseries) > period * 2:  # Need at least 2 complete periods for decomposition
            try:
                decomposition = seasonal_decompose(timeseries, model='additive', period=period)
                strength = calculate_seasonal_strength(decomposition)
                results[period] = {
                    'decomposition': decomposition,
                    'strength': strength
                }
                print(f"Period {period}: Seasonal strength = {strength:.4f}")
            except Exception as e:
                print(f"Failed to decompose for period {period}: {e}")
    
    # Find period with strongest seasonality
    if results:
        strongest_period = max(results.keys(), key=lambda k: results[k]['strength'])
        print(f"Detected strongest seasonal period: {strongest_period} days")
        return results, strongest_period
    else:
        print("No significant seasonality detected")
        return results, None

def create_visualizations(df, output_dir):
    """
    Create and save visualizations
    """
    time_series = df.set_index('date')['time']
    
    # 1. Original time series
    plt.figure(figsize=(12, 6))
    plt.plot(time_series, color='blue')
    plt.title('Original Time Series')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/original_time_series.png")
    plt.close()
    
    # 2. Rolling mean and standard deviation
    rolling_mean = time_series.rolling(window=30).mean()
    rolling_std = time_series.rolling(window=30).std()
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_series, color='blue', label='Original')
    plt.plot(rolling_mean, color='red', label='Rolling Mean (30 days)')
    plt.plot(rolling_std, color='green', label='Rolling Std (30 days)')
    plt.title('Rolling Statistics')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rolling_statistics.png")
    plt.close()
    
    # 3. Time series after first differencing
    diff_series = time_series.diff().dropna()
    
    plt.figure(figsize=(12, 6))
    plt.plot(diff_series, color='purple')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Time Series After First Differencing')
    plt.xlabel('Date')
    plt.ylabel('Difference')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/differenced_series.png")
    plt.close()
    
    # 4. Autocorrelation function
    plt.figure(figsize=(12, 6))
    plot_acf(time_series.dropna(), lags=40, alpha=0.05)
    plt.title('Autocorrelation Function')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/acf.png")
    plt.close()
    
    # 5. Partial autocorrelation function
    plt.figure(figsize=(12, 6))
    plot_pacf(time_series.dropna(), lags=40, alpha=0.05)
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pacf.png")
    plt.close()
    
    # 6. Monthly aggregation
    monthly_data = time_series.resample('M').mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_data, marker='o')
    plt.title('Monthly Average Values')
    plt.xlabel('Date')
    plt.ylabel('Average Value')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/monthly_aggregation.png")
    plt.close()
    
    return time_series

def transform_series(time_series, transformation='difference'):
    """
    Transform time series if it's non-stationary
    """
    if transformation == 'difference':
        transformed = time_series.diff().dropna()
        print("Applied first differencing")
    elif transformation == 'log':
        # Ensure all values are positive for log transform
        if time_series.min() <= 0:
            offset = abs(time_series.min()) + 1
            transformed = np.log(time_series + offset)
            print(f"Applied log transformation with offset {offset}")
        else:
            transformed = np.log(time_series)
            print("Applied log transformation")
    elif transformation == 'boxcox':
        # BoxCox requires positive values
        if time_series.min() <= 0:
            offset = abs(time_series.min()) + 1
            transformed, lambda_param = boxcox(time_series + offset)
            transformed = pd.Series(transformed, index=time_series.index)
            print(f"Applied Box-Cox transformation with lambda={lambda_param:.4f} and offset {offset}")
        else:
            transformed, lambda_param = boxcox(time_series)
            transformed = pd.Series(transformed, index=time_series.index)
            print(f"Applied Box-Cox transformation with lambda={lambda_param:.4f}")
    else:
        transformed = time_series
        print("No transformation applied")
    
    return transformed

def recommend_forecasting_methods(is_stationary, seasonal_periods, seasonal_strength):
    """
    Recommend forecasting methods based on analysis results
    """
    recommendations = []
    
    if is_stationary and (not seasonal_periods or seasonal_strength < 0.3):
        recommendations.append({
            'method': 'ARIMA',
            'reason': 'Time series is stationary with weak or no seasonality',
            'implementation': 'statsmodels.tsa.arima.model.ARIMA'
        })
        recommendations.append({
            'method': 'Exponential Smoothing',
            'reason': 'Simple and effective for stationary data',
            'implementation': 'statsmodels.tsa.holtwinters.ExponentialSmoothing'
        })
    
    elif not is_stationary and (not seasonal_periods or seasonal_strength < 0.3):
        recommendations.append({
            'method': 'ARIMA with differencing',
            'reason': 'Time series is non-stationary but can be made stationary through differencing',
            'implementation': 'statsmodels.tsa.arima.model.ARIMA with d=1 or d=2'
        })
        recommendations.append({
            'method': 'Prophet',
            'reason': 'Handles non-stationary data well with automatic trend detection',
            'implementation': 'fbprophet.Prophet'
        })
    
    elif is_stationary and seasonal_periods and seasonal_strength >= 0.3:
        recommendations.append({
            'method': 'SARIMA',
            'reason': 'Time series is stationary with significant seasonality',
            'implementation': f'statsmodels.tsa.statespace.sarimax.SARIMAX with seasonal_order including period {seasonal_periods}'
        })
        recommendations.append({
            'method': 'Exponential Smoothing with Seasonality',
            'reason': 'Handles seasonality well for stationary data',
            'implementation': f'statsmodels.tsa.holtwinters.ExponentialSmoothing with seasonal_periods={seasonal_periods}'
        })
    
    else:  # not is_stationary and seasonal_periods and seasonal_strength >= 0.3
        recommendations.append({
            'method': 'SARIMA with differencing',
            'reason': 'Time series is non-stationary with significant seasonality',
            'implementation': f'statsmodels.tsa.statespace.sarimax.SARIMAX with d=1 and seasonal_order including period {seasonal_periods}'
        })
        recommendations.append({
            'method': 'Prophet',
            'reason': 'Handles both non-stationarity and seasonality well',
            'implementation': 'fbprophet.Prophet'
        })
        recommendations.append({
            'method': 'TBATS',
            'reason': 'Specifically designed for complex seasonality patterns',
            'implementation': 'tbats.TBATS'
        })
    
    # Always recommend machine learning methods
    recommendations.append({
        'method': 'XGBoost/LightGBM with time features',
        'reason': 'Can capture complex patterns with feature engineering',
        'implementation': 'xgboost.XGBRegressor or lightgbm.LGBMRegressor with date-based features'
    })
    
    return recommendations

def generate_report(df, is_stationary, seasonal_results, strongest_period, recommendations):
    """
    Generate a detailed report of the analysis
    """
    report = []
    
    # Dataset summary
    report.append("# Time Series Analysis Report")
    report.append("\n## Dataset Summary")
    report.append(f"- Number of observations: {len(df)}")
    report.append(f"- Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    report.append(f"- Value statistics:")
    report.append(f"  - Min: {df['time'].min():.2f}")
    report.append(f"  - Max: {df['time'].max():.2f}")
    report.append(f"  - Mean: {df['time'].mean():.2f}")
    report.append(f"  - Median: {df['time'].median():.2f}")
    report.append(f"  - Standard deviation: {df['time'].std():.2f}")
    
    # Stationarity results
    report.append("\n## Stationarity Analysis")
    report.append(f"- The time series is {'stationary' if is_stationary else 'non-stationary'}")
    report.append(f"- {'No transformation needed' if is_stationary else 'Transformation may be required (differencing, log transform, etc.)'}")
    
    # Seasonality results
    report.append("\n## Seasonality Analysis")
    if strongest_period:
        report.append(f"- Strongest seasonal period detected: {strongest_period} days")
        report.append(f"- Seasonal strength: {seasonal_results[strongest_period]['strength']:.4f}")
        if seasonal_results[strongest_period]['strength'] < 0.3:
            report.append("- Seasonality is weak and may not significantly impact forecasting")
        else:
            report.append("- Seasonality is significant and should be incorporated into forecasting models")
    else:
        report.append("- No significant seasonality detected")
    
    # Forecasting recommendations
    report.append("\n## Forecasting Method Recommendations")
    for i, rec in enumerate(recommendations, 1):
        report.append(f"\n### {i}. {rec['method']}")
        report.append(f"- Reason: {rec['reason']}")
        report.append(f"- Implementation: `{rec['implementation']}`")
    
    # Save report
    report_text = '\n'.join(report)
    report_file = os.path.join('output', 'train_data_analysis_report.md')
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(f"\nAnalysis report saved to {report_file}")
    return report_text

def main():
    """
    Main function to run the entire analysis
    """
    print("Starting analysis of train_scraped_entries.csv")
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Create visualizations
    time_series = create_visualizations(df, output_dir)
    
    # Test for stationarity
    is_stationary = adf_test(time_series)
    
    # If not stationary, try transformations
    if not is_stationary:
        print("\nTrying transformations to achieve stationarity:")
        transformations = ['difference', 'log', 'boxcox']
        for transform in transformations:
            transformed_series = transform_series(time_series, transformation=transform)
            print(f"\nTesting stationarity after {transform} transformation:")
            is_transformed_stationary = adf_test(transformed_series)
            if is_transformed_stationary:
                print(f"Success! The series is stationary after {transform} transformation.")
                break
    
    # Analyze seasonality
    print("\nAnalyzing seasonality:")
    # Try different periods: weekly, biweekly, monthly, quarterly
    seasonal_periods = [7, 14, 30, 90]
    seasonal_results, strongest_period = analyze_seasonality(time_series, seasonal_periods)
    
    # Recommend forecasting methods
    seasonal_strength = seasonal_results.get(strongest_period, {}).get('strength', 0) if strongest_period else 0
    recommendations = recommend_forecasting_methods(is_stationary, strongest_period, seasonal_strength)
    
    # Generate report
    report = generate_report(df, is_stationary, seasonal_results, strongest_period, recommendations)
    
    print("\nAnalysis complete!")
    print(f"Visualizations saved to {output_dir}")
    print("To view the sorted data and analysis results, check the output directory.")

if __name__ == "__main__":
    main()
