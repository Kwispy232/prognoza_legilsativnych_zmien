#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LSTM and GRU models for legislation change prediction

This script implements LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) 
neural network models for predicting legislation changes. The models are trained on 
historical data and evaluated using various metrics.

Author: Sebastian Mráz
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import arch.unitroot as unitroot
from scipy import stats
import warnings
import math
import os

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Suppress warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """
    Load and preprocess the legislation data from CSV.
    
    Args:
        file_path (str): Path to the CSV file with the data
        
    Returns:
        pd.DataFrame: Processed DataFrame with date index
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Sort index to ensure chronological order
    df = df.sort_index()
    
    # Check and handle missing values if needed
    if df.isna().sum().sum() > 0:
        print(f"Found {df.isna().sum().sum()} missing values")
        df = df.fillna(0)  # Fill missing values with 0
    
    print(f"\nData loaded successfully. Date range: {df.index.min()} to {df.index.max()}")
    print(f"Total observations: {len(df)}")
    
    return df

def ensure_output_dir(output_dir='legislation/output/lstm_gru'):
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir (str): Path to the output directory
        
    Returns:
        str: Path to the output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def plot_time_series(data, title='Legislation Changes Over Time', output_dir='legislation/output/lstm_gru'):
    """
    Plot the time series data and save to output directory.
    
    Args:
        data (pd.DataFrame): DataFrame with the time series data
        title (str): Title for the plot
        output_dir (str): Directory to save the plot
        
    Returns:
        str: Path to the saved figure
    """
    ensure_output_dir(output_dir)
    plt.figure(figsize=(15, 6))
    plt.plot(data.index, data['Count'], marker='o', linestyle='-', markersize=4)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Number of Changes')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"{output_dir}/time_series.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    return filename

def split_data(data, test_size=0.2):
    """
    Split data into training and testing sets.
    
    Args:
        data (pd.DataFrame): DataFrame with the time series data
        test_size (float): Proportion of data to use for testing (0-1)
        
    Returns:
        tuple: (train_data, test_data) - DataFrames for training and testing
    """
    # Calculate split point
    split_idx = int(len(data) * (1 - test_size))
    
    # Split data
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    print(f"\nData split:")
    print(f"Training set: {len(train_data)} observations ({train_data.index.min()} to {train_data.index.max()})")
    print(f"Testing set: {len(test_data)} observations ({test_data.index.min()} to {test_data.index.max()})")
    
    return train_data, test_data

def create_sequences(data, n_steps):
    """
    Create input sequences for the time series model.
    
    Args:
        data (np.array): Array of time series values
        n_steps (int): Number of time steps for each sequence
        
    Returns:
        tuple: (X, y) - Input sequences and target values
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps), 0])
        y.append(data[i + n_steps, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, units=50, dropout=0.2):
    """
    Build an LSTM model for time series prediction.
    
    Args:
        input_shape (tuple): Shape of input data (n_steps, n_features)
        units (int): Number of units in the LSTM layer
        dropout (float): Dropout rate for regularization
        
    Returns:
        tf.keras.Model: Compiled LSTM model
    """
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    return model

def build_gru_model(input_shape, units=50, dropout=0.2):
    """
    Build a GRU model for time series prediction.
    
    Args:
        input_shape (tuple): Shape of input data (n_steps, n_features)
        units (int): Number of units in the GRU layer
        dropout (float): Dropout rate for regularization
        
    Returns:
        tf.keras.Model: Compiled GRU model
    """
    model = Sequential()
    model.add(GRU(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(GRU(units=units, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    return model

def calculate_metrics(actual, predicted):
    """
    Calculate various evaluation metrics for regression models.
    
    Args:
        actual (np.array): Actual values
        predicted (np.array): Predicted values
        
    Returns:
        dict: Dictionary with calculated metrics
    """
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    # Calculate MAPE with handling for zero values
    actual_non_zero = actual.copy()
    predicted_non_zero = predicted.copy()
    mask = actual_non_zero != 0
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(actual_non_zero[mask], predicted_non_zero[mask]) * 100
    else:
        mape = np.nan
    
    # Calculate AIC
    n = len(actual)
    k = 2  # Number of parameters (simplified)
    aic = n * np.log(mse) + 2 * k
    
    # Calculate MASE (Mean Absolute Scaled Error)
    # Using one-step naïve forecast as benchmark (seasonal=True, m=7 for weekly data)
    seasonal_period = 7  # Weekly seasonality based on memory
    diff = np.abs(np.diff(actual, seasonal_period))
    d = np.mean(diff) if len(diff) > 0 else np.mean(np.abs(actual[1:] - actual[:-1]))
    
    if d != 0:
        mase = mae / d
    else:
        mase = np.nan
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape,
        'AIC': aic,
        'MASE': mase
    }

def test_residuals(actual, predicted):
    """
    Test residuals for normality and autocorrelation.
    
    Args:
        actual (np.array): Actual values
        predicted (np.array): Predicted values
        
    Returns:
        dict: Dictionary with test results
    """
    residuals = actual - predicted
    
    # Normality tests
    shapiro_test = stats.shapiro(residuals)
    ks_test = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
    
    # Autocorrelation test
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    
    # ARCH effect test
    try:
        arch_test = sm.stats.diagnostic.het_arch(residuals, nlags=7)
    except:
        arch_test = (np.nan, np.nan)  # In case of failure
    
    return {
        'Shapiro-Wilk Test': {'statistic': shapiro_test[0], 'p-value': shapiro_test[1]},
        'Kolmogorov-Smirnov Test': {'statistic': ks_test[0], 'p-value': ks_test[1]},
        'Ljung-Box Test (lag=10)': {'statistic': lb_test.iloc[0]['lb_stat'], 'p-value': lb_test.iloc[0]['lb_pvalue']},
        'ARCH Effect Test (lag=7)': {'statistic': arch_test[0], 'p-value': arch_test[1]}
    }

def plot_residual_diagnostics(actual, predicted, model_name, output_dir='legislation/output/lstm_gru'):
    """
    Generate diagnostic plots for residuals and save to output directory.
    
    Args:
        actual (np.array): Actual values
        predicted (np.array): Predicted values
        model_name (str): Name of the model for plot titles
        output_dir (str): Directory to save the plot
        
    Returns:
        str: Path to the saved figure
    """
    ensure_output_dir(output_dir)
    residuals = actual - predicted
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals plot
    axes[0, 0].plot(residuals)
    axes[0, 0].axhline(y=0, color='r', linestyle='-')
    axes[0, 0].set_title(f'Residuals over time - {model_name}')
    axes[0, 0].set_xlabel('Observation')
    axes[0, 0].set_ylabel('Residual')
    
    # Histogram with normal curve
    sns.histplot(residuals, kde=True, ax=axes[0, 1], bins=20)
    axes[0, 1].set_title(f'Residuals Distribution - {model_name}')
    
    # Q-Q plot
    sm.qqplot(residuals, line='45', ax=axes[1, 0])
    axes[1, 0].set_title(f'Q-Q Plot - {model_name}')
    
    # ACF plot
    plot_acf(residuals, lags=20, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title(f'Autocorrelation - {model_name}')
    
    plt.tight_layout()
    filename = f"{output_dir}/{model_name.lower()}_residual_diagnostics.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    return filename

def monte_carlo_prediction_intervals(model, X_last, scaler, n_steps, horizon=30, simulations=300, conf_int=0.95):
    """
    Generate prediction intervals using Monte Carlo simulation.
    
    Args:
        model (tf.keras.Model): Trained model
        X_last (np.array): Last observed sequence
        scaler (MinMaxScaler): Scaler used for normalization
        n_steps (int): Number of time steps in input sequence
        horizon (int): Number of steps to forecast
        simulations (int): Number of Monte Carlo simulations
        conf_int (float): Confidence interval (0-1)
        
    Returns:
        tuple: (mean_predictions, lower_bound, upper_bound) - Forecast with prediction intervals
    """
    # Initialize arrays to store simulations
    all_simulations = np.zeros((simulations, horizon))
    
    # Get residuals standard deviation (approximation of forecast error)
    train_predictions = model.predict(X_last[:1000] if len(X_last) > 1000 else X_last)
    residuals = X_last[:1000, -1, 0] - train_predictions.flatten() if len(X_last) > 1000 else X_last[:, -1, 0] - train_predictions.flatten()
    sigma = np.std(residuals)
    
    # For each simulation
    for sim in range(simulations):
        # Print progress every 10%
        if sim % (simulations // 10) == 0:
            print(f"\rMonte Carlo simulation: {sim}/{simulations}", end="", flush=True)
        # Start with the last known sequence
        current_sequence = X_last[-1:].copy()
        
        # Generate predictions for the horizon
        for step in range(horizon):
            # Predict next value
            next_value = model.predict(current_sequence)
            
            # Add noise based on residuals distribution
            next_value = next_value + np.random.normal(0, sigma)
            
            # Store prediction for this simulation
            all_simulations[sim, step] = next_value[0, 0]
            
            # Update sequence for next prediction (rolling window)
            new_sequence = np.append(current_sequence[0, 1:, :], [[next_value[0, 0]]], axis=0)
            current_sequence = np.array([new_sequence])
    
    # Calculate mean and confidence intervals
    mean_predictions = np.mean(all_simulations, axis=0)
    lower_bound = np.percentile(all_simulations, (1 - conf_int) / 2 * 100, axis=0)
    upper_bound = np.percentile(all_simulations, (1 + conf_int) / 2 * 100, axis=0)
    
    # Inverse transform to original scale
    mean_predictions_original = scaler.inverse_transform(mean_predictions.reshape(-1, 1)).flatten()
    lower_bound_original = scaler.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
    upper_bound_original = scaler.inverse_transform(upper_bound.reshape(-1, 1)).flatten()
    
    return mean_predictions_original, lower_bound_original, upper_bound_original

def bootstrap_parameter_significance(model, X, y, n_bootstraps=100, alpha=0.05):
    """
    Test parameter significance using bootstrap.
    
    Args:
        model (tf.keras.Model): Trained model
        X (np.array): Input data
        y (np.array): Target data
        n_bootstraps (int): Number of bootstrap samples
        alpha (float): Significance level
        
    Returns:
        dict: Dictionary with parameter significance test results
    """
    try:
        n_samples = X.shape[0]
        # Get original weights
        original_weights = model.get_weights()
        
        # Initialize arrays to store bootstrap results
        bootstrap_weights = []
        
        # Perform bootstrap
        for i in range(n_bootstraps):
            print(f"\rBootstrap iteration {i+1}/{n_bootstraps}", end="", flush=True)
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            # Create and train a new model with the same architecture
            temp_model = tf.keras.models.clone_model(model)
            temp_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            
            # Train with a small number of epochs to save time
            temp_model.fit(X_sample, y_sample, epochs=3, verbose=0)
            
            # Store weights
            bootstrap_weights.append(temp_model.get_weights())
        
        # Calculate confidence intervals and p-values for each parameter
        results = {}
        
        # Analyze each layer
        for layer_idx, layer in enumerate(model.layers):
            if len(layer.get_weights()) > 0:  # Skip layers without weights
                # Weights and biases
                for param_idx, param_name in enumerate(['weights', 'biases']):
                    if param_idx < len(layer.get_weights()):
                        try:
                            # Get the original parameter and bootstrap samples
                            original_param = original_weights[layer_idx * 2 + param_idx] 
                            param_samples = np.array([w[param_idx] for w in bootstrap_weights])
                            
                            # Calculate mean and standard error
                            param_mean = np.mean(param_samples, axis=0)
                            param_std_err = np.std(param_samples, axis=0, ddof=1)
                            
                            # Replace zeros in std_err to avoid division by zero
                            param_std_err = np.where(param_std_err == 0, 1e-10, param_std_err)
                            
                            # Ensure shapes match for broadcasting
                            if original_param.shape != param_std_err.shape:
                                print(f"\nShape mismatch for {param_name} in layer {layer_idx}: {original_param.shape} vs {param_std_err.shape}")
                                continue
                            
                            # Calculate z-values and p-values
                            z_values = original_param / param_std_err
                            p_values = 2 * (1 - stats.norm.cdf(np.abs(z_values)))
                            
                            # Calculate confidence intervals
                            lower_ci = np.percentile(param_samples, alpha/2 * 100, axis=0)
                            upper_ci = np.percentile(param_samples, (1-alpha/2) * 100, axis=0)
                            
                            # Store results
                            results[f'Layer {layer_idx} - {param_name}'] = {
                                'mean': param_mean,
                                'std_err': param_std_err,
                                'z_values': z_values,
                                'p_values': p_values,
                                'lower_ci': lower_ci,
                                'upper_ci': upper_ci,
                                'significant': p_values < alpha
                            }
                        except Exception as e:
                            print(f"\nError processing {param_name} in layer {layer_idx}: {str(e)}")
        return results
    except Exception as e:
        print(f"\nParameter significance testing failed: {str(e)}")
        return {}
    
    return results

def plot_forecast_with_history(history, test, forecast, intervals=None, zoom=False, output_dir='legislation/output/lstm_gru'):
    """
    Plot the historical values, test data, and forecast with optional prediction intervals.
    
    Args:
        history (pd.DataFrame): Historical data used for training
        test (pd.DataFrame): Test data
        forecast (np.array): Forecast values
        intervals (tuple): Tuple containing (lower_bound, upper_bound) for prediction intervals
        zoom (bool): Whether to create a zoomed-in view of the forecast
        output_dir (str): Directory to save the plot
        
    Returns:
        list: Paths to saved figures
    """
    ensure_output_dir(output_dir)
    saved_files = []
    
    # Combine all data for the full plot
    all_data = pd.concat([history, test])
    
    # Create date range for the forecast
    last_date = test.index[-1]
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(len(forecast))]
    
    # Create the plot
    plt.figure(figsize=(15, 6))
    
    # Plot training and test data
    plt.plot(history.index, history['Count'], label='Training Data', color='blue')
    plt.plot(test.index, test['Count'], label='Test Data', color='green')
    
    # Plot the forecast
    plt.plot(forecast_dates, forecast, label='Forecast', color='red', linestyle='--')
    
    # Add prediction intervals if provided
    if intervals:
        lower_bound, upper_bound = intervals
        plt.fill_between(forecast_dates, lower_bound, upper_bound, color='red', alpha=0.2, label='95% Prediction Interval')
    
    plt.title('Time Series Forecast with History')
    plt.xlabel('Date')
    plt.ylabel('Number of Changes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"{output_dir}/forecast_with_history.png"
    plt.savefig(filename, dpi=300)
    saved_files.append(filename)
    
    # If zoom is True, create a zoomed-in view
    if zoom:
        plt.figure(figsize=(15, 6))
        
        # Set the zoom period to the last 90 days of history plus the forecast period
        zoom_start = all_data.index[-90]
        
        # Filter the data for the zoom period
        zoom_history = history[history.index >= zoom_start]
        zoom_test = test.copy()
        
        # Plot the zoomed-in data
        plt.plot(zoom_history.index, zoom_history['Count'], label='Training Data', color='blue')
        plt.plot(zoom_test.index, zoom_test['Count'], label='Test Data', color='green')
        plt.plot(forecast_dates, forecast, label='Forecast', color='red', linestyle='--')
        
        # Add prediction intervals if provided
        if intervals:
            lower_bound, upper_bound = intervals
            plt.fill_between(forecast_dates, lower_bound, upper_bound, color='red', alpha=0.2, label='95% Prediction Interval')
        
        plt.title('Zoomed Time Series Forecast')
        plt.xlabel('Date')
        plt.ylabel('Number of Changes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = f"{output_dir}/forecast_zoomed.png"
        plt.savefig(filename, dpi=300)
        saved_files.append(filename)
    
    plt.show()
    return saved_files

def plot_model_fit(data, model_predictions, model_name, n_steps, output_dir='legislation/output/lstm_gru'):
    """
    Plot model fit on the entire dataset.
    
    Args:
        data (pd.DataFrame): DataFrame with the time series data
        model_predictions (np.array): Model predictions
        model_name (str): Name of the model (LSTM or GRU)
        n_steps (int): Number of time steps used in sequence creation
        output_dir (str): Directory to save the plot
        
    Returns:
        str: Path to the saved figure
    """
    ensure_output_dir(output_dir)
    plt.figure(figsize=(15, 6))
    
    # Plot actual data
    plt.plot(data.index, data['Count'], label='Actual', marker='o', markersize=4)
    
    # Adjust index for predictions (they start after n_steps)
    pred_index = data.index[n_steps:n_steps+len(model_predictions)]
    plt.plot(pred_index, model_predictions, label=f'{model_name} Predictions', color='red')
    
    plt.title(f'{model_name} Model Fit on Entire Dataset')
    plt.xlabel('Date')
    plt.ylabel('Number of Changes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"{output_dir}/{model_name.lower()}_model_fit.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    return filename

def generate_markdown_report(lstm_metrics, gru_metrics, n_steps, forecast_mean, 
                           forecast_lower, forecast_upper, last_date, 
                           output_dir='legislation/output/lstm_gru'):
    """
    Generate a markdown report with model results.
    
    Args:
        lstm_metrics (dict): Dictionary with LSTM metrics
        gru_metrics (dict): Dictionary with GRU metrics
        n_steps (int): Number of time steps used in sequence creation
        forecast_mean (np.array): Mean forecast values
        forecast_lower (np.array): Lower bound of forecast
        forecast_upper (np.array): Upper bound of forecast
        last_date (datetime): Last date in the test data
        output_dir (str): Directory to save the report
        
    Returns:
        str: Path to the saved report
    """
    ensure_output_dir(output_dir)
    
    # Create markdown content
    md_content = f"""# LSTM and GRU Model Analysis for Legislative Changes

## Model Overview
This report presents the results of time series forecasting for legislative changes using LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) neural network models.

### Model Parameters
- Sequence Length: {n_steps} days (weekly seasonality)
- LSTM Architecture: 2 LSTM layers with dropout
- GRU Architecture: 2 GRU layers with dropout
- Training: Early stopping with patience of 10 epochs
- Optimizer: Adam with learning rate 0.001

## Performance Metrics

### LSTM Model
| Metric | Value |
|--------|-------|
"""
    
    # Add LSTM metrics
    for metric, value in lstm_metrics.items():
        md_content += f"| {metric} | {value:.4f} |\n"
    
    md_content += f"""
### GRU Model
| Metric | Value |
|--------|-------|
"""
    
    # Add GRU metrics
    for metric, value in gru_metrics.items():
        md_content += f"| {metric} | {value:.4f} |\n"
    
    md_content += f"""
## Model Comparison
| Metric | LSTM | GRU |
|--------|------|-----|
"""
    
    # Add comparison
    for metric in lstm_metrics.keys():
        md_content += f"| {metric} | {lstm_metrics[metric]:.4f} | {gru_metrics[metric]:.4f} |\n"
    
    md_content += f"""
## 30-Day Forecast
The following table shows the forecasted values for the next 30 days with 95% prediction intervals:

| Date | Forecast | Lower Bound | Upper Bound |
|------|----------|------------|------------|
"""
    
    # Add forecast
    for i in range(30):
        forecast_date = last_date + timedelta(days=i+1)
        md_content += f"| {forecast_date.date()} | {forecast_mean[i]:.2f} | {forecast_lower[i]:.2f} | {forecast_upper[i]:.2f} |\n"
    
    md_content += f"""
## Visualizations
- [Time Series Plot](time_series.png)
- [LSTM Model Fit](lstm_model_fit.png)
- [GRU Model Fit](gru_model_fit.png)
- [LSTM Residual Analysis](lstm_residuals.png)
- [GRU Residual Analysis](gru_residuals.png)
- [Forecast with History](forecast_with_history.png)

## Libraries and References

### Libraries Used
- TensorFlow 2.x: Neural network implementation
- Keras: High-level neural networks API
- NumPy: Numerical computing
- Pandas: Data manipulation
- Matplotlib/Seaborn: Data visualization
- Scikit-learn: Metrics and preprocessing
- StatsModels: Statistical analysis
- Arch: Unit root testing

### References
1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
2. Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT Press.
4. Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice. OTexts.
5. Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: forecasting and control. John Wiley & Sons.
"""
    
    # Write markdown to file
    report_path = f"{output_dir}/lstm_gru_report.md"
    with open(report_path, 'w') as f:
        f.write(md_content)
    
    print(f"Markdown report saved to {report_path}")
    return report_path

def main():
    """
    Main function to run the LSTM and GRU analysis.
    """
    # Set error handling and visualization configuration
    plt.style.use('seaborn-v0_8')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Create output directory
    output_dir = ensure_output_dir()
    
    print("======== LSTM and GRU Analysis for Legislation Changes ========")
    
    # 1. Load data
    data = load_data('legislation/unique_dates_counts.csv')
    
    # 2. Visualize the time series
    plot_time_series(data, output_dir=output_dir)
    
    # 3. Split data into training and testing sets (80% training, 20% testing)
    train_data, test_data = split_data(data, test_size=0.2)
    
    # 4. Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data[['Count']])
    test_scaled = scaler.transform(test_data[['Count']])
    
    # 5. Create sequences
    n_steps = 7  # Weekly seasonality based on memory
    X_train, y_train = create_sequences(train_scaled, n_steps)
    X_test, y_test = create_sequences(test_scaled, n_steps)
    
    # Reshape for LSTM/GRU [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print(f"\nSequence shape: {X_train.shape}")
    
    # 6. Build and train LSTM model
    print("\n===== LSTM Model =====")
    lstm_model = build_lstm_model(input_shape=(n_steps, 1))
    lstm_model.summary()
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    lstm_history = lstm_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 7. Evaluate LSTM model
    lstm_train_pred = lstm_model.predict(X_train)
    lstm_test_pred = lstm_model.predict(X_test)
    
    # Inverse transform predictions
    lstm_train_pred = scaler.inverse_transform(lstm_train_pred)
    lstm_test_pred = scaler.inverse_transform(lstm_test_pred)
    
    # Inverse transform actual values
    train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    print("\nLSTM Model Metrics:")
    lstm_metrics = calculate_metrics(test_actual, lstm_test_pred)
    for metric, value in lstm_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Test residuals
    print("\nLSTM Residual Analysis:")
    lstm_residual_tests = test_residuals(test_actual.flatten(), lstm_test_pred.flatten())
    for test_name, result in lstm_residual_tests.items():
        print(f"{test_name}: Statistic = {result['statistic']:.4f}, p-value = {result['p-value']:.4f}")
    
    # Plot residual diagnostics
    plot_residual_diagnostics(test_actual.flatten(), lstm_test_pred.flatten(), "LSTM", output_dir=output_dir)
    
    # 8. Build and train GRU model
    print("\n===== GRU Model =====")
    gru_model = build_gru_model(input_shape=(n_steps, 1))
    gru_model.summary()
    
    # Train the model
    gru_history = gru_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 9. Evaluate GRU model
    gru_train_pred = gru_model.predict(X_train)
    gru_test_pred = gru_model.predict(X_test)
    
    # Inverse transform predictions
    gru_train_pred = scaler.inverse_transform(gru_train_pred)
    gru_test_pred = scaler.inverse_transform(gru_test_pred)
    
    # Calculate metrics
    print("\nGRU Model Metrics:")
    gru_metrics = calculate_metrics(test_actual, gru_test_pred)
    for metric, value in gru_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Test residuals
    print("\nGRU Residual Analysis:")
    gru_residual_tests = test_residuals(test_actual.flatten(), gru_test_pred.flatten())
    for test_name, result in gru_residual_tests.items():
        print(f"{test_name}: Statistic = {result['statistic']:.4f}, p-value = {result['p-value']:.4f}")
    
    # Plot residual diagnostics
    plot_residual_diagnostics(test_actual.flatten(), gru_test_pred.flatten(), "GRU", output_dir=output_dir)
    
    # 10. Compare LSTM and GRU models
    print("\n===== Model Comparison =====")
    print("Metric\t\tLSTM\t\tGRU")
    for metric in lstm_metrics.keys():
        print(f"{metric}\t\t{lstm_metrics[metric]:.4f}\t\t{gru_metrics[metric]:.4f}")
    
    # 11. Parameter significance testing
    print("\n===== Parameter Significance Testing =====")
    print("Performing bootstrap for parameter significance (this may take some time)...")
    
    # Use a smaller subset for bootstrap to save time
    bootstrap_size = min(300, len(X_train))
    X_bootstrap = X_train[:bootstrap_size]
    y_bootstrap = y_train[:bootstrap_size]
    
    # Test parameter significance for the better model
    if lstm_metrics['RMSE'] <= gru_metrics['RMSE']:
        better_model = lstm_model
        model_name = "LSTM"
    else:
        better_model = gru_model
        model_name = "GRU"
    
    try:
        param_significance = bootstrap_parameter_significance(
            better_model, X_bootstrap, y_bootstrap, n_bootstraps=100
        )
        print("\n") # Clear progress line
        
        if param_significance:
            print(f"Parameter significance results for {model_name} model:")
            for param, result in param_significance.items():
                try:
                    significant_count = np.sum(result['significant'])
                    total_count = result['significant'].size
                    print(f"{param}: {significant_count}/{total_count} parameters significant at α=0.05")
                except:
                    print(f"Could not analyze significance for {param}")
        else:
            print("Parameter significance testing was skipped due to errors.")
    except Exception as e:
        print(f"\nError in parameter significance testing: {str(e)}")
        print("Skipping parameter significance testing.")
    
    # 12. Generate forecasts with prediction intervals
    print("\n===== Generating 30-Day Forecast with Prediction Intervals =====")
    
    # Prepare the last sequence for forecasting
    last_sequence = test_scaled[-n_steps:].reshape(1, n_steps, 1)
    
    # Generate forecast using the better model
    if lstm_metrics['RMSE'] <= gru_metrics['RMSE']:
        forecast_mean, forecast_lower, forecast_upper = monte_carlo_prediction_intervals(
            lstm_model, X_test, scaler, n_steps, horizon=30, simulations=1000
        )
        model_name = "LSTM"
    else:
        forecast_mean, forecast_lower, forecast_upper = monte_carlo_prediction_intervals(
            gru_model, X_test, scaler, n_steps, horizon=30, simulations=1000
        )
        model_name = "GRU"
    
    # Ensure non-negative values in forecast
    forecast_mean = np.maximum(0, forecast_mean)
    forecast_lower = np.maximum(0, forecast_lower)
    forecast_upper = np.maximum(0, forecast_upper)
    
    # Print forecast
    print("\n30-Day Forecast:")
    last_date = test_data.index[-1]
    for i in range(30):
        forecast_date = last_date + timedelta(days=i+1)
        print(f"{forecast_date.date()}: {forecast_mean[i]:.2f} [{forecast_lower[i]:.2f}, {forecast_upper[i]:.2f}]")
    
    # 13. Plot the forecast with history
    plot_forecast_with_history(
        train_data, 
        test_data, 
        forecast_mean, 
        intervals=(forecast_lower, forecast_upper),
        zoom=False,
        output_dir=output_dir
    )
    
    # 14. Plot model fit on entire dataset
    # First combine train and test data
    all_data = pd.concat([train_data, test_data])
    
    # For LSTM: we need to combine train and test predictions
    lstm_all_pred = np.vstack((lstm_train_pred, lstm_test_pred)).flatten()
    plot_model_fit(all_data, lstm_all_pred, "LSTM", n_steps, output_dir)
    
    # For GRU: we need to combine train and test predictions
    gru_all_pred = np.vstack((gru_train_pred, gru_test_pred)).flatten()
    plot_model_fit(all_data, gru_all_pred, "GRU", n_steps, output_dir)
    
    # 15. Generate markdown report
    report_path = generate_markdown_report(
        lstm_metrics, 
        gru_metrics, 
        n_steps, 
        forecast_mean, 
        forecast_lower, 
        forecast_upper, 
        last_date,
        output_dir
    )
    
    print(f"\nAll outputs successfully saved to {output_dir}")
    
    # Plot zoomed forecast
    plot_forecast_with_history(
        train_data, 
        test_data, 
        forecast_mean, 
        intervals=(forecast_lower, forecast_upper),
        zoom=True
    )
    
    print("\n===== Analysis Complete =====")

if __name__ == "__main__":
    main()
