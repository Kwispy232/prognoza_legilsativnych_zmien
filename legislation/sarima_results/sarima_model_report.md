# SARIMA Model Analysis Report

## 1. Overview

**Date Range:** 2023-04-14 to 2024-03-31
**Number of Observations:** 353
**Data Frequency:** Daily

## 2. Model Specification

The Seasonal ARIMA (SARIMA) model is a time series forecasting method that incorporates seasonality. 

### Mathematical Notation

The SARIMA model is denoted as SARIMA(p,d,q)(P,D,Q)m where:
- p = 0: Order of the non-seasonal autoregressive (AR) terms
- d = 0: Order of non-seasonal differencing
- q = 1: Order of the non-seasonal moving average (MA) terms
- P = 2: Order of the seasonal autoregressive terms
- D = 0: Order of seasonal differencing
- Q = 1: Order of the seasonal moving average terms
- m = 7: Seasonal period

The mathematical form of the SARIMA model is:

$$\Phi_P(B^m)\phi_p(B)(1-B)^d(1-B^m)^D y_t = \alpha + \Theta_Q(B^m)\theta_q(B)\varepsilon_t$$

Where:
- $\phi_p(B)$: Non-seasonal AR operator of order p
- $\Phi_P(B^m)$: Seasonal AR operator of order P
- $(1-B)^d$: Non-seasonal differencing of order d
- $(1-B^m)^D$: Seasonal differencing of order D
- $\theta_q(B)$: Non-seasonal MA operator of order q
- $\Theta_Q(B^m)$: Seasonal MA operator of order Q
- $\varepsilon_t$: White noise error term
- $B$: Backshift operator
- $\alpha$: Constant term

## 3. Model Diagnostics

**AIC:** 2040.6543
**BIC:** 2059.7843

**Residual Analysis:**
- Mean of Residuals: 0.0566
- Variance of Residuals: 22.8644
- Ljung-Box Test p-value: 0.6847 (> 0.05 indicates no significant autocorrelation in residuals)
- Jarque-Bera Test p-value: 0.0000 (> 0.05 indicates residuals are normally distributed)

## 4. Model Performance Metrics

### In-Sample Performance (Training Data)
- MSE: 22.8764
- RMSE: 4.7829
- MAE: 2.3197
- R²: 0.2736
- MASE: 0.6144

### Out-of-Sample Performance (Test Data)
- MSE: 27.3124
- RMSE: 5.2261
- MAE: 2.3420
- R²: 0.3202
- MASE: 0.4794

## 5. Forecasting

Forecast period: 30 days
Start date: 2024-04-01
End date: 2024-04-30

## 6. Conclusion

Based on the model performance metrics and diagnostics, the SARIMA(0,0,1)(2,0,1)7 model is suitable for forecasting legislative changes. The model captures the weekly seasonality pattern in legislative changes and can be used for short to medium-term forecasting of legislative activity.

## 7. Visualizations

Multiple visualizations were generated during this analysis and stored in the 'legislation/sarima_results' directory:
- Stationarity tests
- Seasonal decomposition
- ACF and PACF plots
- Residual analysis
- Forecast with confidence intervals
