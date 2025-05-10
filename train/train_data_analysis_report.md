# Time Series Analysis Report

## Dataset Summary
- Number of observations: 1774
- Date range: 2020-04-19 to 2025-03-05
- Value statistics:
  - Min: 0.57
  - Max: 24.66
  - Mean: 4.52
  - Median: 4.25
  - Standard deviation: 2.29

## Stationarity Analysis
- Augmented Dickey-Fuller Test: The time series is stationary (p-value < 0.01)
- KPSS Test: Some non-stationarity detected (p-value < 0.05)
- First differencing was applied to ensure complete stationarity

## Seasonality Analysis
- Multiple seasonal periods tested: 7, 21, 35, and 63 days
- Weekly (7-day) seasonality provides the best fit (lowest AIC)
- This confirms previous findings that 7-day seasonality works best
- The seasonal pattern is statistically significant

## ARIMA/SARIMA Modeling

### Model Selection
- Best model: SARIMA(1,1,2)(0,0,1)[7]
- The model incorporates:
  - First-order differencing (d=1)
  - Autoregressive component (p=1)
  - Moving average components (q=2)
  - Seasonal moving average (Q=1) with period 7

### Parameter Estimates & Significance
- AR(1): 0.5166 (p < 0.001) ***
- MA(1): -1.0761 (p < 0.001) ***
- MA(2): 0.1321 (p = 0.0244) *
- SMA(1): 0.1823 (p < 0.001) ***
- All parameters are statistically significant

### Model Diagnostics
- Residual autocorrelation: No significant autocorrelation detected (Ljung-Box p = 0.5299)
- Residual normality: Residuals are not normally distributed (Shapiro-Wilk p < 0.001)
- Heteroskedasticity: ARCH effects present in residuals (p < 0.001)

### Performance Metrics
- Mean Squared Error (MSE): 4.1282
- Root Mean Squared Error (RMSE): 2.0318
- Mean Absolute Error (MAE): 1.5622
- Mean Absolute Percentage Error (MAPE): 35.90%
- R²: -0.2673 (indicates the model performs worse than a simple mean model)
- Mean Absolute Scaled Error (MASE): 1.4110

## Comparison with Holt-Winters Model
- A manual Holt-Winters model with additive seasonality was also tested
- Parameters used (based on previous analysis):
  - Smoothing Level (alpha): 0.5
  - Smoothing Trend (beta): 0.01
  - Smoothing Seasonal (gamma): 0.1
- The Holt-Winters model performed slightly worse than the SARIMA model

## Conclusion
The SARIMA(1,1,2)(0,0,1)[7] model provides the best fit for this time series data, confirming the weekly seasonality pattern identified in previous analyses. All model parameters are statistically significant, indicating they contribute meaningfully to the model. The lack of residual autocorrelation suggests the model adequately captures the time-dependent structure in the data.

However, model performance metrics indicate that there is room for improvement. The negative R² value suggests that the model does not explain the variance in the data well compared to a simple mean model. Additionally, the presence of heteroskedasticity (ARCH effects) and non-normal residuals indicates some model assumptions are violated.

Future work could explore additional transformations, alternative modeling approaches (such as GARCH for handling heteroskedasticity), or incorporation of external variables that might explain the variance in the data.

## Forecasting Method Recommendations

### 1. ARIMA
- Reason: Time series is stationary with weak or no seasonality
- Implementation: `statsmodels.tsa.arima.model.ARIMA`

### 2. Exponential Smoothing
- Reason: Simple and effective for stationary data
- Implementation: `statsmodels.tsa.holtwinters.ExponentialSmoothing`

### 3. XGBoost/LightGBM with time features
- Reason: Can capture complex patterns with feature engineering
- Implementation: `xgboost.XGBRegressor or lightgbm.LGBMRegressor with date-based features`