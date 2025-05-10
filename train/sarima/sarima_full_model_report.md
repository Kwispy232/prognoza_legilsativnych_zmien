# SARIMA Analysis Report - Full Dataset Model

## Data Overview

- Time series frequency: 7 observations per cycle
- Time series length: 1782 observations
- Date range: 2020-04-19 to 2025-03-05 

## Stationarity Analysis

- Augmented Dickey-Fuller Test p-value: 0.01 
- Interpretation: Series is stationary 

## Model Selection

- Best model type: SARIMA 
- Best seasonal period: 7 
- Model specification: ARIMA(1,1,2)(0,0,1)[7]  
- AIC: 6541.978 
- BIC: 6569.403 

## Parameter Estimates & Significance

- **ar1**: 0.5166 (p-value: <2e-16)
- **ma1**: -1.0761 (p-value: <2e-16)
- **ma2**: 0.1321 (p-value: 0.0244)
- **sma1**: 0.1823 (p-value: 7.52e-14)

## Model Diagnostics

- Ljung-Box Test for Autocorrelation: p-value = 0.5299 
  - Interpretation: No significant autocorrelation in residuals 

- Shapiro-Wilk Test for Normality: p-value = 0 
  - Interpretation: Residuals are not normally distributed 

- Box-Ljung Test for ARCH Effects: p-value = 0 
  - Interpretation: ARCH effects present in residuals 

## Performance Metrics

### Test Set Performance (Train-Test Split)

- **MSE**: 1.004
- **RMSE**: 1.002
- **MAE**: 0.7431
- **MAPE (%)**: 26.6109
- **R²**: -0.0392
- **MASE**: 0.6619
- **Theil's U**: 0.1827

### Full Model Performance (In-Sample)

- **MSE**: 2.2898
- **RMSE**: 1.5132
- **MAE**: 0.9718
- **MAPE (%)**: 23.414
- **R²**: 0.5633
- **MASE**: 0.8656
- **Theil's U**: 0.1521

## Forecast Approaches

This analysis implements two complementary forecasting approaches:

1. **Train-Test Split Evaluation**:
   - Uses the last 30 days of data as a test set
   - Allows validation of model performance on known historical data
   - Provides confidence metrics for forecast accuracy

2. **Full Dataset Forecasting**:
   - Uses the complete historical dataset to build the model
   - Produces a true 30-day future forecast (from 2025-03-06 to 2025-04-04 )
   - Maximizes information utilization for optimal predictions

The forecast for the next 30 days (from 2025-03-06 to 2025-04-04) is visualized in the accompanying charts.

## Conclusion

The SARIMA model with period 7 was identified as the best fit for the legislative changes data. The presence of weekly seasonality (period=7) confirms previous findings about cyclical patterns in legislative activities.

With an R² of 0.5633, the model explains a significant portion of the variance in the legislative data. The lack of autocorrelation in the residuals indicates that the model has successfully captured the time-dependent structure in the data.

The presence of ARCH effects in the residuals suggests that a GARCH-type model might be beneficial for capturing the volatility dynamics in legislative activities.

The 30-day forecast provides valuable insights for policy planning and resource allocation, leveraging the full historical dataset to make the most accurate possible predictions about future legislative activity levels.
