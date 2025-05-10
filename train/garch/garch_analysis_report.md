# GARCH Analysis Report

## Data Overview

- Time series frequency: 7 observations per cycle
- Time series length: 1782 observations
- Date range: 2020-04-19 to 2025-03-05 

## ARCH Effects Detection

- ARCH LM Test p-value: 0 
- Interpretation: Significant ARCH effects detected, warranting GARCH modeling 

## Model Selection

Models compared by AIC and BIC:

- GARCH(1,1) t-dist: AIC = 3.22, BIC = 3.24
- EGARCH(1,1) Normal: AIC = 3.5, BIC = 3.52
- GARCH(1,1) Normal: AIC = 3.5, BIC = 3.52
- GARCH(1,2) Normal: AIC = 3.5, BIC = 3.52
- GARCH(2,1) Normal: AIC = 3.5, BIC = 3.52
- APARCH(1,1) Normal: AIC = 3.5, BIC = 3.53

Best model: **GARCH(1,1) t-dist**

## Parameter Estimates

Mean model parameters:
- mu: 2.6921
- ar1: 0.9763
- ma1: -0.6808

Volatility model parameters:
- omega: 0.2363
- alpha1: 0.2491
- beta1: 0.7088

## Model Diagnostics

- ARCH test on standardized residuals p-value: 0.9970675 
- Interpretation: No significant ARCH effects remain in the residuals. The GARCH model successfully captured the volatility dynamics. 

## Forecast Performance Metrics

- **MSE**: 2.505
- **RMSE**: 1.5827
- **MAE**: 0.9881
- **MAPE (%)**: 23.4259
- **R²**: 0.5222
- **MASE**: 0.8797
- **Theil's U**: 0.8882
## Comparison with SARIMA Model

The GARCH model offers several advantages over the SARIMA model for this legislative data:

1. **Volatility Modeling**: GARCH explicitly models the changing variance, capturing periods of higher uncertainty in legislative activities
2. **Risk Assessment**: Provides more reliable prediction intervals during volatile periods
3. **Structural Insights**: Helps identify patterns in legislative volatility, potentially tied to political cycles or events

Unlike the previous approach that split the data into training and test sets, this analysis uses the full dataset to build the model and then generates true future forecasts. This provides:

1. **Maximum Information Utilization**: Uses all available historical data for model building
2. **True Future Forecasting**: Predictions represent actual future values rather than held-out historical data
3. **Complete Volatility Patterns**: Captures all historical volatility patterns for better future uncertainty estimates

The GARCH model shows good in-sample fit, with a positive R² value indicating it explains variation in the data better than a simple mean model.

## Conclusion

The GARCH(1,1) t-dist model was identified as the best fit for capturing both the mean and volatility dynamics in the legislative data. The model successfully addressed the ARCH effects detected in the original time series, providing a more accurate representation of the uncertainty in future predictions.

The 30-day future forecast provides policymakers with valuable insights not just into expected legislative activity levels, but also into the expected volatility, which can be crucial for resource planning and risk management. By using the full historical dataset, the model leverages all available information to make the most accurate possible predictions about truly unknown future values.

Future work might explore incorporating external variables like political events, economic indicators, or seasonal dummy variables to further improve forecast accuracy and volatility prediction.
