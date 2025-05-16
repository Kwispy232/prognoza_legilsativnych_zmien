# LSTM and GRU Model Analysis for Legislative Changes

## Model Overview
This report presents the results of time series forecasting for legislative changes using LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) neural network models.

### Model Parameters
- Sequence Length: 7 days (weekly seasonality)
- LSTM Architecture: 2 LSTM layers with dropout
- GRU Architecture: 2 GRU layers with dropout
- Training: Early stopping with patience of 10 epochs
- Optimizer: Adam with learning rate 0.001

## Performance Metrics

### LSTM Model
| Metric | Value |
|--------|-------|
| MSE | 28.5483 |
| RMSE | 5.3431 |
| MAE | 3.1263 |
| R² | 0.2904 |
| MAPE | 69.5382 |
| AIC | 218.5022 |
| MASE | nan |

### GRU Model
| Metric | Value |
|--------|-------|
| MSE | 28.1307 |
| RMSE | 5.3038 |
| MAE | 2.9731 |
| R² | 0.3008 |
| MAPE | 68.2147 |
| AIC | 217.5591 |
| MASE | nan |

## Model Comparison
| Metric | LSTM | GRU |
|--------|------|-----|
| MSE | 28.5483 | 28.1307 |
| RMSE | 5.3431 | 5.3038 |
| MAE | 3.1263 | 2.9731 |
| R² | 0.2904 | 0.3008 |
| MAPE | 69.5382 | 68.2147 |
| AIC | 218.5022 | 217.5591 |
| MASE | nan | nan |

## 30-Day Forecast
The following table shows the forecasted values for the next 30 days with 95% prediction intervals:

| Date | Forecast | Lower Bound | Upper Bound |
|------|----------|------------|------------|
| 2024-04-01 | 7.77 | 0.00 | 19.59 |
| 2024-04-02 | 2.69 | 0.00 | 15.14 |
| 2024-04-03 | 1.57 | 0.00 | 15.15 |
| 2024-04-04 | 0.70 | 0.00 | 13.57 |
| 2024-04-05 | 0.94 | 0.00 | 15.67 |
| 2024-04-06 | 1.52 | 0.00 | 15.30 |
| 2024-04-07 | 3.18 | 0.00 | 17.06 |
| 2024-04-08 | 3.86 | 0.00 | 17.64 |
| 2024-04-09 | 2.89 | 0.00 | 17.48 |
| 2024-04-10 | 2.58 | 0.00 | 16.66 |
| 2024-04-11 | 1.70 | 0.00 | 15.75 |
| 2024-04-12 | 1.92 | 0.00 | 15.69 |
| 2024-04-13 | 2.17 | 0.00 | 17.96 |
| 2024-04-14 | 3.06 | 0.00 | 18.20 |
| 2024-04-15 | 3.13 | 0.00 | 18.75 |
| 2024-04-16 | 2.10 | 0.00 | 17.44 |
| 2024-04-17 | 2.45 | 0.00 | 16.79 |
| 2024-04-18 | 2.22 | 0.00 | 17.21 |
| 2024-04-19 | 1.79 | 0.00 | 15.98 |
| 2024-04-20 | 2.82 | 0.00 | 17.35 |
| 2024-04-21 | 2.91 | 0.00 | 18.55 |
| 2024-04-22 | 2.69 | 0.00 | 17.76 |
| 2024-04-23 | 2.20 | 0.00 | 17.84 |
| 2024-04-24 | 2.44 | 0.00 | 16.91 |
| 2024-04-25 | 2.22 | 0.00 | 17.81 |
| 2024-04-26 | 2.70 | 0.00 | 18.41 |
| 2024-04-27 | 2.60 | 0.00 | 17.44 |
| 2024-04-28 | 2.28 | 0.00 | 17.57 |
| 2024-04-29 | 2.51 | 0.00 | 17.04 |
| 2024-04-30 | 2.30 | 0.00 | 17.32 |

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
