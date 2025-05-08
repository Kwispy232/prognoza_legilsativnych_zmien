# Knižnice a nástroje použité v analýzach legislatívnych zmien

## Základné knižnice
- **Python** - https://www.python.org/
- **NumPy** - https://numpy.org/
- **pandas** - https://pandas.pydata.org/
- **Matplotlib** - https://matplotlib.org/
- **seaborn** - https://seaborn.pydata.org/

## Časové rady a štatistická analýza
- **statsmodels** - https://www.statsmodels.org/
  - **ExponentialSmoothing** (Holt-Winters) - https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
  - **SARIMAX** - https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
  - **seasonal_decompose** - https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html
  - **adfuller, acf, pacf** - https://www.statsmodels.org/stable/tsa.html
  - **acorr_ljungbox** - https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html

- **pmdarima** (Auto ARIMA) - https://alkaline-ml.com/pmdarima/

- **scipy** - https://scipy.org/
  - **stats** - https://docs.scipy.org/doc/scipy/reference/stats.html

- **sklearn.metrics** - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

- **PyCaret** - https://pycaret.org/
  - **time_series** - https://pycaret.gitbook.io/docs/get-started/time-series

## Teoretické zdroje

### Holt-Winters
- Holt, C. E. (1957). Forecasting seasonals and trends by exponentially weighted averages. O.N.R. Memorandum 52/1957.
- Winters, P. R. (1960). Forecasting sales by exponentially weighted moving averages. Management Science, 6(3), 324–342.
- Gardner, E. S. (1985). Exponential smoothing: The state of the art. Journal of Forecasting, 4(1), 1-28.

### SARIMA
- Box, G. E. P., & Jenkins, G. M. (1970). Time Series Analysis: Forecasting and Control. San Francisco: Holden-Day.
- Hyndman, R. J., & Khandakar, Y. (2008). Automatic time series forecasting: the forecast package for R. Journal of Statistical Software, 26(3), 1-22.

### Bootstraping
- Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall/CRC.

### Všeobecné zdroje o časových radách
- Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice (2nd ed.). OTexts. Online: https://otexts.com/fpp2/
