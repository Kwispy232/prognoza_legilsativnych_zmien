# Knižnice a nástroje použité v analýzach legislatívnych zmien

*Kompletný zoznam pre analýzy Holt-Winters a SARIMA modelov*

## Python a základné knižnice
- **Python**: Programovací jazyk použitý na implementáciu analýz
  - Dokumentácia: https://docs.python.org/3/
  - Oficiálna stránka: https://www.python.org/

- **NumPy**: Fundamentálna knižnica pre vedecké výpočty v Pythone
  - Dokumentácia: https://numpy.org/doc/
  - Oficiálna stránka: https://numpy.org/
  - Vedecká citácia: Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2

- **pandas**: Knižnica pre manipuláciu s dátami a analýzu
  - Dokumentácia: https://pandas.pydata.org/docs/
  - Oficiálna stránka: https://pandas.pydata.org/
  - Vedecká citácia: McKinney, W. Data Structures for Statistical Computing in Python. In Proceedings of the 9th Python in Science Conference, 51-56 (2010).

## Vizualizácia
- **Matplotlib**: Základná knižnica pre tvorbu grafov a vizualizácií
  - Dokumentácia: https://matplotlib.org/stable/contents.html
  - Oficiálna stránka: https://matplotlib.org/
  - Vedecká citácia: Hunter, J. D. Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3), 90-95 (2007). DOI: 10.1109/MCSE.2007.55

- **seaborn**: Knižnica pre štatistickú vizualizáciu založená na Matplotlib
  - Dokumentácia: https://seaborn.pydata.org/
  - Oficiálna stránka: https://seaborn.pydata.org/
  - GitHub: https://github.com/mwaskom/seaborn

## Štatistická analýza a modelovanie časových radov
- **statsmodels**: Knižnica pre odhad a testovanie štatistických modelov
  - Dokumentácia: https://www.statsmodels.org/stable/index.html
  - Oficiálna stránka: https://www.statsmodels.org/
  - Vedecká citácia: Seabold, S., & Perktold, J. Statsmodels: Econometric and Statistical Modeling with Python. In Proceedings of the 9th Python in Science Conference (2010).

### Modely Holt-Winters
- **statsmodels.tsa.holtwinters**: Implementácia Holt-Winters exponenciálneho vyhladzovacieho modelu
  - Dokumentácia: https://www.statsmodels.org/stable/tsa.html#exponential-smoothing
  - Špecifická dokumentácia: https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html

### Modely SARIMA
- **statsmodels.tsa.statespace.sarimax**: Implementácia SARIMA a SARIMAX modelov
  - Dokumentácia: https://www.statsmodels.org/stable/statespace.html
  - Špecifická dokumentácia: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

- **pmdarima**: Knižnica pre automatický výber a optimalizáciu ARIMA modelov (ekvivalent funkcie auto.arima z R)
  - Dokumentácia: https://alkaline-ml.com/pmdarima/
  - GitHub: https://github.com/alkaline-ml/pmdarima
  - PyPI: https://pypi.org/project/pmdarima/

### Dekompozícia časových radov
- **statsmodels.tsa.seasonal**: Implementácia sezónnej dekompozície časových radov
  - Dokumentácia: https://www.statsmodels.org/stable/tsa.html#seasonal-decomposition
  - Špecifická dokumentácia: https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html

### Nástroje pre analýzu časových radov
- **statsmodels.tsa.stattools**: Nástroje pre testovanie stacionarity (ADF test) a autokorelácie
  - Dokumentácia: https://www.statsmodels.org/stable/tsa.html#stationarity-tests
  - Špecifická dokumentácia (ADF test): https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
  - Špecifická dokumentácia (ACF): https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html
  - Špecifická dokumentácia (PACF): https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html

- **statsmodels.graphics.tsaplots**: Funkcie na vizualizáciu ACF a PACF
  - Dokumentácia: https://www.statsmodels.org/stable/graphics.html
  - Špecifická dokumentácia (ACF plot): https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_acf.html
  - Špecifická dokumentácia (PACF plot): https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_pacf.html

- **scipy**: Vedecké výpočty v Pythone (používané pre štatistické testy a bootstraping)
  - Dokumentácia: https://docs.scipy.org/doc/scipy/
  - Oficiálna stránka: https://scipy.org/
  - Vedecká citácia: Virtanen, P., Gommers, R., Oliphant, T. E., et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods, 17(3), 261-272 (2020). DOI: 10.1038/s41592-019-0686-2

- **scipy.stats**: Modul pre štatistické funkcie a testy
  - Dokumentácia: https://docs.scipy.org/doc/scipy/reference/stats.html

- **statsmodels.stats.diagnostic**: Nástroje pre diagnostiku modelu (Ljung-Box test)
  - Dokumentácia: https://www.statsmodels.org/stable/stats.html
  - Špecifická dokumentácia: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html

## Metriky pre vyhodnocovanie modelov
- **sklearn.metrics**: Funkcie pre výpočet rôznych štatistických metrík
  - Dokumentácia: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
  - Oficiálna stránka: https://scikit-learn.org/
  - Vedecká citácia: Pedregosa, F., Varoquaux, G., Gramfort, A., et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830 (2011).

## Teoretické zdroje

### Teoretické zdroje pre Holt-Winters model
- **Holt-Winters pôvodné publikácie**:
  - Holt, C. E. (1957). Forecasting seasonals and trends by exponentially weighted averages. O.N.R. Memorandum 52/1957, Carnegie Institute of Technology.
  - Winters, P. R. (1960). Forecasting sales by exponentially weighted moving averages. Management Science, 6(3), 324–342. DOI: 10.1287/mnsc.6.3.324

- **Prehľadové publikácie o exponenciálnom vyhladzovaní**:
  - Gardner, E. S. (1985). Exponential smoothing: The state of the art. Journal of Forecasting, 4(1), 1-28. DOI: 10.1002/for.3980040103
  - Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice (2nd ed.). OTexts. Online: https://otexts.com/fpp2/

### Teoretické zdroje pre SARIMA model
- **Box-Jenkins pôvodné publikácie**:
  - Box, G. E. P., & Jenkins, G. M. (1970). Time Series Analysis: Forecasting and Control. San Francisco: Holden-Day.
  - Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2008). Time Series Analysis: Forecasting and Control (4th ed.). Wiley.

- **Prehľadové publikácie o ARIMA a SARIMA modeloch**:
  - Brockwell, P. J., & Davis, R. A. (2016). Introduction to Time Series and Forecasting (3rd ed.). Springer.
  - Shumway, R. H., & Stoffer, D. S. (2017). Time Series Analysis and Its Applications: With R Examples (4th ed.). Springer.
  - Wei, W. W. S. (2006). Time Series Analysis: Univariate and Multivariate Methods (2nd ed.). Pearson Addison Wesley.

- **Auto ARIMA metodológia**:
  - Hyndman, R. J., & Khandakar, Y. (2008). Automatic time series forecasting: the forecast package for R. Journal of Statistical Software, 26(3), 1-22. DOI: 10.18637/jss.v027.i03

## Nástroje pre bootstraping a význam parametrov
- **Bootstraping metodológia**:
  - Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall/CRC.
  - Davison, A. C., & Hinkley, D. V. (1997). Bootstrap Methods and Their Application. Cambridge University Press.

## Ďalšie nástroje
- **os**: Modul pre interakciu s operačným systémom (práca so súbormi a adresármi)
  - Dokumentácia: https://docs.python.org/3/library/os.html

- **datetime**: Modul pre manipuláciu s dátumami a časmi
  - Dokumentácia: https://docs.python.org/3/library/datetime.html

- **itertools**: Modul pre efektívnu iteráciu (použité v grid search)
  - Dokumentácia: https://docs.python.org/3/library/itertools.html

- **warnings**: Modul pre správu varovaní
  - Dokumentácia: https://docs.python.org/3/library/warnings.html
