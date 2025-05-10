# Knižnice a nástroje použité v train analýzach

*Kompletný zoznam pre analýzy SARIMA a GARCH modelov*

## R a základné knižnice
- **R**: Štatistický programovací jazyk použitý na implementáciu analýz
  - Dokumentácia: https://www.r-project.org/other-docs.html
  - Oficiálna stránka: https://www.r-project.org/
  - Vedecká citácia: R Core Team (2022). R: A language and environment for statistical computing. R Foundation for Statistical Computing, Vienna, Austria. URL https://www.R-project.org/.

- **forecast**: Knižnica pre časové rady a predikčné modely
  - Dokumentácia: https://pkg.robjhyndman.com/forecast/
  - Oficiálna stránka: https://github.com/robjhyndman/forecast
  - Vedecká citácia: Hyndman R, Athanasopoulos G, Bergmeir C, Caceres G, Chhay L, O'Hara-Wild M, Petropoulos F, Razbash S, Wang E, Yasmeen F (2022). _forecast: Forecasting functions for time series and linear models_. R package version 8.16, <https://pkg.robjhyndman.com/forecast/>.

- **tseries**: Knižnica pre analýzu časových radov
  - Dokumentácia: https://cran.r-project.org/web/packages/tseries/
  - Vedecká citácia: Trapletti A, Hornik K (2022). _tseries: Time Series Analysis and Computational Finance_. R package version 0.10-52.

- **rugarch**: Knižnica pre GARCH modely
  - Dokumentácia: https://cran.r-project.org/web/packages/rugarch/
  - Vedecká citácia: Ghalanos A (2022). _rugarch: Univariate GARCH models_. R package version 1.4-5.

## Ďalšie knižnice pre analýzu dát
- **ggplot2**: Knižnica pre tvorbu štatistických grafov a vizualizácií
  - Dokumentácia: https://ggplot2.tidyverse.org/
  - Oficiálna stránka: https://ggplot2.tidyverse.org/
  - Vedecká citácia: Wickham H (2016). ggplot2: Elegant Graphics for Data Analysis. Springer-Verlag New York. ISBN 978-3-319-24277-4, https://ggplot2.tidyverse.org.

- **lmtest**: Knižnica pre testovanie lineárnych regresných modelov
  - Dokumentácia: https://cran.r-project.org/web/packages/lmtest/
  - Vedecká citácia: Zeileis A, Hothorn T (2002). "Diagnostic Checking in Regression Relationships." _R News_ *2*(3), 7-10. https://CRAN.R-project.org/doc/Rnews/

- **urca**: Knižnica pre testy jednotkového koreňa a kointegrácie
  - Dokumentácia: https://cran.r-project.org/web/packages/urca/
  - Vedecká citácia: Pfaff B (2008). "Analysis of Integrated and Cointegrated Time Series with R." Second Edition. Springer, New York. ISBN 0-387-27960-1.

- **zoo**: Knižnica pre manipuláciu s nepravidelnými časovými radmi
  - Dokumentácia: https://cran.r-project.org/web/packages/zoo/
  - Vedecká citácia: Zeileis A, Grothendieck G (2005). "zoo: S3 Infrastructure for Regular and Irregular Time Series." _Journal of Statistical Software_, *14*(6), 1-27. doi:10.18637/jss.v014.i06.

- **parallel**: Základný balík pre paralelné výpočty v R (používaný balíkom rugarch)
  - Dokumentácia: https://stat.ethz.ch/R-manual/R-devel/library/parallel/doc/parallel.pdf
  - Vedecká citácia: R Core Team (2022). R: A language and environment for statistical computing. R Foundation for Statistical Computing, Vienna, Austria. URL https://www.R-project.org/.

## Metodologické zdroje
- **SARIMA**: Box-Jenkins metodológia pre sezónne časové rady
  - Vedecká citácia: Box G, Jenkins G, Reinsel G, Ljung G (2015). Time Series Analysis: Forecasting and Control. 5th Edition. Wiley. ISBN: 978-1-118-67502-1.

- **GARCH**: Generalizované autoregresívne podmienené heteroskedastické modely
  - Vedecká citácia: Bollerslev T (1986). "Generalized autoregressive conditional heteroskedasticity." _Journal of Econometrics_. 31 (3): 307–327. doi:10.1016/0304-4076(86)90063-1.
