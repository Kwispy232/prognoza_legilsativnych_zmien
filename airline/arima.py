# Only enable critical logging (Optional)
import os
os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "CRITICAL"

import time
import numpy as np
import pandas as pd

from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment

y = get_data('airline', verbose=False)

# We want to forecast the next 12 months of data and we will use 3 fold cross-validation to test the models.
fh = 12 # or alternately fh = np.arange(1,13)
fold = 3

fig_kwargs = {
    # "renderer": "notebook",
    "renderer": "png",
    "width": 1000,
    "height": 600,
}

exp = TSForecastingExperiment()
exp.setup(data=y, fh=fh, fold=fold, fig_kwargs=fig_kwargs, session_id=42)

# Vytvoríme ARIMA model s nasledujúcimi parametrami:
# - order=(1,1,0): 
#   p=1 (autoregresívny rád) - počet oneskorených pozorovaní použitých v modeli
#   d=1 (diferenciácia) - počet krát, koľko sa časový rad diferencuje na dosiahnutie stacionarity
#   q=0 (pohyblivý priemer) - počet oneskorených chýb predpovede použitých v modeli
# - seasonal_order=(0,1,0,12): 
#   P=0 (sezónny AR) - počet sezónnych autoregresívnych členov
#   D=1 (sezónna diferenciácia) - počet sezónnych diferencií
#   Q=0 (sezónny MA) - počet sezónnych pohyblivých priemerov
#   s=12 (sezónna perióda - mesačné dáta) - dĺžka sezónneho cyklu
model = exp.create_model("arima", order=(1,1,0), seasonal_order=(0,1,0,12))

# Out-of-sample Forecasts
y_predict = exp.predict_model(model)
y_predict

# Plot the out-of-sample forecasts
exp.plot_model(estimator=model, save = "airline")

# # Alternately the following will plot the same thing.
# exp.plot_model(estimator=model, plot="forecast")

exp.check_stats(model)

exp.plot_model(model, plot='diagnostics', fig_kwargs={"height": 800, "width": 1000}, save = "airline")
exp.plot_model(model, plot='insample', save = "airline")
exp.plot_model(model, plot="decomp", save = "airline")
exp.plot_model(model, plot="decomp_stl", save = "airline")

model = exp.create_model("lightgbm_cds_dt")
y_predict = exp.predict_model(model)
exp.plot_model(estimator=model, save = "airline")

tuned_model = exp.tune_model(model)
exp.plot_model(estimator=tuned_model, save = "airline")

print(model)
print(tuned_model)

exp.plot_model([model, tuned_model], data_kwargs={"labels": ["Baseline", "Tuned"]}, save = "airline")

final_model = exp.finalize_model(tuned_model)
exp.plot_model(final_model, save = "airline")
exp.predict_model(final_model)

print(tuned_model)
print(final_model)

_ = exp.save_model(final_model, "arima_airline")
