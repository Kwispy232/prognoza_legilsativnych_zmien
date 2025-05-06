import os
import pandas as pd
import numpy as np
from pycaret.time_series import *


def load_data(filename='legislation/unique_dates_counts.csv'):
    """Načítanie a príprava údajov časového radu
    
    Táto funkcia načíta údaje zo súboru CSV, skonvertuje stĺpec s dátumami
    na formát datetime, nastaví dátum ako index a zoradí dáta podľa dátumu.
    
    Parametre:
    ----------
    filename : str, voliteľný
        Cesta k súboru CSV s údajmi (predvolene 'unique_dates_counts.csv')
        
    Návratová hodnota:
    -----------------
    pandas.DataFrame
        Dataframe s načítanými a pripravenými údajmi
    """
    print(f"Loading dataset: {filename}")
    df = pd.read_csv(filename)
    
    # Konverzia dátumu na formát datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Nastavenie dátumu ako indexu
    df = df.set_index('Date')
    
    # Zoradenie podľa dátumu
    df = df.sort_index()
    
    # Zobrazenie základných informácií o dátach
    print("\nData Overview:")
    print(f"Date Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"Number of observations: {len(df)}")
    print(f"Data frequency: {pd.infer_freq(df.index) or 'D'}")

    # df['Date'] = df.index.strftime('%Y-%m-%d')

    df['Month'] = df.index.month
    df['Year'] = df.index.year

    df['Series'] = np.arange(1,len(df)+1)

    df = df[['Series', 'Year', 'Month', 'Count']] 

    return df

data = load_data()
data.head()

train = data[data['Year'] < 2024]
test = data[data['Year'] >= 2024]
fig_kwargs = {
    # "renderer": "notebook",
    "renderer": "png",
    "width": 1000,
    "height": 600,
}


# eda = TSForecastingExperiment()
# eda.setup(data=y, fh=fh, fig_kwargs=fig_kwargs)
s = setup(data = data, target = 'Count', fh=30, max_sp_to_consider = 90, remove_harmonics = False, fig_kwargs=fig_kwargs)
best = compare_models()

output_dir = 'legislation/visualizations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plot_model(save = output_dir)
# plot_model(data_kwargs={"plot_data_type": ["original", "imputed", "transformed"]})
# plot_model(plot="acf")
# plot_model(plot="pacf", data_kwargs={'nlags':36}, fig_kwargs={'height': 500, "width": 800})
# plot_model(plot="periodogram")
# plot_model(plot="fft")
# plot_model(plot="diagnostics", fig_kwargs={"height": 800, "width": 1000})
# plot_model(
#     plot="diff",
#     data_kwargs={"lags_list": [[1], [1, 12]], "acf": True, "pacf": True, "periodogram": True},
#     fig_kwargs={"height": 800, "width": 1500}
# )
# plot_model(plot="decomp", fig_kwargs={"height": 500})
# plot_model(plot="train_test_split", fig_kwargs={"height": 400, "width": 900})
# plot_model(plot="cv", fig_kwargs={"height": 400, "width": 900})