#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LSTM model pre predikciu cien akcií Mastercard

Tento skript implementuje LSTM (Long Short-Term Memory) neurónový model 
pre predikciu cien akcií spoločnosti Mastercard. Model je trénovaný na 
historických dátach a následne použitý na predikciu budúcich cien.

Autor: Sebastian Mráz
"""

# Import knižníc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.random import set_seed


def nacitaj_data(subor):
    """
    Načíta a predspracuje dáta zo súboru CSV.
    
    Args:
        subor (str): Cesta k súboru s dátami
        
    Returns:
        pd.DataFrame: Načítaný a predspracovaný DataFrame
    """
    dataset = pd.read_csv(
        subor, index_col="Date", parse_dates=["Date"]
    ).drop(["Dividends", "Stock Splits"], axis=1)
    return dataset


def zobraz_info_o_datach(dataset):
    """
    Zobrazí základné informácie o datasete.
    
    Args:
        dataset (pd.DataFrame): Dataset s cenami akcií
    """
    print(dataset.head())
    print(dataset.describe())
    print("\nRozsah dátumov v datasete:")
    print(f"Počiatočný dátum: {dataset.index.min()}")
    print(f"Koncový dátum: {dataset.index.max()}")
    print("\nChýbajúce hodnoty:")
    print(dataset.isna().sum())
    
    # Zistenie rozsahu rokov v dátach
    start_year = dataset.index.min().year
    end_year = dataset.index.max().year
    print(f"\nRozsah rokov: {start_year} až {end_year}")
    return start_year, end_year


def rozdel_data(dataset, train_years):
    """
    Rozdelí dataset na trénovaciu a testovaciu množinu podľa roku.
    
    Args:
        dataset (pd.DataFrame): Dataset s cenami akcií
        train_years (int): Rok, do ktorého (vrátane) budú dáta použité na trénovanie
        
    Returns:
        tuple: (train_mask, test_mask) - masky pre trénovacie a testovacie dáta
    """
    train_mask = dataset.index.map(lambda x: x.year <= train_years)
    test_mask = dataset.index.map(lambda x: x.year > train_years)
    print(f"\nPoužívam roky: do {train_years} na trénovanie, po {train_years} na testovanie")
    return train_mask, test_mask


def zobraz_rozdelenie_dat(dataset, train_mask, test_mask, train_years):
    """
    Zobrazí graf rozdelenia dát na trénovaciu a testovaciu množinu.
    
    Args:
        dataset (pd.DataFrame): Dataset s cenami akcií
        train_mask (pd.Series): Maska pre trénovacie dáta
        test_mask (pd.Series): Maska pre testovacie dáta
        train_years (int): Rok rozdelenia dát
    """
    dataset.loc[train_mask, "High"].plot(figsize=(16, 4), legend=True)
    dataset.loc[test_mask, "High"].plot(figsize=(16, 4), legend=True)
    plt.legend([f"Trénovacie dáta (do {train_years})", f"Testovacie dáta (po {train_years})"])
    plt.title("Cena akcií MasterCard")
    plt.show()


def extrahuj_data(dataset, train_mask, test_mask):
    """
    Extrahuje hodnoty z datasetu podľa masiek.
    
    Args:
        dataset (pd.DataFrame): Dataset s cenami akcií
        train_mask (pd.Series): Maska pre trénovacie dáta
        test_mask (pd.Series): Maska pre testovacie dáta
        
    Returns:
        tuple: (training_set, test_set) - trénovacie a testovacie hodnoty
    """
    train = dataset.loc[train_mask, "High"].values
    test = dataset.loc[test_mask, "High"].values
    return train, test


def priprav_sekvencie(sequence, n_steps):
    """
    Pripraví sekvencie pre LSTM model. Táto funkcia rozdelí časový rad na prekrývajúce sa sekvencie,
    kde každá sekvencia má dĺžku n_steps a cieľová hodnota je hodnota nasledujúca po sekvencii.
    
    Args:
        sequence (np.array): Postupnosť hodnôt časového radu (normalizovaná)
        n_steps (int): Počet krokov v jednej sekvencii - určuje, koľko predchádzajúcich hodnôt
                       sa použije na predikciu nasledujúcej hodnoty. Hodnota 60 znamená, že
                       sa použije 60 predchádzajúcich dní na predikciu nasledujúceho dňa.
        
    Returns:
        tuple: (X, y) - vstupné sekvencie a cieľové hodnoty, kde:
               X je pole tvarou (počet_sekvencií, n_steps)
               y je pole cieľových hodnôt tvarom (počet_sekvencií,)
    """
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def vytvor_lstm_model(n_steps, features):
    """
    Vytvorí a skompiluje LSTM model pre predikciu časového radu.
    
    Args:
        n_steps (int): Počet časových krokov v sekvencii - určuje veľkosť vstupného okna
                       (počet predchádzajúcich hodnôt použitých na predikciu)
        features (int): Počet vstupných vlastností - v našom prípade 1, pretože používame
                        len hodnotu 'High' (najvyššia cena akcie v daný deň)
        
    Returns:
        Sequential: Skompilovaný LSTM model s nasledujúcou architektúrou:
                   - LSTM vrstva so 125 neurónmi a aktivačnou funkciou tanh
                   - Výstupná Dense vrstva s 1 neurónom (predikcia jednej hodnoty)
                   - Optimalizátor: RMSprop
                   - Stratová funkcia: MSE (Mean Squared Error)
    """
    model = Sequential()
    model.add(LSTM(units=125, activation="tanh", input_shape=(n_steps, features)))
    model.add(Dense(units=1))
    model.compile(optimizer="RMSprop", loss="mse")
    return model


def vytvor_gru_model(n_steps, features):
    """
    Vytvorí a skompiluje GRU model pre predikciu časového radu.
    
    GRU (Gated Recurrent Unit) je typ rekurentnej neurónovej siete podobný LSTM,
    ale s jednoduchšou architektúrou a menším počtom parametrov, čo môže viesť
    k rýchlejšiemu trénovaniu a niekedy k lepším výsledkom pri menších datasetoch.
    
    Args:
        n_steps (int): Počet časových krokov v sekvencii - určuje veľkosť vstupného okna
                       (počet predchádzajúcich hodnôt použitých na predikciu)
        features (int): Počet vstupných vlastností - v našom prípade 1, pretože používame
                        len hodnotu 'High' (najvyššia cena akcie v daný deň)
        
    Returns:
        Sequential: Skompilovaný GRU model s nasledujúcou architektúrou:
                   - GRU vrstva so 125 neurónmi a aktivačnou funkciou tanh
                   - Výstupná Dense vrstva s 1 neurónom (predikcia jednej hodnoty)
                   - Optimalizátor: RMSprop
                   - Stratová funkcia: MSE (Mean Squared Error)
    """
    model = Sequential()
    model.add(GRU(units=125, activation="tanh", input_shape=(n_steps, features)))
    model.add(Dense(units=1))
    model.compile(optimizer="RMSprop", loss="mse")
    return model


def zobraz_predikcie(test, predicted):
    """
    Zobrazí graf porovnania skutočných a predikovaných hodnôt cien akcií.
    Graf zobrazuje dve krivky - šedú pre skutočné hodnoty a červenú pre predikované hodnoty,
    čo umožňuje vizuálne porovnanie presnosti predikcie modelu.
    
    Args:
        test (np.array): Skutočné hodnoty cien akcií z testovacieho obdobia
        predicted (np.array): Predikované hodnoty cien akcií z modelu LSTM
    """
    plt.figure(figsize=(16, 8))
    plt.plot(test, color="gray", label="Skutočné hodnoty")
    plt.plot(predicted, color="red", label="Predikované hodnoty")
    plt.title("Predikcia cien akcií MasterCard")
    plt.xlabel("Čas")
    plt.ylabel("Cena akcií MasterCard")
    plt.legend()
    plt.show()


def vypocitaj_metriky(test, predicted):
    """
    Vypočíta a zobrazí metriky presnosti predikcie modelu LSTM.
    RMSE (Root Mean Squared Error) je štandardná metrika pre hodnotenie presnosti predikcie
    v regresných modeloch. Nižšia hodnota RMSE znamená presnejšiu predikciu.
    
    Na základe pamäte o Holt-Winters modeli, kde najlepší model dosiahol RMSE 4.9642,
    môžeme porovnať výkon LSTM modelu s týmto benchmarkom.
    
    Args:
        test (np.array): Skutočné hodnoty cien akcií
        predicted (np.array): Predikované hodnoty cien akcií
        
    Returns:
        float: RMSE (Root Mean Squared Error) - odmocnina z priemernej kvadratickej chyby
    """
    rmse = np.sqrt(mean_squared_error(test, predicted))
    print("Stredná kvadratická chyba (RMSE) je {:.2f}.".format(rmse))
    return rmse


def main():
    """
    Hlavná funkcia, ktorá riadi celý proces predikcie cien akcií pomocou LSTM a GRU modelov.
    
    Proces zahŕňa nasledujúce kroky:
    1. Načítanie a analýza historických dát cien akcií Mastercard
    2. Rozdelenie dát na trénovaciu (70%) a testovaciu (30%) množinu podľa rokov
    3. Príprava dát pre modely (normalizácia, vytvorenie sekvencií)
    4. Vytvorenie a trénovanie LSTM a GRU modelov s nasledujúcimi parametrami:
       - 125 neurónov v rekurentnej vrstve
       - 50 epoch trénovania
       - Veľkosť dávky (batch size): 32
       - Časové okno (n_steps): 60 dní
    5. Predikcia cien akcií na testovacej množine pomocou oboch modelov
    6. Vyhodnotenie a porovnanie presnosti modelov pomocou RMSE
    """
    
    # Nastavenie seedu pre reprodukovateľnosť
    set_seed(455)
    np.random.seed(455)
    
    # Načítanie dát
    dataset = nacitaj_data("Mastercard_stock_history.csv")
    
    # Zobrazenie informácií o dátach
    start_year, end_year = zobraz_info_o_datach(dataset)
    
    # Rozdelenie dát na trénovacie a testovacie
    train_years = int((end_year - start_year) * 0.7) + start_year
    train_mask, test_mask = rozdel_data(dataset, train_years)
    
    # Zobrazenie rozdelenia dát
    zobraz_rozdelenie_dat(dataset, train_mask, test_mask, train_years)
    
    # Extrakcia hodnôt
    training_set, test_set = extrahuj_data(dataset, train_mask, test_mask)
    
    # Škálovanie dát
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set = training_set.reshape(-1, 1)
    training_set_scaled = sc.fit_transform(training_set)
    
    # Parametre modelu
    n_steps = 60  # Počet časových krokov v jednej sekvencii (60 dní histórie)
    features = 1  # Počet vstupných vlastností (používame len stĺpec 'High')
    
    # Príprava sekvencií pre trénovanie
    X_train, y_train = priprav_sekvencie(training_set_scaled, n_steps)
    # Reshape na tvar [vzorky, časové_kroky, vlastnosti] požadovaný LSTM vrstvou
    # Príklad: Ak máme 1000 sekvencií, každá s dĺžkou 60 dní a 1 vlastnosťou (cena),
    # výsledný tvar bude (1000, 60, 1)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], features)
    
    # Vytvorenie a trénovanie LSTM modelu
    print("\n=== LSTM Model ===\n")
    model_lstm = vytvor_lstm_model(n_steps, features)
    model_lstm.summary()  # Výpis architektúry modelu
    # Trénovanie modelu s nasledujúcimi parametrami:
    # - epochs=50: model sa naučí na celom trénovacom datasete 50-krát
    # - batch_size=32: váhy sa aktualizujú po každých 32 vzorkách
    # Tieto parametre sú štandardné hodnoty, ktoré zvyčajne poskytujú dobrý kompromis
    # medzi rýchlosťou trénovania a kvalitou modelu
    model_lstm.fit(X_train, y_train, epochs=50, batch_size=32)
    
    # Príprava testovacích dát
    dataset_total = dataset.loc[:, "High"]  # Extrakcia stĺpca 'High' z celého datasetu
    # Získanie relevantných vstupov pre testovanie - potrebujeme aj n_steps predchádzajúcich hodnôt
    # pred začiatkom testovacieho obdobia, aby sme mohli predpovedať prvé hodnoty
    inputs = dataset_total[len(dataset_total) - len(test_set) - n_steps:].values
    inputs = inputs.reshape(-1, 1)  # Reshape na 2D pole požadované MinMaxScaler-om
    inputs = sc.transform(inputs)  # Škálovanie vstupov pomocou rovnakého scalera ako pri trénovaní
    
    # Príprava sekvencií pre testovanie
    X_test, y_test = priprav_sekvencie(inputs, n_steps)  # Vytvorenie sekvencií rovnakým spôsobom ako pri trénovaní
    # Reshape na tvar [vzorky, časové_kroky, vlastnosti] požadovaný LSTM vrstvou
    # Príklad: Ak máme 200 testovacích sekvencií, každá s dĺžkou 7 dní a 1 vlastnosťou,
    # výsledný tvar bude (200, 7, 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], features)
    
    # Predikcia pomocou LSTM modelu
    # Použitie natrénovaného modelu na predikciu hodnôt na testovacej množine
    predicted_stock_price_lstm = model_lstm.predict(X_test)
    # Inverzná transformácia normalizovaných predikovaných hodnôt späť na pôvodnú škálu cien
    # Toto je dôležité pre interpretáciu výsledkov a porovnanie s reálnymi cenami
    predicted_stock_price_lstm = sc.inverse_transform(predicted_stock_price_lstm)
    
    # Zobrazenie výsledkov LSTM modelu
    print("\nVýsledky LSTM modelu:")
    zobraz_predikcie(test_set, predicted_stock_price_lstm)
    lstm_rmse = vypocitaj_metriky(test_set, predicted_stock_price_lstm)
    
    # Vytvorenie a trénovanie GRU modelu
    print("\n=== GRU Model ===\n")
    model_gru = vytvor_gru_model(n_steps, features)
    model_gru.summary()  # Výpis architektúry modelu
    # Trénovanie GRU modelu s rovnakými parametrami ako LSTM model
    model_gru.fit(X_train, y_train, epochs=50, batch_size=32)
    
    # Predikcia pomocou GRU modelu
    predicted_stock_price_gru = model_gru.predict(X_test)
    predicted_stock_price_gru = sc.inverse_transform(predicted_stock_price_gru)
    
    # Zobrazenie výsledkov GRU modelu
    print("\nVýsledky GRU modelu:")
    zobraz_predikcie(test_set, predicted_stock_price_gru)
    gru_rmse = vypocitaj_metriky(test_set, predicted_stock_price_gru)
    
    # Porovnanie modelov
    print("\n=== Porovnanie modelov ===\n")
    print(f"LSTM model RMSE: {lstm_rmse:.2f}")
    print(f"GRU model RMSE: {gru_rmse:.2f}")
    
    if lstm_rmse < gru_rmse:
        print("\nLSTM model dosiahol lepšie výsledky ako GRU model.")
    elif gru_rmse < lstm_rmse:
        print("\nGRU model dosiahol lepšie výsledky ako LSTM model.")
    else:
        print("\nLSTM a GRU modely dosiahli rovnaké výsledky.")

if __name__ == "__main__":
    main()
