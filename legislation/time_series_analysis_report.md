# Komplexná analýza časového radu

## Súhrn dát
- Celkový počet pozorovaní: 353
- Rozsah dátumov: 2023-04-14 až 2024-03-31
- Priemerný počet: 2.07
- Minimálny počet: 0
- Maximálny počet: 29
- Štandardná odchýlka: 5.78

## Analýza stacionarity
- **Výsledok ADF testu**: Časový rad je stacionárny

## Analýza sezónnosti
- **Zistené sezónne periódy**:
  - 7 dní: Stredná sezónnosť (sila = 0.3019)
  - 30 dní: Slabá sezónnosť (sila = 0.0694)
  - 90 dní: Stredná sezónnosť (sila = 0.3720)
- **Najsilnejšia sezónna perióda**: 90 dní

## Interpretácia vizualizácií
- **Pôvodný časový rad**: Zobrazuje surový počet legislatívnych zmien v čase, odhaľuje celkový vzor a potenciálne odchodýlky.
- **Kĺzavé štatistiky**: Kĺzavý priemer indikuje trendovú zložku, zatáľ čo kĺzavá štandardná odchýlka ukazuje volatilitu v čase.
- **Diferencovaný rad**: Prvá diferencia odstraňuje trendovú zložku, čo pomáha dosiahnuť stacionaritu. Hodnoty kolisajúce okolo nuly indikujú úspešné odstránenie trendu.
- **ACF graf**: Autokorelačná funkcia zobrazuje korelácie medzi pozorovaniami v rôznych časových oneskoreniach. Významné vrcholy v pravidelných intervaloch naznačujú sezónnosť.
- **PACF graf**: Parciálna autokorelačná funkcia pomáha identifikovať rád autoregresného modelu. Zobrazuje priamy vzťah medzi pozorovaním a jeho oneskorením.

## Odporúčania pre metódy prognózovania
- **SARIMA** (Priorita: Vysoká)
  - Zdôvodnenie: Dokáže spracovať stacionárne dáta aj sezónnosť (perióda=90)
- **Holt-Winters** (Priorita: Stredná)
  - Zdôvodnenie: Metóda exponenciálneho vyrovnávania, ktorá zohľadňuje sezónnosť
- **LSTM neurónové siete** (Priorita: Nízka)
  - Zdôvodnenie: Dokáže zachytiť komplexné nelineárne vzory v dátach
- **Random Forest alebo XGBoost** (Priorita: Nízka)
  - Zdôvodnenie: Ensemblové metódy, ktoré dokážu zachytiť nelineárne vzťahy

## Obmedzenia a ďalšie úvahy
- Analýza predpokladá, že minulé vzory budú pokračovať aj v budúcnosti.
- Externé faktory ovplyvňujúce legislatívne zmeny (napr. politické udalosti, voľby) nie sú v tejto analýze zohľadnené.
- Sila sezónnosti sa môže časom meniť, čo si vyžaduje pravidelné prehodnotenie.
- Pre optimálne prognózovanie by mohli byť zahrnuté dodatočné externé premenné ako príznaky.
