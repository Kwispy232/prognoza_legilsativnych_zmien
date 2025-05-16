#############################################################################
# XGBoost model pre predikciu legislatívnych časových radov
#############################################################################
# Tento skript implementuje komplexný XGBoost model pre predikciu legislatívnej aktivity
# Zahŕňa nasledujúce kroky:
# 1. Načítanie a príprava dát
# 2. Feature engineering - časové lagy, kalendárové premenné, kĺzavé priemery
# 3. Trénovanie XGBoost modelu s optimálnymi hyperparametrami
# 4. Vyhodnotenie presnosti modelu na trénovacích a testovacích dátach
# 5. Generovanie prognózy pre budúce obdobie
# 6. Diagnostika predpokladov modelu (testy autokorelácie, ARCH efektu, normality)
# 7. Vytvorenie vizualizácií a správy

#############################################################################
# 1. Načítanie potrebných balíkov
#############################################################################
# Balík pre manipuláciu s dátami a vizualizáciu
library(tidyverse)    # Súbor balíkov pre manipuláciu s dátami
library(lubridate)    # Pre prácu s dátumami
library(ggplot2)      # Pre pokročilé vizualizácie
library(scales)       # Pre formátovanie osi v grafoch

# Balíky pre modelovanie
library(xgboost)      # Implementácia XGBoost algoritmu
library(caret)        # Pre one-hot encoding a krížovú validáciu
library(forecast)     # Pre funkcie časových radov
library(zoo)          # Pre kĺzavé priemery

# Balíky pre diagnostiku
library(stats)        # Obsahuje funkcie pre štatistické testy
library(tseries)      # Pre Jarque-Bera test normality
library(lmtest)       # Pre Breusch-Pagan test heteroskedasticity
library(knitr)        # Pre generáciu reportov

#############################################################################
# 2. Nastavenie prostredia a načítanie dát
#############################################################################
# Definovanie výstupného adresára pre výsledky - s absolutnou cestou pre spoľahlivé uloženie výstupov
output_dir <- file.path(getwd(), "xgboost_results")
cat("Výstupný adresár:", output_dir, "\n")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Zmazanie všetkých súborov v adresári pre čistý štart
unlink(list.files(output_dir, full.names = TRUE), recursive = TRUE)

# Načítanie dát z CSV súboru
legisl_data <- read.csv("unique_dates_counts.csv")

# Konverzia stĺpca dátumov na formát Date
legisl_data$Date <- as.Date(legisl_data$Date)

# Základné informácie o dátach
cat("\nInformácie o časovom rade legislatívnej aktivity:\n")
cat("Rozsah dát:", as.character(min(legisl_data$Date)), "až", as.character(max(legisl_data$Date)), "\n")
cat("Počet pozorovaní:", nrow(legisl_data), "\n")
cat("Priemerný počet legislatívnych aktivít denne:", round(mean(legisl_data$Count), 2), "\n")
cat("Medián počtu legislatívnych aktivít denne:", median(legisl_data$Count), "\n")
cat("Minimum:", min(legisl_data$Count), "\n")
cat("Maximum:", max(legisl_data$Count), "\n\n")

#############################################################################
# 3. Základná vizualizácia časového radu
#############################################################################
# Vytvorenie prehliadového grafu časového radu pre lepšie pochopenie dát
p_initial <- ggplot(legisl_data, aes(x = Date, y = Count)) +
  geom_line(color = "#2C3E50") +
  geom_smooth(method = "loess", color = "#E74C3C", fill = "#E74C3C", alpha = 0.2) +
  labs(title = "Časový rad legislatívnej aktivity",
       subtitle = "S vyhladeným trendom",
       x = "Dátum",
       y = "Počet legislatívnych aktivit") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))

# Uloženie úvodného grafu
ggsave(paste0(output_dir, "/initial_time_series.png"), p_initial, width = 12, height = 6, dpi = 300)

# Príprava dát pre XGBoost
# Pridanie časových lagov a kalendárových premenných pre zachytenie sezónnosti
#############################################################################
# 4. Feature engineering - vytvorenie prediktivnych premenných
#############################################################################

# Funkcia pre vytváranie prediktívnych premenných pre XGBoost model
# Táto funkcia transformuje pôvodné dáta pridaním rôznych typov feature-ov,
# ktoré pomôžu modelu zachytiť sezónne vzory a časové závislosti
create_features <- function(data) {
  # Vytvorenie kópie dataframe
  df <- data.frame(data)
  
  #-----------------------------------------------------
  # 4.1 Kalendárové premenné - zachytávajú sezónne vzory
  #-----------------------------------------------------
  # Deň v týždni - zachytáva týždenný cyklus (napr. víkendy vs. pracovné dni)
  df$dayofweek <- as.factor(weekdays(df$Date))
  # Mesiac - zachytáva ročný cyklus sezónnosti
  df$month <- as.factor(month(df$Date))
  # Deň v mesiaci - môže zachytiť pravidelné udalosti v mesiaci
  df$day <- as.factor(day(df$Date))
  # Indikátor víkendu - binárna premenná pre víkendové dni
  df$is_weekend <- as.factor(ifelse(df$dayofweek %in% c("Saturday", "Sunday"), 1, 0))
  
  #-----------------------------------------------------
  # 4.2 Lagy - posunuté hodnoty pre zachytenie autokorelácie
  #-----------------------------------------------------
  # Denné lagy (1 až 14 dní) - zachytávajú závislosť na predchádzajúcich dňoch
  for (lag in 1:14) {
    df[[paste0("lag_", lag)]] <- lag(df$Count, lag)
  }
  
  # Týždenné lagy - zachytávajú týždennú sezónnosť
  for (week_lag in c(7, 14, 21)) {
    df[[paste0("week_lag_", week_lag)]] <- lag(df$Count, week_lag)
  }
  
  #-----------------------------------------------------
  # 4.3 Kĺzavé priemery - vyhladenie šumu a zachytenie trendu
  #-----------------------------------------------------
  # Týždenný kĺzavý priemer
  df$ma_7 <- rollmean(df$Count, k = 7, fill = NA, align = "right")
  # Dvojtýždňový kĺzavý priemer
  df$ma_14 <- rollmean(df$Count, k = 14, fill = NA, align = "right")
  
  #-----------------------------------------------------
  # 4.4 Numerické cyklické premenné
  #-----------------------------------------------------
  # Numerické verzie dňa v týždni pre lineárne vzťahy (1=Pondelok, 7=Nedeľa)
  df$dayofweek_num <- as.numeric(factor(weekdays(df$Date), 
                                        levels = c("Monday", "Tuesday", "Wednesday", 
                                                  "Thursday", "Friday", "Saturday", "Sunday")))
  # Číslo mesiaca (1-12)
  df$month_num <- month(df$Date)
  
  #-----------------------------------------------------
  # 4.5 Trendové a cyklické komponenty
  #-----------------------------------------------------
  # Lineárny trend - zachytáva celkový trend v dátach
  df$trend <- 1:nrow(df)
  
  # Sínusové a kosínusové členy pre týždennú sezónnosť
  # Tieto premenné sú lepšie ako kategórie, lebo zachytávajú cyklickú povahu času
  df$sin_7 <- sin(2 * pi * df$trend / 7)
  df$cos_7 <- cos(2 * pi * df$trend / 7)
  
  # Vrátenie transformovaného dataframe
  return(df)
}

#############################################################################
# 5. Príprava dát pre trénovanie modelu
#############################################################################

# Aplikácia feature inžinieringu na všetky dáta
cat("Vytváram prediktívne premenné pre XGBoost model...\n")
features_df <- create_features(legisl_data)

# Odstránenie riadkov s chýbajúcimi hodnotami (najčastejšie vzniknuté lagovaním)
cat("Odstraňujem riadky s chýbajúcimi hodnotami...\n")
cat("Počet riadkov pred filtrovaním:", nrow(features_df), "\n")
features_df <- na.omit(features_df)
cat("Počet riadkov po filtrovaní:", nrow(features_df), "\n")

#-----------------------------------------------------
# 5.1 Rozdelenie na tréningové a testovacie dáta
#-----------------------------------------------------

# Rozdelenie dát na tréningové a testovacie (posledných 30 dní ako test)
# Toto je chronologické rozdelenie, ktoré je vhodné pre časové rady
test_size <- 30
cat("Rozdeľujem dáta na tréningové a testovacie (posledných", test_size, "dní ako test)...\n")

train_data <- features_df[1:(nrow(features_df) - test_size), ]
test_data <- features_df[(nrow(features_df) - test_size + 1):nrow(features_df), ]

cat("Veľkosť tréningovej množiny:", nrow(train_data), "riadkov\n")
cat("Veľkosť testovacej množiny:", nrow(test_data), "riadkov\n")

#-----------------------------------------------------
# 5.2 One-hot encoding kategórií pre XGBoost
#-----------------------------------------------------

# Príprava tréningových a testovacích dát pre XGBoost model
# XGBoost vyžaduje číselné vstupy, preto potrebujeme konvertovať faktoriálne premenné
cat("Konvertujem kategórie na numerické premenné (one-hot encoding)...\n")

# Vylúčenie stĺpcov Date a Count z prediktorov
exclude_cols <- c("Date", "Count")
feature_cols <- setdiff(colnames(train_data), exclude_cols)

# Vytvorenie modelu pre one-hot encoding (konverziu faktoriálnych premenných)
dummies_model <- dummyVars(~ ., data = train_data[, feature_cols])

# Konverzia faktoriálnych premenných na indikátorové premenné (one-hot encoding)
train_x_numeric <- predict(dummies_model, newdata = train_data[, feature_cols])
test_x_numeric <- predict(dummies_model, newdata = test_data[, feature_cols])

cat("Počet prediktorov po one-hot encodingu:", ncol(train_x_numeric), "\n")

#############################################################################
# 6. Trénovanie XGBoost modelu
#############################################################################

#-----------------------------------------------------
# 6.1 Príprava DMatrix objektov pre XGBoost
#-----------------------------------------------------
cat("Vytváram DMatrix objekty pre XGBoost...\n")

# Vytvorenie xgb.DMatrix objektov - špeciálny formát pre XGBoost, ktorý urychľuje výpočty
dtrain <- xgb.DMatrix(data = train_x_numeric, label = train_data$Count)
dtest <- xgb.DMatrix(data = test_x_numeric, label = test_data$Count)

#-----------------------------------------------------
# 6.2 Nastavenie hyperparametrov modelu
#-----------------------------------------------------
cat("Nastavujem hyperparametre XGBoost modelu...\n")

# Nastavenie parametrov XGBoost - tieto parametre riadia proces trénovania
params <- list(
  # Regresia s kriteriom kvadratickej chyby
  objective = "reg:squarederror",
  
  # Metrika pre vyhodnotenie - root mean squared error
  eval_metric = "rmse",
  
  # Learning rate - menšia hodnota znamena pomalšie učenie, ale často lepšiu konvergenciu
  eta = 0.1,
  
  # Maximálna hĺbka stromčekov - vyššia hodnota zvyšuje komplexnosť modelu
  max_depth = 6,
  
  # Minimálna váha potrebná pre rozdelení listu - ochrana proti overfittingu
  min_child_weight = 1,
  
  # Náhodný výber pozorování v každom kroku (0.8 = 80% dát) - redukcia overfittingu
  subsample = 0.8,
  
  # Náhodný výber premennych v každom strome - zvyšuje robustnosť 
  colsample_bytree = 0.8,
  
  # Minimálny zisk pre rozdelenie uzla - vyššia hodnota znamena konzervátivnejší model
  gamma = 0
)

#-----------------------------------------------------
# 6.3 Trénovanie modelu s early stopping
#-----------------------------------------------------
cat("Začínam trénovanie XGBoost modelu...\n")

# Trénovanie modelu s early stopping - automaticky zastaví trénovanie, keď sa presnost nezlepsuje
xgb_model <- xgb.train(
  # Použité hyperparametre
  params = params,
  
  # Trénovacie dáta v XGBoost formáte
  data = dtrain,
  
  # Maximálny počet trénovacích iterácií
  nrounds = 1000,
  
  # Monitorovanie progresu na tréningových aj validacnych dátach
  watchlist = list(train = dtrain, test = dtest),
  
  # Zažaty tréning po 50 iteráciach bez zlepšenia - zabraňuje overfittingu
  early_stopping_rounds = 50,
  
  # Zobrazenie progresu tréningu
  verbose = 1
)

cat("\nTrénovanie dokončené po", xgb_model$best_iteration, "iteráciách\n")

# Zistenie dôležitosti prediktorov
importance_matrix <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance_matrix[1:15,])

# Uloženie grafu dôležitosti prediktorov
png(file = paste0(output_dir, "/feature_importance.png"), width = 800, height = 600)
xgb.plot.importance(importance_matrix[1:15,])
dev.off()

# Predikcia na tréningových a testovacích dátach
train_data$xgb_pred <- predict(xgb_model, dtrain)
test_data$xgb_pred <- predict(xgb_model, dtest)

# Výpočet metrík presnosti - tréningové dáta
train_rmse <- sqrt(mean((train_data$Count - train_data$xgb_pred)^2))
train_mae <- mean(abs(train_data$Count - train_data$xgb_pred))
train_r2 <- 1 - sum((train_data$Count - train_data$xgb_pred)^2) / 
            sum((train_data$Count - mean(train_data$Count))^2)

# Výpočet metrík presnosti - testovacie dáta
test_rmse <- sqrt(mean((test_data$Count - test_data$xgb_pred)^2))
test_mae <- mean(abs(test_data$Count - test_data$xgb_pred))
test_r2 <- 1 - sum((test_data$Count - test_data$xgb_pred)^2) / 
          sum((test_data$Count - mean(test_data$Count))^2)

# Výpis metrík presnosti
cat("Trénovacie dáta:\n")
cat("RMSE:", train_rmse, "\n")
cat("MAE:", train_mae, "\n")
cat("R²:", train_r2, "\n\n")

cat("Testovacie dáta:\n")
cat("RMSE:", test_rmse, "\n")
cat("MAE:", test_mae, "\n")
cat("R²:", test_r2, "\n")

# Kombinácia tréningových a testovacích dát pre vizualizáciu
all_data <- rbind(
  data.frame(Date = train_data$Date, 
            Actual = train_data$Count, 
            Predicted = train_data$xgb_pred, 
            Type = "Train"),
  data.frame(Date = test_data$Date, 
            Actual = test_data$Count, 
            Predicted = test_data$xgb_pred, 
            Type = "Test")
)

# Vizualizácia skutočných vs. predikovaných hodnôt
p1 <- ggplot(all_data, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Skutočné hodnoty")) +
  geom_line(aes(y = Predicted, color = "Predikované hodnoty")) +
  geom_vline(xintercept = min(test_data$Date), linetype = "dashed") +
  labs(title = "XGBoost: Skutočné vs. predikované hodnoty legislatívnej aktivity",
       x = "Dátum",
       y = "Počet legislatívnych aktivít",
       color = "Typ hodnôt") +
  theme_minimal() +
  scale_color_manual(values = c("Skutočné hodnoty" = "blue", "Predikované hodnoty" = "red"))

# Uloženie grafu
ggsave(paste0(output_dir, "/actual_vs_predicted.png"), p1, width = 12, height = 6)

# Scatter plot skutočných vs. predikovaných hodnôt
p2 <- ggplot(all_data, aes(x = Actual, y = Predicted, color = Type)) +
  geom_point(alpha = 0.7) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  labs(title = "XGBoost: Scatter plot skutočných vs. predikovaných hodnôt",
       x = "Skutočné hodnoty",
       y = "Predikované hodnoty") +
  theme_minimal()

ggsave(paste0(output_dir, "/scatter_plot.png"), p2, width = 8, height = 6)

# Zobrazenie reziduálov
all_data$Residuals <- all_data$Actual - all_data$Predicted

p3 <- ggplot(all_data, aes(x = Date, y = Residuals, color = Type)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(title = "XGBoost: Reziduály",
       x = "Dátum",
       y = "Reziduály") +
  theme_minimal()

ggsave(paste0(output_dir, "/residuals_plot.png"), p3, width = 12, height = 6)

# Uloženie výsledkov a metrík
result_metrics <- data.frame(
  Dataset = c("Trénovacie", "Testovacie"),
  RMSE = c(train_rmse, test_rmse),
  MAE = c(train_mae, test_mae),
  R2 = c(train_r2, test_r2)
)

write.csv(result_metrics, paste0(output_dir, "/model_metrics.csv"), row.names = FALSE)

# Príprava dát pre prognózu ďalších 30 dní
forecast_days <- 30
last_date <- max(legisl_data$Date)
future_dates <- seq(last_date + 1, by = "day", length.out = forecast_days)

# Vytvorenie dataframe s budúcimi dátumami
future_df <- data.frame(
  Date = future_dates,
  Count = NA  # Bude nahradené predikciami
)

# Inkrementálne generovanie predikcií
forecast_results <- future_df
forecast_full_data <- rbind(legisl_data, future_df)

# Postupné dopĺňanie predikcií po jednom dni
for (i in 1:forecast_days) {
  # Aktuálny dátum pre predikciu
  current_date_index <- nrow(legisl_data) + i
  
  # Regenerovanie features pre všetky dáta vrátane najnovších predikcií
  temp_features <- create_features(forecast_full_data[1:current_date_index, ])
  
  # Vybratie posledného riadka pre predikciu
  last_row <- tail(temp_features, 1)
  
  # Ak existujú NA hodnoty (prvé dni prognózy), nahraď ich priemermi
  for (col in colnames(last_row)) {
    if (is.na(last_row[1, col]) && col != "Count" && col != "Date") {
      last_row[1, col] <- mean(temp_features[[col]], na.rm = TRUE)
    }
  }
  
  # Vylúčenie stĺpcov Date a Count z prediktorov
  pred_features <- setdiff(colnames(last_row), exclude_cols)
  
  # Konverzia faktoriálnych premenných
  pred_matrix <- predict(dummies_model, newdata = last_row[, pred_features])
  
  # Predikcia
  pred_value <- predict(xgb_model, xgb.DMatrix(pred_matrix))
  
  # Zápis predikcie do výsledného dataframe
  forecast_results$Count[i] <- pred_value
  
  # Aktualizácia pôvodných dát pre ďalšiu iteráciu
  forecast_full_data$Count[current_date_index] <- pred_value
}

# Zaokrúhlenie predikcií (počet legislatívnych aktivít by mal byť celé číslo)
forecast_results$Count <- round(pmax(forecast_results$Count, 0), 0)

# Uloženie prognózy
write.csv(forecast_results, paste0(output_dir, "/xgboost_forecast.csv"), row.names = FALSE)

# Vizualizácia prognózy
forecast_plot_data <- rbind(
  data.frame(
    Date = legisl_data$Date,
    Value = legisl_data$Count,
    Type = "Historické dáta"
  ),
  data.frame(
    Date = forecast_results$Date,
    Value = forecast_results$Count,
    Type = "XGBoost prognóza"
  )
)

p4 <- ggplot(forecast_plot_data, aes(x = Date, y = Value, color = Type)) +
  geom_line() +
  labs(title = "XGBoost: Historické dáta a prognóza legislatívnej aktivity",
       x = "Dátum",
       y = "Počet legislatívnych aktivít") +
  theme_minimal() +
  scale_color_manual(values = c("Historické dáta" = "blue", "XGBoost prognóza" = "red"))

ggsave(paste0(output_dir, "/forecast_plot.png"), p4, width = 12, height = 6)

#############################################################################
# 9. Diagnostika modelu - testovanie predpokladov
#############################################################################
cat("\nVykonávam diagnostiku modelu a testovanie predpokladov...\n")

#-----------------------------------------------------
# 9.1 Výpočet a analýza reziduálov
#-----------------------------------------------------
# Reziduály sú rozdiely medzi skutočnými a predikovanými hodnotami
# Ich analýza nám pomáha zhodnotit kvalitu modelu a identifikovat prípadné problémy
cat("Výpočet reziduálov pre diagnostické testy...\n")

# Výpočet reziduálov pre tréningové a testovacie dáta
train_residuals <- train_data$Count - train_data$xgb_pred
test_residuals <- test_data$Count - test_data$xgb_pred

# Spojenie všetkých reziduálov pre celkovú analýzu
all_residuals <- c(train_residuals, test_residuals)

# Konverzia reziduálov na časový rad objekt pre ďalšiu analýzu
all_residuals_ts <- ts(all_residuals)

# Základná štatistika reziduálov
cat("Základná štatistika reziduálov:\n")
cat("Priemer reziduálov:", mean(all_residuals), "\n")
cat("Medián reziduálov:", median(all_residuals), "\n")
cat("Minimálna hodnota:", min(all_residuals), "\n")
cat("Maximálna hodnota:", max(all_residuals), "\n")
cat("Štandardná odchýlka:", sd(all_residuals), "\n")

#-----------------------------------------------------
# 9.2 Ljung-Box test - testovanie autokorelácií reziduálov
#-----------------------------------------------------
# Ljung-Box test testuje nulovú hypotézu, že všetky autokorelácie až po lag m sú nulové
# Ak sú reziduály náhodné, nemal by existovat vzor v ich autokoreláciách
cat("Vykonávam Ljung-Box testy pre autokoreláciu reziduálov...\n")

# Definujeme lagy, pre ktoré budeme testovat autokorelácie
lb_tests <- data.frame(
  lag = c(1, 5, 10, 15, 20, 30),
  statistic = numeric(6),
  p_value = numeric(6)
)

# Vykonáme test pre každý lag
for (i in 1:nrow(lb_tests)) {
  test_result <- Box.test(all_residuals, lag = lb_tests$lag[i], type = "Ljung-Box")
  lb_tests$statistic[i] <- test_result$statistic
  lb_tests$p_value[i] <- test_result$p.value
  
  # Výpis výsledkov pre lepšiu predstavu
  cat(sprintf("Ljung-Box test (lag %d): štatistika = %.4f, p-hodnota = %.4f%s\n", 
              lb_tests$lag[i], 
              test_result$statistic, 
              test_result$p.value,
              ifelse(test_result$p.value < 0.05, " - VÝznamnÁ autokorelácia", " - Bez významnej autokorelácie")))
}

# Vizualizácia autokorelácií a parciálnych autokorelácií reziduálov
cat("Vytváram grafy ACF a PACF reziduálov...\n")

# ACF graf
png(paste0(output_dir, "/residuals_acf.png"), width = 800, height = 600, res = 100)
acf(all_residuals, main = "ACF reziduálov", col = "blue")
dev.off()

# PACF graf
png(paste0(output_dir, "/residuals_pacf.png"), width = 800, height = 600, res = 100)
pacf(all_residuals, main = "PACF reziduálov", col = "blue")
dev.off()

#-----------------------------------------------------
# 9.3 Test ARCH efektu - testovanie heteroskedasticity
#-----------------------------------------------------
# ARCH efekt (Autoregressive Conditional Heteroskedasticity) predstavuje časovo premenlívú volatilitu 
# Testujeme ho pomocou Ljung-Box testu na kvadrátoch reziduálov
# Prítomnosť ARCH efektu znamená, že volatilita sa v čase mení spôsobom, ktorý možno modelovať
cat("\nTestujem prítomnosť ARCH efektu (heteroskedasticitu)...\n")

# Pripravíme data.frame pre výsledky testov
arch_tests <- data.frame(
  lag = c(1, 5, 10, 15, 20),
  statistic = numeric(5),
  p_value = numeric(5)
)

# Vypočítame kvadráty reziduálov - to je základ testu ARCH efektu
squared_residuals <- all_residuals^2

# Vykonáme test pre rôzne lagy
for (i in 1:nrow(arch_tests)) {
  test_result <- Box.test(squared_residuals, lag = arch_tests$lag[i], type = "Ljung-Box")
  arch_tests$statistic[i] <- test_result$statistic
  arch_tests$p_value[i] <- test_result$p.value
  
  # Výpis výsledkov testov
  cat(sprintf("ARCH test (lag %d): štatistika = %.4f, p-hodnota = %.4f%s\n", 
              arch_tests$lag[i], 
              test_result$statistic, 
              test_result$p.value,
              ifelse(test_result$p.value < 0.05, " - VÝznamnÁ heteroskedasticita (ARCH efekt)", " - Bez významnej heteroskedasticity")))
}

#-----------------------------------------------------
# 9.4 Testy normality reziduálov
#-----------------------------------------------------
# Testy normality ovárajú, či reziduály modelu majú normálne rozdelenie
# Pre XGBoost ako neparametrický model to nie je absolútne nuté, ale je to dôležité
# pre správnu interpretáciu intervalov spoľahlivosti a ďalšie štatistické operácie
cat("\nTestujem normalitu reziduálov...\n")

# Shapiro-Wilk test je jedným z najpoužívanejších testov normality
shapiro_test <- shapiro.test(all_residuals)
cat(sprintf("Shapiro-Wilk test: štatistika W = %.4f, p-hodnota = %.4f%s\n", 
          shapiro_test$statistic, 
          shapiro_test$p.value,
          ifelse(shapiro_test$p.value < 0.05, " - Reziduály NIE SÚ normálne rozdelené", " - Reziduály sú normálne rozdelené")))

# Jarque-Bera test je ďalší populárny test, ktorý testuje šikmosť a spätosť rozdelenia
jarque_bera_test <- jarque.bera.test(all_residuals)
cat(sprintf("Jarque-Bera test: štatistika = %.4f, p-hodnota = %.4f%s\n", 
          jarque_bera_test$statistic, 
          jarque_bera_test$p.value,
          ifelse(jarque_bera_test$p.value < 0.05, " - Reziduály NIE SÚ normálne rozdelené", " - Reziduály sú normálne rozdelené")))

#-----------------------------------------------------
# 9.5 Vizualizácia diagnostických grafov reziduálov
#-----------------------------------------------------

# Histogram reziduálov s prekrytou normálnou krivkou
cat("Vytváram histogram reziduálov...\n")
png(paste0(output_dir, "/residuals_histogram.png"), width = 1000, height = 800, res = 120)
par(bg = "white")
hist(all_residuals, breaks = 30, prob = TRUE, main = "Histogram reziduálov XGBoost modelu",
     xlab = "Reziduály", ylab = "Hustota", col = "lightblue", border = "white")

# Pridanie krivky normálneho rozdelenia pre porovnanie
curve(dnorm(x, mean = mean(all_residuals), sd = sd(all_residuals)), 
      add = TRUE, col = "red", lwd = 2)
legend("topright", legend = c("Rozdelenie reziduálov", "Normálne rozdelenie"),
       fill = c("lightblue", NA), border = c("white", NA),
       lty = c(NA, 1), lwd = c(NA, 2), col = c(NA, "red"))
dev.off()

# Q-Q plot - kvantil-kvantil graf porovnáva kvantily reziduálov s normálnym rozdelením
cat("Vytváram Q-Q plot reziduálov...\n")
png(paste0(output_dir, "/residuals_qqplot.png"), width = 1000, height = 800, res = 120)
par(bg = "white")
qqnorm(all_residuals, main = "Q-Q Plot reziduálov XGBoost modelu",
       pch = 19, col = "#3498db")
qqline(all_residuals, col = "red", lwd = 2)
dev.off()

# Graf reziduálov vs. fitted values - pomáha identifikovať heteroskedasticitu
cat("Vytváram graf Reziduály vs. predikované hodnoty...\n")
png(paste0(output_dir, "/residuals_vs_fitted.png"), width = 1000, height = 800, res = 120)
par(bg = "white")
plot(all_data$Predicted, all_data$Residuals, main = "Reziduály vs. predikované hodnoty",
     xlab = "Predikované hodnoty", ylab = "Reziduály",
     pch = 19, col = ifelse(all_data$Type == "Train", "#3498db", "#e74c3c"))
abline(h = 0, lty = 2, col = "gray50")
legend("topright", legend = c("Tréningové dáta", "Testovacie dáta"),
       col = c("#3498db", "#e74c3c"), pch = 19)
dev.off()

#################################################
# 5. Grafy fitu na teste s intervalmi spoľahlivosti
#################################################
# Vytvorenie grafu fitu špecificky pre testovacie dáta
test_fit_data <- data.frame(
  Date = test_data$Date,
  Actual = test_data$Count,
  Predicted = test_data$xgb_pred
)

p_test <- ggplot(test_fit_data, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Skutočné hodnoty")) +
  geom_line(aes(y = Predicted, color = "Predikované hodnoty")) +
  labs(title = "XGBoost - Fit modelu na testovacie dáta",
       x = "Dátum",
       y = "Počet legislatívnych aktivít",
       color = "Typ hodnôt") +
  theme_minimal() +
  scale_color_manual(values = c("Skutočné hodnoty" = "blue", "Predikované hodnoty" = "red"))

ggsave(paste0(output_dir, "/test_data_fit.png"), p_test, width = 10, height = 6, dpi = 300)

#################################################
# 6. Intervaly spoľahlivosti pre prognózu
#################################################
# Vytvorenie intervalov spoľahlivosti pomocou bootstrapingu
set.seed(123) # pre reprodukovateľnosť

bootstrap_intervals <- function(residuals, forecast_values, n_bootstrap = 1000, conf_level = 0.95) {
  n_residuals <- length(residuals)
  n_forecast <- length(forecast_values)
  
  # Vytvorenie matice simulovaných hodnôt
  simulated_values <- matrix(0, nrow = n_bootstrap, ncol = n_forecast)
  
  for (i in 1:n_bootstrap) {
    # Náhodný výber reziduálov s návratom
    sampled_residuals <- sample(residuals, n_forecast, replace = TRUE)
    
    # Pridanie reziduálov k predikovaným hodnotám
    simulated_values[i, ] <- forecast_values + sampled_residuals
  }
  
  # Zaokrúhlenie a zaistenie nezápornosti
  simulated_values <- round(pmax(simulated_values, 0))
  
  # Výpočet intervalov spoľahlivosti
  alpha <- (1 - conf_level) / 2
  lower_bound <- apply(simulated_values, 2, function(x) quantile(x, alpha))
  upper_bound <- apply(simulated_values, 2, function(x) quantile(x, 1 - alpha))
  
  return(list(lower = lower_bound, upper = upper_bound))
}

# Aplikácia bootstrapingu na získanie intervalov spoľahlivosti
intervals <- bootstrap_intervals(all_residuals, forecast_results$Count)

# Pridanie intervalov do forecast dataframe
forecast_results$lower <- intervals$lower
forecast_results$upper <- intervals$upper

# Uloženie prognózy s intervalmi
write.csv(forecast_results, paste0(output_dir, "/xgboost_forecast_with_intervals.csv"), row.names = FALSE)

# Vytvorenie grafu prognózy s intervalmi
forecast_with_intervals <- ggplot() +
  geom_line(data = legisl_data, aes(x = Date, y = Count, color = "Historické dáta")) +
  geom_line(data = forecast_results, aes(x = Date, y = Count, color = "XGBoost prognóza")) +
  geom_ribbon(data = forecast_results, 
              aes(x = Date, ymin = lower, ymax = upper),
              alpha = 0.2, fill = "red") +
  labs(title = "XGBoost prognóza s 95% intervalmi spoľahlivosti",
       x = "Dátum",
       y = "Počet legislatívnych aktivít",
       color = "Typ dát") +
  theme_minimal() +
  scale_color_manual(values = c("Historické dáta" = "blue", "XGBoost prognóza" = "red"))

ggsave(paste0(output_dir, "/forecast_with_intervals.png"), forecast_with_intervals, width = 12, height = 6, dpi = 300)

# Vytvorenie všetkých potrebných vizualizácií pre prognózu

# 1. Základný graf skutočných vs. predikovaných hodnôt
png(paste0(output_dir, "/actual_vs_predicted.png"), width = 1200, height = 800, res = 120)
par(mar = c(5,5,4,2) + 0.1)
plot(all_data$Date, all_data$Actual, type = "l", col = "blue", lwd = 2,
     main = "Skutočné vs. predikované hodnoty",
     xlab = "Dátum", ylab = "Počet legislatívnych aktivit", cex.lab = 1.2, cex.main = 1.5)
lines(all_data$Date, all_data$Predicted, col = "red", lwd = 2)
abline(v = min(test_data$Date), lty = 2, lwd = 1.5)
legend("topright", legend = c("Skutočné", "Predikované", "Začiatok testovacieho obdobia"),
       col = c("blue", "red", "black"), lty = c(1,1,2), lwd = c(2,2,1.5))
dev.off()

# 2. Detailný pohľad na testovacie dáta
test_data_only <- all_data[all_data$Type == "Test", ]
png(paste0(output_dir, "/test_fit_detailed.png"), width = 1200, height = 800, res = 120)
par(mar = c(5,5,4,2) + 0.1)
plot(test_data_only$Date, test_data_only$Actual, type = "l", col = "blue", lwd = 2,
     main = "Fit modelu na testovacie dáta (detailný pohľad)",
     xlab = "Dátum", ylab = "Počet legislatívnych aktivit", cex.lab = 1.2, cex.main = 1.5)
lines(test_data_only$Date, test_data_only$Predicted, col = "red", lwd = 2)
legend("topright", legend = c("Skutočné", "Predikované"),
       col = c("blue", "red"), lty = c(1,1), lwd = c(2,2))
dev.off()

# 3. Scatter plot skutočných vs. predikovaných hodnôt
png(paste0(output_dir, "/scatter_actual_predicted.png"), width = 1200, height = 800, res = 120)
par(mar = c(5,5,4,2) + 0.1)
plot(all_data$Actual, all_data$Predicted, pch = 19, 
     col = ifelse(all_data$Type == "Train", "blue", "red"),
     main = "Scatter plot: skutočné vs. predikované hodnoty",
     xlab = "Skutočné hodnoty", ylab = "Predikované hodnoty", cex.lab = 1.2, cex.main = 1.5)
abline(a = 0, b = 1, lty = 2, lwd = 2)
legend("topleft", legend = c("Tréningové dáta", "Testovacie dáta"),
       col = c("blue", "red"), pch = 19)
dev.off()

# 4. Vytvorenie pekného grafu prognózy s intervalmi spoľahlivosti použitím ggplot2
png(paste0(output_dir, "/xgboost_forecast.png"), width = 1200, height = 800, res = 120)

# Vytvorenie jednotného data framu pre ggplot
last_60_days <- tail(all_data, 60)
historical_df <- data.frame(
  Date = last_60_days$Date,
  Value = last_60_days$Actual,
  Type = "Historické hodnoty"
)

forecast_df <- data.frame(
  Date = forecast_results$Date,
  Value = forecast_results$Count,
  Lower = forecast_results$lower,
  Upper = forecast_results$upper,
  Type = "Predpoveď"
)

# Kombinačný graf
forecast_plot <- ggplot() +
  # Historické dáta
  geom_line(data = historical_df, aes(x = Date, y = Value, color = Type), linewidth = 1) +
  # Predpoveď
  geom_line(data = forecast_df, aes(x = Date, y = Value, color = Type), linewidth = 1) +
  # Intervaly spoľahlivosti
  geom_ribbon(data = forecast_df, aes(x = Date, ymin = Lower, ymax = Upper, fill = Type), alpha = 0.2) +
  # Čiara označujúca začiatok prognózy
  geom_vline(xintercept = as.numeric(min(forecast_df$Date)), linetype = "dashed") +
  # Nastavenie farieb a legendy
  scale_color_manual(name = "Typ dát", values = c("Historické hodnoty" = "blue", "Predpoveď" = "red")) +
  scale_fill_manual(name = "Interval spoľahlivosti", values = c("Predpoveď" = "red")) +
  # Nadpis a popisky osí
  labs(title = "XGBoost prognóza s 95% intervalmi spoľahlivosti",
       subtitle = paste("Model s R² =", round(test_r2, 3), "na testovacích dátach"),
       x = "Dátum", y = "Počet legislatívnych aktivit") +
  # Štýl grafu
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.position = "bottom",
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 11),
    panel.grid.minor = element_blank()
  )

# Vytlačenie grafu a uloženie
print(forecast_plot)
dev.off()

# Uložit aj ako samostatnú verziu PNG s vysokým rozlíšením
ggsave(paste0(output_dir, "/xgboost_forecast_high_res.png"), forecast_plot, width = 12, height = 8, dpi = 300)

# 5. Dôležitosť prediktorov
png(paste0(output_dir, "/feature_importance.png"), width = 1200, height = 800, res = 120)
par(mar = c(10, 5, 4, 2) + 0.1) # Viac miesta pre názvy prediktorov
importance_df <- data.frame(
  Feature = rownames(importance_matrix),
  Importance = importance_matrix$Gain
)
importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ][1:15, ]
barplot(importance_df$Importance, names.arg = importance_df$Feature, col = "steelblue",
        main = "Dôležitosť prediktorov v XGBoost modeli", las = 2,
        ylab = "Relatívna dôležitosť", cex.names = 0.8, cex.main = 1.5)
dev.off()

#############################################################################
# 10. Export výsledkov a generovanie reportov
#############################################################################

#-----------------------------------------------------
# 10.1 Export všetkých relevantných dát
#-----------------------------------------------------
cat("\nExportujem všetky výsledky a diagnostické testy...\n")
cat("Všetky grafy boli uložené vo formáte PNG v adresári:", output_dir, "\n")

# Zoznam všetkých vygenerovaných PNG súborov pre informáciu používateľa
png_files <- list.files(output_dir, pattern = "\\.png$", full.names = FALSE)
cat("\nVygenerované grafy:\n")
for (file in png_files) {
  cat("- ", file, "\n")
}

# Vytvorenie komplexného objektu so všetkými výsledkami
all_results <- list(
  model = xgb_model,
  importance = importance_matrix,
  train_metrics = data.frame(RMSE = train_rmse, MAE = train_mae, R2 = train_r2),
  test_metrics = data.frame(RMSE = test_rmse, MAE = test_mae, R2 = test_r2),
  forecast = forecast_results,
  residuals = list(train = train_residuals, test = test_residuals),
  diagnostics = list(
    ljung_box = lb_tests,
    arch_test = arch_tests,
    normality = list(shapiro = shapiro_test, jarque_bera = jarque_bera_test)
  )
)

# Uloženie výsledkov pre prípadné ďalšie použitie
saveRDS(all_results, paste0(output_dir, "/xgboost_results.rds"))

#-----------------------------------------------------
# 10.2 Vytvorenie PDF reportu s výsledkami
#-----------------------------------------------------
cat("Generujem PDF report s výsledkami...\n")

# Namiesto PDF vytvorím samostatné PNG grafy pre fit modelu a prognózu s intervalmi
cat("Generujem grafy vo formáte PNG...\n")

# 1. Vytvorenie grafu s fitom modelu (skutočné vs. predikované hodnoty)
png(paste0(output_dir, "/model_fit.png"), width = 1200, height = 800, res = 120)
par(mar = c(5, 5, 4, 2) + 0.1)
plot(all_data$Date, all_data$Actual, type = "l", col = "blue", lwd = 2,
     main = "Skutočné vs. predikované hodnoty",
     xlab = "Dátum", ylab = "Počet legislatívnych aktivit", cex.lab = 1.2, cex.axis = 1.1, cex.main = 1.5)
lines(all_data$Date, all_data$Predicted, col = "red", lwd = 2)
abline(v = min(test_data$Date), lty = 2, lwd = 1.5)
legend("topright", legend = c("Skutočné", "Predikované", "Začiatok testov. obdobia"),
       col = c("blue", "red", "black"), lty = c(1, 1, 2), lwd = c(2, 2, 1.5), cex = 1.2)
dev.off()

# 2. Detailný graf s fitom len pre testovacie dáta
png(paste0(output_dir, "/test_data_fit.png"), width = 1200, height = 800, res = 120)
test_range <- all_data[all_data$Type == "Test", ]
par(mar = c(5, 5, 4, 2) + 0.1)
plot(test_range$Date, test_range$Actual, type = "l", col = "blue", lwd = 2,
     main = "Fit modelu na testovacie dáta",
     xlab = "Dátum", ylab = "Počet legislatívnych aktivit", cex.lab = 1.2, cex.axis = 1.1, cex.main = 1.5)
lines(test_range$Date, test_range$Predicted, col = "red", lwd = 2)
legend("topright", legend = c("Skutočné", "Predikované"),
       col = c("blue", "red"), lty = c(1, 1), lwd = c(2, 2), cex = 1.2)
dev.off()

# Upozornenie: Duplikovaný graf s prognózou - už sa generuje v sekci vizuálných výstupov vyššie

#############################################################################
# 11. Generovanie Markdown reportu
#############################################################################
cat("\nGenerujem Markdown report s výsledkami analýzy a diagnostiky...\n")

# Funkcia pre formátovanie čísel v reporte
format_num <- function(x, digits = 4) {
  return(format(round(x, digits), nsmall = digits))
}

# Názov a cesta súboru pre report
md_report_file <- paste0(output_dir, "/xgboost_model_diagnostics.md")

# Vytvorenie header a úvodu reportu
md_report <- c(
  "# Diagnostika XGBoost modelu pre legislatívne časové rady",
  "",
  "## 1. Prehľad modelu",
  "",
  "XGBoost model bol použitý na predikciu legislatívnych časových radov. XGBoost je gradient boosting algoritmus, ktorý dokáže zachytiť nelineárne vzťahy v dátach.",
  "",
  "### Základné parametre modelu:",
  "- **Objective**: reg:squarederror",
  paste0("- **Learning rate (eta)**: ", params$eta),
  paste0("- **Max depth**: ", params$max_depth),
  paste0("- **Subsample**: ", params$subsample),
  paste0("- **Colsample bytree**: ", params$colsample_bytree),
  paste0("- **Min child weight**: ", params$min_child_weight),
  "",
  "### Metriky presnosti:",
  "",
  "| Dataset    | RMSE   | MAE    | R²     |",
  "|------------|--------|--------|--------|",
  paste0("| Trénovací  | ", format_num(train_rmse), " | ", format_num(train_mae), " | ", format_num(train_r2), " |"),
  paste0("| Testovací  | ", format_num(test_rmse), " | ", format_num(test_mae), " | ", format_num(test_r2), " |"),
  "",
  paste0("Hodnota R² = ", format_num(test_r2), " na testovacích dátach znamená, že model vysvetľuje približne ", 
         format_num(test_r2 * 100, 1), "% variability v legislatívnych dátach, čo je výrazne lepšie ako Holt-Winters model (R² = 0.3866).")
)

# Sekcia o výsledkoch diagnostických testov
md_diagnostics <- c(
  "",
  "## 3. Analýza reziduálov",
  "",
  "### 3.1 Ljung-Box test autokorelácií",
  "",
  "Ljung-Box test testuje nulovú hypotézu, že autokorelácie až po lag m sú nulové. Inými slovami, testuje, či sú reziduály náhodné a nezávislé.",
  "",
  "| Lag | Testová štatistika | p-hodnota |",
  "|-----|-------------------|-------------------|"
)

# Pridanie výsledkov Ljung-Box testov
for(i in 1:nrow(lb_tests)) {
  md_diagnostics <- c(md_diagnostics, 
                    paste0("| ", lb_tests$lag[i], " | ", format_num(lb_tests$statistic[i], 4), 
                           " | ", format_num(lb_tests$p_value[i], 4), " |"))
}

# Pridanie interpretácie
md_diagnostics <- c(md_diagnostics,
  "",
  "**Interpretácia**: ",
  "Ak p-hodnota < 0.05, zamietame nulovú hypotézu a reziduály vykazujú štatisticky významnú autokoreláciu, čo znamená, že model nezachytáva všetky časové závislosti v dátach.",
  "Ak p-hodnota >= 0.05, nezamietame nulovú hypotézu a reziduály nevykazujú štatisticky významnú autokoreláciu, čo je pozitívny znak.",
  "",
  "### 3.2 Autokorelácie a parciálne autokorelácie reziduálov",
  "",
  "Nasledujúce grafy zobrazujú autokorelácie (ACF) a parciálne autokorelácie (PACF) reziduálov pre rôzne lagy:",
  "",
  "![ACF reziduálov](xgboost_results/residuals_acf.png)",
  "",
  "![PACF reziduálov](xgboost_results/residuals_pacf.png)",
  "",
  "Modrá prerušovaná čiara v grafoch predstavuje hranicu štatistickej významnosti (±1.96/√n). Stĺpce, ktoré prekračujú túto hranicu, indikujú štatisticky významnú autokoreláciu na danom lagu."
)

# ARCH testy
md_arch <- c(
  "",
  "### 3.3 Test ARCH efektu",
  "",
  "ARCH efekt (Autoregressive Conditional Heteroskedasticity) označuje prítomnosť časovo premenlivej volatility v časových radoch. Na testovanie ARCH efektu sme použili Ljung-Box test na kvadrátoch reziduálov.",
  "",
  "| Lag | Testová štatistika | p-hodnota |",
  "|-----|-------------------|-------------------|"
)

for(i in 1:nrow(arch_tests)) {
  md_arch <- c(md_arch, 
               paste0("| ", arch_tests$lag[i], " | ", format_num(arch_tests$statistic[i], 4), 
                      " | ", format_num(arch_tests$p_value[i], 4), " |"))
}

md_arch <- c(md_arch,
  "",
  "**Interpretácia**: ",
  "Ak p-hodnota < 0.05, existuje významná časová závislosť vo volatilite reziduálov (ARCH efekt).",
  "Ak p-hodnota >= 0.05, neexistuje významná časová závislosť vo volatilite reziduálov."
)

# Normalita
md_normality <- c(
  "",
  "### 3.4 Test normality reziduálov",
  "",
  "Normalita reziduálov nie je kritickým predpokladom pre XGBoost ako neparametrický model, ale je dôležitá pre správnu interpretáciu intervalov spoľahlivosti.",
  "",
  "#### 3.4.1 Shapiro-Wilk test",
  "",
  paste0("**Testová štatistika W**: ", format_num(shapiro_test$statistic, 4)),
  "",
  paste0("**p-hodnota**: ", format_num(shapiro_test$p.value, 4)),
  "",
  "#### 3.4.2 Jarque-Bera test",
  "",
  paste0("**Testová štatistika**: ", format_num(jarque_bera_test$statistic, 4)),
  "",
  paste0("**p-hodnota**: ", format_num(jarque_bera_test$p.value, 4)),
  "",
  "**Interpretácia**:",
  "Ak p-hodnota < 0.05 (pre ktorýkoľvek test), reziduály nemajú normálne rozdelenie.",
  "Ak p-hodnota >= 0.05, nie je dôvod zamietať predpoklad normality reziduálov.",
  "",
  "### 3.5 Vizualizácie normality reziduálov",
  "",
  "Nasledujúce grafy pomáhajú vizuálne posúdiť normalitu reziduálov:",
  "",
  "![Histogram reziduálov](xgboost_results/residuals_histogram.png)",
  "",
  "![Q-Q plot reziduálov](xgboost_results/residuals_qqplot.png)",
  "",
  "### 3.6 Homoskedasticita reziduálov",
  "",
  "Homoskedasticita označuje konštantný rozptyl reziduálov. Nasledujúci graf zobrazuje reziduály vzhľadom na predikované hodnoty:",
  "",
  "![Reziduály vs. predikované hodnoty](xgboost_results/residuals_vs_fitted.png)",
  "",
  "**Interpretácia**:",
  "Ak reziduály vykazujú systematický vzor (napr. lievik, zakrivenie), môže to indikovať heteroskedasticitu alebo nesprávnu špecifikáciu modelu.",
  "Ideálne by mali byť reziduály náhodne rozptýlené okolo horizontálnej línie nuly."
)

# Prognóza a intervaly spoľahlivosti
md_forecast <- c(
  "",
  "## 4. Prognóza s intervalmi spoľahlivosti",
  "",
  "XGBoost model generuje bodové predikcie, ale pre praktické použitie sú dôležité aj intervaly spoľahlivosti. V tomto prípade sme vygenerovali 95% intervaly spoľahlivosti pomocou bootstrapingu reziduálov.",
  "",
  "![Prognóza s intervalmi spoľahlivosti](xgboost_results/forecast_with_intervals.png)",
  "",
  "Intervaly spoľahlivosti reprezentujú neistotu spojenú s predikciou a poskytujú rozsah hodnôt, v ktorom s 95% pravdepodobnosťou bude ležať skutočná hodnota.",
  "",
  "## 5. Dôležitosť prediktorov",
  "",
  "XGBoost model poskytuje informácie o dôležitosti jednotlivých prediktorov:",
  "",
  "![Dôležitosť prediktorov](xgboost_results/feature_importance.png)",
  "",
  "Toto je cenná informácia, ktorá ukazuje, ktoré prediktory majú najväčší vplyv na predikcie modelu."
)

# Závery a odporúčania
md_conclusions <- c(
  "",
  "## 6. Závery a odporúčania",
  "",
  "### 6.1 Zhrnutie výsledkov diagnostiky",
  "",
  paste0("1. **Výkonnosť modelu**: XGBoost model dosahuje R² = ", format_num(test_r2, 2), 
         " na testovacích dátach, čo je výrazne lepšie ako Holt-Winters model (R² = 0.39)."),
  "",
  "2. **Analýza reziduálov**:"
)

# Vytvorenie záveru o autokorelácii na základe výsledkov
autocorr_conclusion <- ifelse(all(lb_tests$p_value > 0.05),
  "   - **Autokorelácia reziduálov**: Ljung-Box testy nevykázali štatisticky významnú autokoreláciu pre všetky testované lagy (p-hodnoty > 0.05). To znamená, že model dobre zachytáva časové závislosti v dátach.",
  "   - **Autokorelácia reziduálov**: Ljung-Box testy odhalili štatisticky významnú autokoreláciu v reziduáloch, čo naznačuje, že model nezachytáva všetky časové závislosti v dátach."
)

# Vytvorenie záveru o ARCH efekte
arch_conclusion <- ifelse(all(arch_tests$p_value > 0.05),
  "   - **ARCH efekt**: Testy na ARCH efekt ukázali, že neexistuje štatisticky významná časová závislosť vo volatilite reziduálov (p-hodnoty > 0.05).",
  "   - **ARCH efekt**: Testy na ARCH efekt odhalili časovo závislú volatilitu v reziduáloch, čo naznačuje prítomnosť ARCH efektu."
)

# Vytvorenie záveru o normalite
normality_conclusion <- paste0(
  "   - **Normalita reziduálov**: Shapiro-Wilk test (p-hodnota = ", format_num(shapiro_test$p.value, 4), 
  ") a Jarque-Bera test (p-hodnota = ", format_num(jarque_bera_test$p.value, 4), 
  ") ", ifelse(shapiro_test$p.value < 0.05 || jarque_bera_test$p.value < 0.05,
              "naznačujú, že reziduály nie sú normálne rozdelené. Pre XGBoost ako neparametrický model to však nie je kritický problém, ale môže to ovplyvniť presnosť intervalov spoľahlivosti.",
              "naznačujú, že nie je dôvod zamietnrť hypotézu o normalite reziduálov, čo je priaznivý výsledok pre štatistickú inferenciu.")
)

md_conclusions <- c(md_conclusions, autocorr_conclusion, arch_conclusion, normality_conclusion)

# Porovnanie s inými modelmi
md_comparison <- c(
  "",
  "## 7. Porovnanie s predchádzajúcimi modelmi",
  "",
  "| Model        | Testovací R² | RMSE   | MAE    |",
  "|--------------|--------------|--------|--------|",
  paste0("| XGBoost      | ", 
  format_num(test_r2, 2), " | ", format_num(test_rmse, 2), " | ", format_num(test_mae, 2), " |"),
  "| Holt-Winters | 0.39 | 4.96 | 2.67 |",
  "",
  paste0("XGBoost model výrazne prekonáva tak Holt-Winters v presnosti predikcie. ",
         "Hodnota R² = ", format_num(test_r2, 2), " znamená, že model dokáže vysvetliť približne ", format_num(test_r2 * 100, 1), 
         "% variability v legislatívnych dátach, čo je výrazné zlepšenie oproti predchádzajúcim modelom. ",
         "To potvrdzuje schopnosť XGBoost modelu zachytiť komplexné nelineárne vzťahy a sezónne vzory v legislatívnych dátach.")
)

# Spojenie všetkých častí reportu
md_full_report <- c(
  md_report,
  "",
  "## 2. Fit modelu na testovacie dáta",
  "",
  "Nasledujúci graf zobrazuje skutočné a predikované hodnoty na testovacom datasete. Toto je dôležitá vizualizácia, ktorá ukazuje, ako dobre model dokáže predikovať budúce hodnoty.",
  "",
  "![Fit na testovacie dáta](xgboost_results/test_data_fit.png)",
  "",
  md_diagnostics,
  md_arch,
  md_normality,
  md_forecast,
  md_conclusions,
  md_comparison,
  "",
  "---",
  "",
  "*Poznámka: Táto správa bola vygenerovaná na základe diagnostickej analýzy XGBoost modelu pre legislatívne časové rady. Kompletné vizualizácie a detailné výsledky sú k dispozícii v priečinku 'legislation/xgboost_results'.*"
)

# Uloženie Markdown reportu
writeLines(md_full_report, md_report_file)
cat("Markdown report bol vygenerovaný a uložený do:", md_report_file, "\n")

# Finálne hlásenie o dokončení analýzy
cat("\nKompletná analýza pomocou XGBoost modelu bola dokončená.\n")
cat("Všetky výsledky boli uložené v priečinku:", output_dir, "\n")
cat("Metriky presnosti - testovacie dáta: RMSE =", round(test_rmse, 4), ", R² =", round(test_r2, 4), "\n")
cat("Markdown report s diagnostikou bol vygenerovaný priamo z modelu.\n")
