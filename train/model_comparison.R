## Model Comparison Script
## Visualizing SARIMA and GARCH model fits and forecasts on the same plot

# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Install required packages if not already installed
if (!require("forecast")) install.packages("forecast")
if (!require("tseries")) install.packages("tseries")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("rugarch")) install.packages("rugarch")
if (!require("zoo")) install.packages("zoo")

# Load required libraries
library(forecast)
library(tseries)
library(ggplot2)
library(rugarch)
library(zoo)

# Load data
data_file <- "sorted_train_entries.csv"
if (!file.exists(data_file)) {
  data_file <- "sarima/sorted_train_entries.csv"
  if (!file.exists(data_file)) {
    data_file <- "garch/sorted_train_entries.csv"
    if (!file.exists(data_file)) {
      stop("Data file not found. Please check the file path.")
    }
  }
}

cat("Loading data...\n")
data <- read.csv(data_file)
data$date <- as.Date(data$date)

# Create time series object
date_range <- seq(min(data$date), max(data$date), by="day")
full_data <- data.frame(date = date_range)
full_data <- merge(full_data, data, by="date", all.x=TRUE)

# Handle missing values
if(any(is.na(full_data$time))) {
  cat("Missing values detected in the time series. Imputing missing values...\n")
  full_data$time <- na.approx(full_data$time, na.rm=FALSE)
  
  if(any(is.na(full_data$time))) {
    full_data$time <- na.locf(full_data$time, fromLast=TRUE)
    full_data$time <- na.locf(full_data$time)
  }
}

# Create time series object with weekly seasonality
ts_data <- ts(full_data$time, frequency=7)

#----------------------------------------
# 1. SARIMA Model Fitting
#----------------------------------------

cat("\n========================================\n")
cat("Fitting SARIMA model...\n")
cat("========================================\n")

# Create a function to fit the SARIMA model as in sarima_full_forecast.R
fit_sarima <- function(ts_data) {
  # Test for seasonal periods
  seasonal_periods <- c(7, 21, 35, 63)
  sarima_models <- list()
  
  for (period in seasonal_periods) {
    cat("Testing seasonal period:", period, "\n")
    temp_ts <- ts(full_data$time, frequency=period)
    tryCatch({
      model <- auto.arima(temp_ts, seasonal=TRUE, stepwise=TRUE, approximation=TRUE)
      sarima_models[[as.character(period)]] <- list(model=model, aic=model$aic, bic=model$bic)
      cat("AIC:", model$aic, "BIC:", model$bic, "\n")
    }, error=function(e) {
      cat("Error with period", period, ":", e$message, "\n")
    })
  }
  
  # Also fit a non-seasonal ARIMA model
  arima_model <- auto.arima(ts_data, seasonal=FALSE, stepwise=FALSE, approximation=FALSE)
  
  # Find the best seasonal period based on AIC
  best_period <- 7  # Default to 7 if no better model is found
  best_aic <- Inf
  
  if (length(sarima_models) > 0) {
    for (period in names(sarima_models)) {
      if (sarima_models[[period]]$aic < best_aic) {
        best_aic <- sarima_models[[period]]$aic
        best_period <- as.numeric(period)
      }
    }
    best_sarima_model <- sarima_models[[as.character(best_period)]]$model
    cat("\nBest seasonal period based on AIC:", best_period, "\n")
  }
  
  # Select the best model between seasonal and non-seasonal
  if (exists("best_sarima_model") && best_sarima_model$aicc < arima_model$aicc) {
    best_model <- best_sarima_model
    model_type <- "SARIMA"
    cat("\nSARIMA model with period", best_period, "selected as best model\n")
  } else {
    best_model <- arima_model
    model_type <- "ARIMA"
    cat("\nARIMA model selected as best model\n")
  }
  
  cat("\nBest model:", model_type, "\n")
  print(summary(best_model))
  
  return(best_model)
}

# Fit SARIMA model
sarima_model <- fit_sarima(ts_data)

# Generate SARIMA fitted values and forecasts
sarima_fitted <- fitted(sarima_model)
sarima_forecast <- forecast(sarima_model, h=30)

#----------------------------------------
# 2. GARCH Model Fitting
#----------------------------------------

cat("\n========================================\n")
cat("Fitting GARCH model...\n")
cat("========================================\n")

# Create a function to fit the GARCH model similar to garch_analysis.R
fit_garch <- function(ts_data) {
  # Check for ARCH effects
  squared_returns <- ts_data^2
  arch_test <- Box.test(squared_returns, lag=10, type="Ljung-Box")
  cat("\n--- ARCH Test for ARCH Effects ---\n")
  print(arch_test)
  
  if (arch_test$p.value <= 0.05) {
    cat("p-value:", arch_test$p.value, "\n")
    cat("There are significant ARCH effects in the data. GARCH modeling is appropriate.\n")
  } else {
    cat("p-value:", arch_test$p.value, "\n")
    cat("No significant ARCH effects in the data. GARCH modeling may not be necessary.\n")
    cat("Proceeding with GARCH modeling for comparison purposes...\n")
  }
  
  cat("\n--- Fitting Various GARCH Models ---\n")
  
  # Specify model forms
  garch11_spec <- ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    mean.model = list(armaOrder = c(1, 1), include.mean = TRUE),
    distribution.model = "std"
  )
  
  # Fit the models on the full dataset
  cat("Fitting GARCH(1,1) with t distribution...\n")
  garch11_fit <- try(ugarchfit(spec = garch11_spec, data = ts_data))
  
  # Check if the model fit succeeded
  if(class(garch11_fit)[1] == "try-error") {
    cat("Error fitting GARCH model. Using a simpler specification...\n")
    # If there was an error, try a simpler model
    garch11_spec <- ugarchspec(
      variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
      mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
      distribution.model = "norm"
    )
    garch11_fit <- ugarchfit(spec = garch11_spec, data = ts_data)
  }
  
  cat("\n--- Best GARCH Model Summary ---\n")
  print(garch11_fit)
  
  return(garch11_fit)
}

# Fit GARCH model
garch_model <- fit_garch(ts_data)

# Extract GARCH fitted values and forecasts
garch_fitted <- fitted(garch_model)
garch_forecast <- ugarchforecast(garch_model, n.ahead=30)
garch_forecast_mean <- garch_forecast@forecast$seriesFor

#----------------------------------------
# 3. Forecast Dates
#----------------------------------------

# Generate dates for the 30-day forecast period
last_date <- max(data$date)
forecast_dates <- seq(last_date + 1, by="day", length.out=30)

#----------------------------------------
# 4. Combined Visualization
#----------------------------------------

cat("\n========================================\n")
cat("Creating combined model visualization...\n")
cat("========================================\n")

# Create data frames for plotting
# Original data
full_ts_df <- data.frame(
  date = full_data$date,
  value = as.numeric(ts_data)
)

# SARIMA fitted values and forecast
sarima_df <- data.frame(
  date = c(full_data$date, forecast_dates),
  fitted = c(as.numeric(sarima_fitted), rep(NA, 30)),
  forecast = c(rep(NA, length(ts_data)), as.numeric(sarima_forecast$mean)),
  lower_95 = c(rep(NA, length(ts_data)), as.numeric(sarima_forecast$lower[,2])),
  upper_95 = c(rep(NA, length(ts_data)), as.numeric(sarima_forecast$upper[,2]))
)

# GARCH fitted values and forecast
garch_df <- data.frame(
  date = c(full_data$date, forecast_dates),
  fitted = c(as.numeric(garch_fitted), rep(NA, 30)),
  forecast = c(rep(NA, length(ts_data)), as.numeric(garch_forecast_mean)),
  # Note: GARCH forecasts don't have built-in prediction intervals like SARIMA
  # We would need to create custom intervals based on the forecast variance
  lower_95 = rep(NA, length(ts_data) + 30),
  upper_95 = rep(NA, length(ts_data) + 30)
)

# Create a PDF for the combined visualization
pdf("model_comparison_fit.pdf", width=14, height=10)

# Full dataset visualization
# Create a layout for the plots
layout(matrix(c(1,2,3), nrow=3), heights=c(3,2,2))
par(mar=c(4,4,3,2))

# Plot 1: Full time series with both model fits and forecasts
plot(full_ts_df$date, full_ts_df$value, type="l", col="black", lwd=1.5,
     main="SARIMA vs. GARCH: Full Historical Data and 30-Day Forecasts",
     xlab="Date", ylab="Value", xaxt="n")

# Add the model fits
lines(sarima_df$date, sarima_df$fitted, col="blue", lwd=1)
lines(garch_df$date, garch_df$fitted, col="red", lwd=1)

# Add forecasts
lines(sarima_df$date, sarima_df$forecast, col="blue", lwd=2)
lines(garch_df$date, garch_df$forecast, col="red", lwd=2)

# Add prediction intervals for SARIMA
polygon(c(sarima_df$date, rev(sarima_df$date)),
        c(sarima_df$lower_95, rev(sarima_df$upper_95)),
        col=rgb(0,0,1,0.1), border=NA)

# Add a vertical line separating historical data from forecast
abline(v=as.numeric(last_date), lty=2, col="gray50")

# Add date axis
axis.Date(1, at=seq(min(full_ts_df$date), max(forecast_dates), by="6 months"), format="%b %Y")

# Add legend
legend("topleft", 
       legend=c("Actual Data", "SARIMA Fit", "GARCH Fit", "SARIMA Forecast", "GARCH Forecast", "SARIMA 95% PI"),
       col=c("black", "blue", "red", "blue", "red", rgb(0,0,1,0.3)),
       lwd=c(1.5, 1, 1, 2, 2, 10),
       bg="white")

# Plot 2: Zoomed view of recent data and forecast
# Determine how many days to show
recent_days <- 90
recent_start_date <- max(min(full_ts_df$date), max(full_ts_df$date) - recent_days)
recent_ts_df <- full_ts_df[full_ts_df$date >= recent_start_date,]

# Plot the recent data and forecasts
plot_range <- range(c(recent_ts_df$value, sarima_df$upper_95[sarima_df$date >= recent_start_date], 
                      sarima_df$lower_95[sarima_df$date >= recent_start_date], 
                      garch_df$forecast[garch_df$date >= recent_start_date]), na.rm=TRUE)
plot_range <- plot_range + c(-0.1, 0.1) * diff(plot_range)  # Add padding

plot(recent_ts_df$date, recent_ts_df$value, type="l", col="black", lwd=1.5,
     main="SARIMA vs. GARCH: Recent Data with 30-Day Forecasts",
     xlab="Date", ylab="Value", ylim=plot_range, xaxt="n")

# Add the model fits for the recent period
lines(sarima_df$date[sarima_df$date >= recent_start_date], 
      sarima_df$fitted[sarima_df$date >= recent_start_date], col="blue", lwd=1)
lines(garch_df$date[garch_df$date >= recent_start_date], 
      garch_df$fitted[garch_df$date >= recent_start_date], col="red", lwd=1)

# Add forecasts
lines(sarima_df$date[sarima_df$date > last_date], 
      sarima_df$forecast[sarima_df$date > last_date], col="blue", lwd=2)
lines(garch_df$date[garch_df$date > last_date], 
      garch_df$forecast[garch_df$date > last_date], col="red", lwd=2)

# Add prediction intervals for SARIMA
polygon(c(sarima_df$date[sarima_df$date > last_date], 
          rev(sarima_df$date[sarima_df$date > last_date])),
        c(sarima_df$lower_95[sarima_df$date > last_date], 
          rev(sarima_df$upper_95[sarima_df$date > last_date])),
        col=rgb(0,0,1,0.1), border=NA)

# Add a vertical line separating historical data from forecast
abline(v=as.numeric(last_date), lty=2, col="gray50")

# Add date axis with more frequent labels
axis.Date(1, at=seq(recent_start_date, max(forecast_dates), by="15 days"), format="%d %b")

# Add legend
legend("topleft", 
       legend=c("Actual Data", "SARIMA Fit", "GARCH Fit", "SARIMA Forecast", "GARCH Forecast", "SARIMA 95% PI"),
       col=c("black", "blue", "red", "blue", "red", rgb(0,0,1,0.3)),
       lwd=c(1.5, 1, 1, 2, 2, 10),
       bg="white")

# Plot 3: Residuals comparison
sarima_residuals <- ts_data - sarima_fitted
garch_residuals <- ts_data - garch_fitted

# Ensure equal length
min_length <- min(length(sarima_residuals), length(garch_residuals))
sarima_residuals <- sarima_residuals[1:min_length]
garch_residuals <- garch_residuals[1:min_length]
residual_dates <- full_data$date[1:min_length]

# Create a data frame for residuals
residuals_df <- data.frame(
  date = residual_dates,
  sarima = as.numeric(sarima_residuals),
  garch = as.numeric(garch_residuals)
)

# Plot residuals
plot(residuals_df$date, residuals_df$sarima, type="l", col="blue", lwd=1,
     main="Model Residuals Comparison",
     xlab="Date", ylab="Residuals", xaxt="n", 
     ylim=range(c(residuals_df$sarima, residuals_df$garch), na.rm=TRUE))
lines(residuals_df$date, residuals_df$garch, col="red", lwd=1)
abline(h=0, lty=2, col="black")

# Add date axis
axis.Date(1, at=seq(min(residuals_df$date), max(residuals_df$date), by="6 months"), format="%b %Y")

# Add legend
legend("topleft", 
       legend=c("SARIMA Residuals", "GARCH Residuals"),
       col=c("blue", "red"),
       lwd=c(1, 1),
       bg="white")

dev.off()

# Create a performance metrics comparison table
sarima_mse <- mean((ts_data - sarima_fitted)^2, na.rm = TRUE)
sarima_rmse <- sqrt(sarima_mse)
sarima_mae <- mean(abs(ts_data - sarima_fitted), na.rm = TRUE)
non_zero <- which(ts_data != 0)
sarima_mape <- mean(abs((ts_data[non_zero] - sarima_fitted[non_zero])/ts_data[non_zero]), na.rm = TRUE) * 100
sarima_sse <- sum((ts_data - sarima_fitted)^2, na.rm = TRUE)
sarima_sst <- sum((ts_data - mean(ts_data, na.rm = TRUE))^2, na.rm = TRUE)
sarima_r_squared <- 1 - (sarima_sse / sarima_sst)

garch_mse <- mean((ts_data - garch_fitted)^2, na.rm = TRUE)
garch_rmse <- sqrt(garch_mse)
garch_mae <- mean(abs(ts_data - garch_fitted), na.rm = TRUE)
garch_mape <- mean(abs((ts_data[non_zero] - garch_fitted[non_zero])/ts_data[non_zero]), na.rm = TRUE) * 100
garch_sse <- sum((ts_data - garch_fitted)^2, na.rm = TRUE)
garch_r_squared <- 1 - (garch_sse / sarima_sst)

cat("\n--- Model Performance Comparison ---\n")
performance_comparison <- data.frame(
  Metric = c("MSE", "RMSE", "MAE", "MAPE (%)", "R²"),
  SARIMA = c(sarima_mse, sarima_rmse, sarima_mae, sarima_mape, sarima_r_squared),
  GARCH = c(garch_mse, garch_rmse, garch_mae, garch_mape, garch_r_squared)
)
print(performance_comparison)

# Save the performance comparison to a text file
sink("model_comparison_metrics.txt")
cat("# SARIMA vs. GARCH Model Performance Comparison\n\n")
cat("## Performance Metrics\n\n")
cat("| Metric | SARIMA | GARCH |\n")
cat("|--------|--------|-------|\n")
for (i in 1:nrow(performance_comparison)) {
  cat(sprintf("| %s | %.4f | %.4f |\n", 
              as.character(performance_comparison$Metric[i]),
              performance_comparison$SARIMA[i],
              performance_comparison$GARCH[i]))
}

cat("\n## Model Summary\n\n")
cat("### SARIMA Model\n")
cat("- Model specification:", paste(capture.output(sarima_model)[2], collapse=" "), "\n")
cat("- AIC:", sarima_model$aic, "\n")
cat("- BIC:", sarima_model$bic, "\n\n")

cat("### GARCH Model\n")
cat("- Model specification: GARCH(1,1) with t-distribution\n")
cat("- Information criteria from GARCH model summary\n\n")

cat("## Conclusion\n\n")
if (sarima_r_squared > garch_r_squared) {
  cat("The SARIMA model provides a better fit to the historical data based on R² and error metrics.\n")
} else {
  cat("The GARCH model provides a better fit to the historical data based on R² and error metrics.\n")
}

cat("\nBoth models have strengths:\n")
cat("- SARIMA captures the seasonal patterns in the legislative data\n")
cat("- GARCH explicitly models the volatility dynamics present in the residuals\n\n")

cat("A hybrid approach might be optimal, using SARIMA for point forecasts and GARCH for volatility/uncertainty estimation.\n")
sink()

cat("\nModel comparison complete!\n")
cat("Visualization saved to 'model_comparison_fit.pdf'\n")
cat("Performance metrics saved to 'model_comparison_metrics.txt'\n")
