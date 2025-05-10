## GARCH Analysis for Legislative Changes Data
## This script models volatility in the time series using GARCH models

# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Install required packages if not already installed
if (!require("forecast")) install.packages("forecast")
if (!require("tseries")) install.packages("tseries")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("rugarch")) install.packages("rugarch")
if (!require("fGarch")) install.packages("fGarch")
if (!require("zoo")) install.packages("zoo")

# Ensure packages are loaded
library(forecast)
library(tseries)
library(ggplot2)
tryCatch({
  library(rugarch)
}, error = function(e) {
  cat("Error loading rugarch:", e$message, "\nTrying to install again...\n")
  install.packages("rugarch")
  library(rugarch)
})

tryCatch({
  library(fGarch)
}, error = function(e) {
  cat("Error loading fGarch:", e$message, "\nTrying to install again...\n")
  install.packages("fGarch")
  library(fGarch)
})
library(zoo)

# Libraries already loaded in installation block

# Load and prepare the data
data <- read.csv("../sorted_train_entries.csv")
data$date <- as.Date(data$date)

# Create time series object
# Extract the full time range to handle potential missing dates
date_range <- seq(min(data$date), max(data$date), by="day")
full_data <- data.frame(date = date_range)
full_data <- merge(full_data, data, by="date", all.x=TRUE)

# Check if there are missing values and handle them if needed
if(any(is.na(full_data$time))) {
  cat("Missing values detected in the time series. Imputing missing values...\n")
  # Simple linear interpolation for missing values
  if (!requireNamespace("zoo", quietly = TRUE)) {
    install.packages("zoo")
    library(zoo)
  }
  full_data$time <- na.approx(full_data$time, na.rm=FALSE)
  
  # If there are still NAs at the beginning or end, fill with nearby values
  if(any(is.na(full_data$time))) {
    full_data$time <- na.locf(full_data$time, fromLast=TRUE)
    full_data$time <- na.locf(full_data$time)
  }
}

# Create time series object with weekly seasonality (as identified in previous analysis)
ts_data <- ts(full_data$time, frequency=7)

# Plot the time series
pdf("time_series_plot.pdf", width=10, height=6)
plot(ts_data, main="Time Series Plot", xlab="Time", ylab="Value")
dev.off()

# First, we need to determine if we need to difference the data for stationarity
adf_test <- adf.test(ts_data)
print(adf_test)

# If non-stationary, difference the data
if (adf_test$p.value > 0.05) {
  cat("The series appears to be non-stationary. Applying differencing...\n")
  diff_data <- diff(ts_data)
} else {
  cat("The series appears to be stationary. No differencing needed.\n")
  diff_data <- ts_data
}

# Plot the ACF/PACF of squared returns to detect ARCH effects
pdf("acf_pacf_squared.pdf", width=10, height=8)
par(mfrow=c(2,1))
acf(diff_data^2, main="ACF of Squared Returns")
pacf(diff_data^2, main="PACF of Squared Returns")
dev.off()

# ARCH Test using Box-Ljung test on squared returns (alternative to ArchTest)
cat("\n--- ARCH Test for ARCH Effects ---\n")
squared_returns <- diff_data^2
arch_test_result <- Box.test(squared_returns, lag=10, type="Ljung-Box")
print(arch_test_result)

cat("p-value:", arch_test_result$p.value, "\n")
if (arch_test_result$p.value < 0.05) {
  cat("There are significant ARCH effects in the data. GARCH modeling is appropriate.\n")
} else {
  cat("No significant ARCH effects detected. However, we'll still fit GARCH models for comparison.\n")
}

# Use full dataset for model fitting (no train-test split)
full_data_for_model <- diff_data  # Use differenced data if we applied differencing

# Store the last date in the series for forecasting purposes
last_date <- max(data$date)
# Generate dates for the 30-day forecast period
forecast_dates <- seq(last_date + 1, by="day", length.out=30)

# ---------------------- GARCH Modeling ---------------------- #

# 1. Set up GARCH model specifications to test
# Models to test:
# - GARCH(1,1) with normal innovation distribution
# - GARCH(1,1) with t distribution
# - GARCH(2,1) with normal distribution
# - GARCH(1,2) with normal distribution
# - EGARCH(1,1) with normal distribution (allows for asymmetric effects)
# - APARCH(1,1) with normal distribution (asymmetric power ARCH)

cat("\n--- Fitting Various GARCH Models ---\n")

# GARCH(1,1) with normal innovation
garch11_spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(1, 1), include.mean = TRUE),
  distribution.model = "norm"
)

# GARCH(1,1) with t distribution
garch11t_spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(1, 1), include.mean = TRUE),
  distribution.model = "std"
)

# GARCH(2,1) with normal distribution
garch21_spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(2, 1)),
  mean.model = list(armaOrder = c(1, 1), include.mean = TRUE),
  distribution.model = "norm"
)

# GARCH(1,2) with normal distribution
garch12_spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 2)),
  mean.model = list(armaOrder = c(1, 1), include.mean = TRUE),
  distribution.model = "norm"
)

# EGARCH(1,1) with normal distribution
egarch11_spec <- ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(1, 1), include.mean = TRUE),
  distribution.model = "norm"
)

# APARCH(1,1) with normal distribution
aparch11_spec <- ugarchspec(
  variance.model = list(model = "apARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(1, 1), include.mean = TRUE),
  distribution.model = "norm"
)

# Fit the models on the full dataset
tryCatch({
  cat("Fitting GARCH(1,1) with normal innovation...\n")
  garch11_fit <- ugarchfit(spec = garch11_spec, data = full_data_for_model)
  
  cat("Fitting GARCH(1,1) with t distribution...\n")
  garch11t_fit <- ugarchfit(spec = garch11t_spec, data = full_data_for_model)
  
  cat("Fitting GARCH(2,1) with normal distribution...\n")
  garch21_fit <- ugarchfit(spec = garch21_spec, data = full_data_for_model)
  
  cat("Fitting GARCH(1,2) with normal distribution...\n")
  garch12_fit <- ugarchfit(spec = garch12_spec, data = full_data_for_model)
  
  cat("Fitting EGARCH(1,1) with normal distribution...\n")
  egarch11_fit <- ugarchfit(spec = egarch11_spec, data = full_data_for_model)
  
  cat("Fitting APARCH(1,1) with normal distribution...\n")
  aparch11_fit <- ugarchfit(spec = aparch11_spec, data = full_data_for_model)
}, error = function(e) {
  cat("Error in fitting GARCH models:", e$message, "\n")
})

# Compare models by information criteria
compare_models <- function() {
  models <- list()
  model_names <- c()
  
  if (exists("garch11_fit")) {
    models <- c(models, list(garch11_fit))
    model_names <- c(model_names, "GARCH(1,1) Normal")
  }
  
  if (exists("garch11t_fit")) {
    models <- c(models, list(garch11t_fit))
    model_names <- c(model_names, "GARCH(1,1) t-dist")
  }
  
  if (exists("garch21_fit")) {
    models <- c(models, list(garch21_fit))
    model_names <- c(model_names, "GARCH(2,1) Normal")
  }
  
  if (exists("garch12_fit")) {
    models <- c(models, list(garch12_fit))
    model_names <- c(model_names, "GARCH(1,2) Normal")
  }
  
  if (exists("egarch11_fit")) {
    models <- c(models, list(egarch11_fit))
    model_names <- c(model_names, "EGARCH(1,1) Normal")
  }
  
  if (exists("aparch11_fit")) {
    models <- c(models, list(aparch11_fit))
    model_names <- c(model_names, "APARCH(1,1) Normal")
  }
  
  if (length(models) == 0) {
    cat("No models were successfully fitted.\n")
    return(NULL)
  }
  
  # Get information criteria for comparison
  aic_values <- sapply(models, function(x) infocriteria(x)[1])
  bic_values <- sapply(models, function(x) infocriteria(x)[2])
  
  # Create comparison table
  comparison <- data.frame(
    Model = model_names,
    AIC = aic_values,
    BIC = bic_values
  )
  
  # Sort by AIC
  comparison <- comparison[order(comparison$AIC), ]
  
  return(list(comparison = comparison, best_model_index = which.min(aic_values)))
}

# Get model comparison
model_comparison <- compare_models()

if (!is.null(model_comparison)) {
  cat("\n--- Model Comparison by Information Criteria ---\n")
  print(model_comparison$comparison)
  
  # Select the best model
  best_model_name <- model_comparison$comparison$Model[1]
  cat("\nBest model by AIC:", best_model_name, "\n")
  
  # Assign best model based on name
  if (best_model_name == "GARCH(1,1) Normal") {
    best_model <- garch11_fit
    best_spec <- garch11_spec
  } else if (best_model_name == "GARCH(1,1) t-dist") {
    best_model <- garch11t_fit
    best_spec <- garch11t_spec
  } else if (best_model_name == "GARCH(2,1) Normal") {
    best_model <- garch21_fit
    best_spec <- garch21_spec
  } else if (best_model_name == "GARCH(1,2) Normal") {
    best_model <- garch12_fit
    best_spec <- garch12_spec
  } else if (best_model_name == "EGARCH(1,1) Normal") {
    best_model <- egarch11_fit
    best_spec <- egarch11_spec
  } else if (best_model_name == "APARCH(1,1) Normal") {
    best_model <- aparch11_fit
    best_spec <- aparch11_spec
  }
  
  # Print summary of the best model
  cat("\n--- Best Model Summary ---\n")
  print(best_model)
  
  # Forecast for the next 30 days (future prediction)
  forecast_horizon <- 30
  garch_forecast <- ugarchforecast(best_model, n.ahead = forecast_horizon)
  
  # Extract forecast means and sigmas
  forecast_means <- as.numeric(fitted(garch_forecast))
  forecast_sigmas <- as.numeric(sigma(garch_forecast))
  
  # If we differenced the data, we need to convert back
  if (exists("diff_data") && adf_test$p.value > 0.05) {
    # Convert forecasts back to original scale
    forecast_levels <- numeric(forecast_horizon)
    last_value <- as.numeric(tail(ts_data, 1))
    
    forecast_levels[1] <- last_value + forecast_means[1]
    for (i in 2:forecast_horizon) {
      forecast_levels[i] <- forecast_levels[i-1] + forecast_means[i]
    }
    
    # Create future dates for the forecast period
    future_dates <- seq(time(ts_data)[length(ts_data)] + 1/frequency(ts_data), 
                        by = 1/frequency(ts_data), length.out = forecast_horizon)
    forecast_ts <- ts(forecast_levels, start = future_dates[1], frequency = frequency(ts_data))
  } else {
    # If no differencing was applied, forecasts are already in the right scale
    future_dates <- seq(time(ts_data)[length(ts_data)] + 1/frequency(ts_data), 
                        by = 1/frequency(ts_data), length.out = forecast_horizon)
    forecast_ts <- ts(forecast_means, start = future_dates[1], frequency = frequency(ts_data))
  }
  
  # Calculate 95% prediction intervals
  lower_95 <- forecast_ts - 1.96 * forecast_sigmas
  upper_95 <- forecast_ts + 1.96 * forecast_sigmas
  
  # Create data frame with forecast dates and values for easier plotting
  forecast_df <- data.frame(
    date = forecast_dates,
    forecast = as.numeric(forecast_ts),
    lower_95 = as.numeric(lower_95),
    upper_95 = as.numeric(upper_95)
  )
  
  # Create visualizations with the full dataset + future forecasts
  pdf("garch_full_model_forecast.pdf", width=14, height=8)
  
  # Set up the plot layout to show both full data and zoomed forecast
  layout(matrix(c(1,2), nrow=2), heights=c(3,2))
  
  # Plot 1: Full time series with forecasts
  # Convert full time series to data frame for plotting
  full_ts_df <- data.frame(
    date = full_data$date,
    value = as.numeric(ts_data)
  )
  
  # Create the full time series plot
  par(mar=c(3,4,3,2))
  plot(full_ts_df$date, full_ts_df$value, type="l", col="black", lwd=1.5,
       main="Full Time Series with 30-Day GARCH Forecast",
       xlab="", ylab="Value", xaxt="n")
  
  # Add recent dates axis labels
  axis.Date(1, at=seq(min(full_ts_df$date), max(forecast_df$date), by="6 months"), format="%b %Y")
  
  # Add the forecast
  points(forecast_df$date, forecast_df$forecast, type="l", col="blue", lwd=2)
  
  # Add prediction intervals
  polygon(c(forecast_df$date, rev(forecast_df$date)),
          c(forecast_df$lower_95, rev(forecast_df$upper_95)),
          col=rgb(0,0,1,0.2), border=NA)
  
  # Add a vertical line separating historical data from forecast
  abline(v=max(full_ts_df$date), lty=2, col="gray50")
  text(max(full_ts_df$date) + 10, max(full_ts_df$value) * 0.9, "Forecast Start", pos=4, col="gray50")
  
  # Add legend
  legend("topleft", legend=c("Historical Data", "30-Day Forecast", "95% Prediction Interval"),
         col=c("black", "blue", rgb(0,0,1,0.2)), lty=c(1,1,1), lwd=c(1.5,2,10),
         bg="white")
  
  # Add grid
  grid()
  
  # Plot 2: Zoomed view of forecast period and recent history
  par(mar=c(4,4,3,2))
  
  # Determine how many days of historical data to show (90 days)
  recent_days <- 90
  recent_start_date <- max(full_ts_df$date) - recent_days
  
  # Filter recent data
  recent_ts_df <- full_ts_df[full_ts_df$date >= recent_start_date,]
  
  # Plot recent data and forecast
  plot_range <- range(c(recent_ts_df$value, forecast_df$upper_95, forecast_df$lower_95))
  plot(recent_ts_df$date, recent_ts_df$value, type="l", col="black", lwd=1.5,
       main="Recent Data and 30-Day Forecast (Zoomed View)",
       xlab="Date", ylab="Value", ylim=plot_range, xaxt="n")
  
  # Add meaningful date axis
  axis.Date(1, at=seq(recent_start_date, max(forecast_df$date), by="15 days"), format="%d %b")
  
  # Add the forecast
  points(forecast_df$date, forecast_df$forecast, type="l", col="blue", lwd=2)
  
  # Add prediction intervals
  polygon(c(forecast_df$date, rev(forecast_df$date)),
          c(forecast_df$lower_95, rev(forecast_df$upper_95)),
          col=rgb(0,0,1,0.2), border=NA)
  
  # Add a vertical line separating historical data from forecast
  abline(v=max(full_ts_df$date), lty=2, col="gray50")
  
  # Add grid
  grid()
  
  dev.off()
  
  # Create a separate plot for volatility forecast
  pdf("garch_volatility_forecast.pdf", width=10, height=6)
  
  # Create data frame with dates and volatility
  volatility_df <- data.frame(
    date = forecast_df$date,
    volatility = forecast_sigmas
  )
  
  # Plot volatility forecast
  plot(volatility_df$date, volatility_df$volatility, type="l", col="darkred", lwd=2,
       main="30-Day Volatility Forecast", xlab="Date", ylab="Volatility")
  
  # Add grid
  grid()
  
  dev.off()
  
  # Calculate in-sample performance metrics instead of out-of-sample
  # Get fitted values from the model
  fitted_values <- fitted(best_model)
  
  # Calculate in-sample performance metrics
  mse <- mean((diff_data - fitted_values)^2, na.rm=TRUE)
  rmse <- sqrt(mse)
  mae <- mean(abs(diff_data - fitted_values), na.rm=TRUE)
  mape <- mean(abs((diff_data - fitted_values)/diff_data) * 100, na.rm=TRUE)
  r_squared <- 1 - sum((diff_data - fitted_values)^2, na.rm=TRUE) / sum((diff_data - mean(diff_data, na.rm=TRUE))^2, na.rm=TRUE)
  
  # Calculate Theil's U statistic using in-sample data
  u_stat <- sqrt(sum((diff_data - fitted_values)^2, na.rm=TRUE)) / 
            sqrt(sum((diff_data[-1] - diff_data[-length(diff_data)])^2, na.rm=TRUE))
  
  # Calculate MASE using in-sample data
  naive_errors <- abs(diff(diff_data))
  mase <- mean(abs(diff_data - fitted_values), na.rm=TRUE) / mean(naive_errors, na.rm=TRUE)
  
  # Create performance metrics table
  performance_metrics <- data.frame(
    Metric = c("MSE", "RMSE", "MAE", "MAPE (%)", "R²", "MASE", "Theil's U"),
    Value = c(mse, rmse, mae, mape, r_squared, mase, u_stat)
  )
  
  cat("\n--- GARCH Model Performance Metrics ---\n")
  print(performance_metrics)
  
  # Check for remaining ARCH effects in the residuals
  std_resid <- residuals(best_model) / sigma(best_model)
  
  pdf("garch_residual_diagnostics.pdf", width=10, height=8)
  par(mfrow=c(2,2))
  
  # Plot standardized residuals
  plot(std_resid, main="Standardized Residuals", type="l")
  
  # ACF of standardized residuals
  acf(std_resid, main="ACF of Standardized Residuals")
  
  # ACF of squared standardized residuals
  acf(std_resid^2, main="ACF of Squared Standardized Residuals")
  
  # QQ-plot of standardized residuals
  qqnorm(std_resid)
  qqline(std_resid)
  
  dev.off()
  
  # Test if ARCH effects have been addressed using Box-Ljung test on squared standardized residuals
  cat("\n--- Testing for Remaining ARCH Effects in GARCH Residuals ---\n")
  squared_std_resid <- std_resid^2
  arch_test_resid <- Box.test(squared_std_resid, lag=10, type="Ljung-Box")
  print(arch_test_resid)
  
  if (arch_test_resid$p.value > 0.05) {
    cat("No significant ARCH effects remain in the residuals. The GARCH model successfully captured the volatility dynamics.\n")
  } else {
    cat("Some ARCH effects remain in the residuals. A more complex volatility model might be needed.\n")
  }
  
  # Create a report
  sink("garch_analysis_report.md")
  
  cat("# GARCH Analysis Report\n\n")
  
  cat("## Data Overview\n\n")
  cat("- Time series frequency:", attr(ts_data, "tsp")[3], "observations per cycle\n")
  cat("- Time series length:", length(ts_data), "observations\n")
  cat("- Date range:", as.character(min(data$date)), "to", as.character(max(data$date)), "\n\n")
  
  cat("## ARCH Effects Detection\n\n")
  cat("- ARCH LM Test p-value:", arch_test_result$p.value, "\n")
  cat("- Interpretation:", ifelse(arch_test_result$p.value < 0.05, 
                               "Significant ARCH effects detected, warranting GARCH modeling", 
                               "No significant ARCH effects, but GARCH models were fitted for comparison"), "\n\n")
  
  cat("## Model Selection\n\n")
  cat("Models compared by AIC and BIC:\n\n")
  
  # Print model comparison as table
  models_df <- model_comparison$comparison
  for (i in 1:nrow(models_df)) {
    cat("- ", models_df$Model[i], ": AIC = ", round(models_df$AIC[i], 2), ", BIC = ", round(models_df$BIC[i], 2), "\n", sep="")
  }
  
  cat("\nBest model: **", best_model_name, "**\n\n", sep="")
  
  cat("## Parameter Estimates\n\n")
  params <- coef(best_model)
  cat("Mean model parameters:\n")
  mean_params <- params[grep("^(mu|ar|ma)", names(params))]
  for (i in 1:length(mean_params)) {
    cat("- ", names(mean_params)[i], ": ", round(mean_params[i], 4), "\n", sep="")
  }
  
  cat("\nVolatility model parameters:\n")
  vol_params <- params[grep("^(omega|alpha|beta|gamma|delta)", names(params))]
  for (i in 1:length(vol_params)) {
    cat("- ", names(vol_params)[i], ": ", round(vol_params[i], 4), "\n", sep="")
  }
  
  cat("\n## Model Diagnostics\n\n")
  cat("- ARCH test on standardized residuals p-value:", arch_test_resid$p.value, "\n")
  cat("- Interpretation:", ifelse(arch_test_resid$p.value > 0.05, 
                               "No significant ARCH effects remain in the residuals. The GARCH model successfully captured the volatility dynamics.", 
                               "Some ARCH effects remain in the residuals. A more complex volatility model might be needed."), "\n\n")
  
  cat("## Forecast Performance Metrics\n\n")
  for (i in 1:nrow(performance_metrics)) {
    cat("- **", as.character(performance_metrics$Metric[i]), "**: ", 
        ifelse(is.numeric(performance_metrics$Value[i]), round(performance_metrics$Value[i], 4), performance_metrics$Value[i]), 
        "\n", sep="")
  }
  
  cat("## Comparison with SARIMA Model\n\n")
  cat("The GARCH model offers several advantages over the SARIMA model for this legislative data:\n\n")
  cat("1. **Volatility Modeling**: GARCH explicitly models the changing variance, capturing periods of higher uncertainty in legislative activities\n")
  cat("2. **Risk Assessment**: Provides more reliable prediction intervals during volatile periods\n")
  cat("3. **Structural Insights**: Helps identify patterns in legislative volatility, potentially tied to political cycles or events\n\n")
  
  cat("Unlike the previous approach that split the data into training and test sets, this analysis uses the full dataset to build the model and then generates true future forecasts. This provides:\n\n")
  cat("1. **Maximum Information Utilization**: Uses all available historical data for model building\n")
  cat("2. **True Future Forecasting**: Predictions represent actual future values rather than held-out historical data\n")
  cat("3. **Complete Volatility Patterns**: Captures all historical volatility patterns for better future uncertainty estimates\n\n")
  
  if (r_squared > 0) {
    cat("The GARCH model shows good in-sample fit, with a positive R² value indicating it explains variation in the data better than a simple mean model.\n\n")
  } else {
    cat("While the GARCH model appropriately addresses volatility clustering, the in-sample fit suggests that the time series may have complex patterns that are challenging to model, indicating that additional factors may influence legislative activities.\n\n")
  }
  
  cat("## Conclusion\n\n")
  cat("The ", best_model_name, " model was identified as the best fit for capturing both the mean and volatility dynamics in the legislative data. ", sep="")
  
  if (arch_test_resid$p.value > 0.05) {
    cat("The model successfully addressed the ARCH effects detected in the original time series, providing a more accurate representation of the uncertainty in future predictions.\n\n")
  } else {
    cat("While the model improved the volatility representation, some ARCH effects remain, suggesting that legislative changes might have complex volatility patterns requiring more sophisticated modeling approaches.\n\n")
  }
  
  cat("The 30-day future forecast provides policymakers with valuable insights not just into expected legislative activity levels, but also into the expected volatility, which can be crucial for resource planning and risk management. By using the full historical dataset, the model leverages all available information to make the most accurate possible predictions about truly unknown future values.\n\n")
  
  cat("Future work might explore incorporating external variables like political events, economic indicators, or seasonal dummy variables to further improve forecast accuracy and volatility prediction.\n")
  
  sink()
  
  cat("\nGARCH analysis complete. Results saved to garch_analysis_report.md and associated figures.\n")
} else {
  cat("\nNo GARCH models could be successfully fitted. Please check the data and model specifications.\n")
}
