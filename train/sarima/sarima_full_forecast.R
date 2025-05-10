## Comprehensive SARIMA Analysis for Legislative Changes Data
## Model selection, diagnostics, full dataset fitting, and 30-day future forecast
## Merged script from time_series_analysis.R and sarima_full_forecast.R

# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Install required packages if not already installed
if (!require("forecast")) install.packages("forecast")
if (!require("tseries")) install.packages("tseries")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("lmtest")) install.packages("lmtest")
if (!require("zoo")) install.packages("zoo")
if (!require("urca")) install.packages("urca")  # For additional unit root tests

# Load required libraries
library(forecast)
library(tseries)
library(ggplot2)
library(lmtest)
library(zoo)

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
  full_data$time <- na.approx(full_data$time, na.rm=FALSE)
  
  # If there are still NAs at the beginning or end, fill with nearby values
  if(any(is.na(full_data$time))) {
    full_data$time <- na.locf(full_data$time, fromLast=TRUE)
    full_data$time <- na.locf(full_data$time)
  }
}

# Create time series object with weekly seasonality (as identified in previous analysis)
ts_data <- ts(full_data$time, frequency=7)  # Weekly seasonality

# Plot the time series
pdf("time_series_plot.pdf", width=10, height=6)
plot(ts_data, main="Time Series Plot", xlab="Time", ylab="Value")
dev.off()

# ACF and PACF
pdf("acf_pacf.pdf", width=10, height=8)
par(mfrow=c(2,1))
acf(ts_data, main="ACF of Original Series")
pacf(ts_data, main="PACF of Original Series")
dev.off()

# Check for stationarity with ADF test
cat("\n--- Augmented Dickey-Fuller Test ---\n")
adf_test <- adf.test(ts_data)
print(adf_test)

# KPSS test for additional confirmation
cat("\n--- KPSS Test ---\n")
kpss_result <- kpss.test(ts_data)
print(kpss_result)

# If non-stationary by either test, apply differencing
if (adf_test$p.value > 0.05 || kpss_result$p.value <= 0.05) {
  cat("The series appears to be non-stationary. Applying differencing...\n")
  diff_data <- diff(ts_data)
  
  # Plot differenced series
  pdf("differenced_series.pdf", width=10, height=6)
  plot(diff_data, main="First Differenced Series", xlab="Time", ylab="Differenced Value")
  dev.off()
  
  # Check stationarity of differenced series
  cat("\n--- ADF Test on Differenced Series ---\n")
  adf_diff <- adf.test(diff_data)
  print(adf_diff)
  
  cat("\n--- KPSS Test on Differenced Series ---\n")
  kpss_diff <- kpss.test(diff_data)
  print(kpss_diff)
  
  # If still non-stationary, try second differencing
  if (adf_diff$p.value > 0.05 || kpss_diff$p.value <= 0.05) {
    cat("The differenced series is still non-stationary. Applying second differencing...\n")
    diff2_data <- diff(diff_data)
    
    pdf("second_differenced_series.pdf", width=10, height=6)
    plot(diff2_data, main="Second Differenced Series", xlab="Time", ylab="Differenced Value")
    dev.off()
    
    cat("\n--- ADF Test on Second Differenced Series ---\n")
    print(adf.test(diff2_data))
    
    cat("\n--- KPSS Test on Second Differenced Series ---\n")
    print(kpss.test(diff2_data))
    
    model_data <- diff2_data
    diff_order <- 2
  } else {
    model_data <- diff_data
    diff_order <- 1
  }
} else {
  cat("The series appears to be stationary. No differencing needed.\n")
  model_data <- ts_data
  diff_data <- ts_data  # For consistency in variable names
  diff_order <- 0
}

# ACF and PACF for differenced series if applicable
if (exists("diff_data") && !identical(diff_data, ts_data)) {
  pdf("acf_pacf_diff.pdf", width=10, height=8)
  par(mfrow=c(2,1))
  acf(diff_data, main="ACF of Differenced Series")
  pacf(diff_data, main="PACF of Differenced Series")
  dev.off()
}

# Seasonal decomposition
decomp <- stl(ts_data, s.window="periodic")
pdf("seasonal_decomposition.pdf", width=10, height=8)
plot(decomp, main="Seasonal Decomposition")
dev.off()

# Generate dates for the 30-day forecast period
last_date <- max(data$date)
forecast_dates <- seq(last_date + 1, by="day", length.out=30)

# Try multiple SARIMA models with different seasonal periods
cat("\n--- Testing Multiple Seasonal Periods ---\n")

# Define seasonal periods to test based on previous findings
seasonal_periods <- c(7, 21, 35, 63)
sarima_models <- list()

for (period in seasonal_periods) {
  cat("\nTesting seasonal period:", period, "\n")
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
cat("\n--- ARIMA Model (Non-Seasonal) ---\n")
arima_model <- auto.arima(ts_data, seasonal=FALSE, stepwise=FALSE, approximation=FALSE, trace=TRUE)
print(summary(arima_model))

# Find the best seasonal period based on AIC
best_period <- 7  # Default to 7 if no better model is found
best_aic <- Inf

if (length(sarima_models) > 0) {
  for (period in names(sarima_models)) {
    if (sarima_models[[period]]$aic < best_aic) {
      best_aic <- sarima_models[[period]]$aic
      best_period <- as.numeric(period)
      best_sarima_model <- sarima_models[[period]]$model
    }
  }
  cat("\nBest seasonal period based on AIC:", best_period, "\n")
}

# Determine whether to use SARIMA or ARIMA
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

# Try also a manual Holt-Winters model based on previous analysis
cat("\n--- Manual Holt-Winters Model (Based on Previous Analysis) ---\n")
hw_model <- hw(ts_data, seasonal="additive", h=10, alpha=0.5, beta=0.01, gamma=0.1)
print(summary(hw_model))

# Save the best model
if (hw_model$model$aic < best_model$aic) {
  cat("\nHolt-Winters model has better AIC than", model_type, "model.\n")
  cat("However, we'll continue with", model_type, "for consistency and interpretability.\n")
}

# Residual analysis
residuals <- residuals(best_model)

pdf("residual_diagnostics.pdf", width=10, height=10)
par(mfrow=c(2,2))
plot(residuals, main="Residuals")
acf(residuals, main="ACF of Residuals")
hist(residuals, main="Histogram of Residuals", breaks=20)
qqnorm(residuals)
qqline(residuals)
dev.off()

# Tests on residuals
ljung_box <- Box.test(residuals, lag=10, type="Ljung-Box")
shapiro <- shapiro.test(residuals)
arch_test <- Box.test(residuals^2, lag=10, type="Ljung-Box")

# Parameter significance tests
param_tests <- coeftest(best_model)
print(param_tests)

# Create a proper parameter table for reporting
param_table <- data.frame(
  Parameter = rownames(param_tests),
  Estimate = as.numeric(param_tests[,1]),
  StdError = as.numeric(param_tests[,2]),
  Pvalue = as.numeric(param_tests[,4])
)

# Perform both train-test split evaluation and full dataset forecasting

# 1. Train-test split evaluation (for model validation)
n <- length(ts_data)
test_size <- min(30, n/5)  # Use 30 days or 20% of data, whichever is smaller
train_end <- n - test_size
train_data <- window(ts_data, end=c(1, train_end))
test_data <- window(ts_data, start=c(1, train_end + 1))

# Fit model on training data
train_model <- Arima(train_data, model=best_model)

# Forecast for the test period
test_forecast <- forecast(train_model, h=length(test_data))

# Visualize test forecast vs. actual
pdf("forecast_vs_actual.pdf", width=12, height=8)
plot(test_forecast, main="Forecast vs. Actual (Test Period)",
     xlab="Time", ylab="Value",
     include=min(60, length(train_data)),
     fcol="blue", shadecols=c("lightblue", "grey80"))

# Add actual test data
lines(test_data, col="red", lwd=2)

# Add legend
legend("topleft", legend=c("Historical", "Forecast", "Actual Test Data"),
       col=c("black", "blue", "red"), lty=1, lwd=c(1,1,2), bg="white")
dev.off()

# Plot forecast errors
pdf("forecast_error.pdf", width=12, height=6)
plot(test_data - test_forecast$mean, main="Forecast Errors",
     xlab="Time", ylab="Error (Actual - Forecast)",
     type="h", col=ifelse(test_data - test_forecast$mean >= 0, "darkred", "darkblue"))
abline(h=0, col="black", lty=2)
dev.off()

# Calculate test set performance metrics
test_mse <- mean((test_data - test_forecast$mean)^2)
test_rmse <- sqrt(test_mse)
test_mae <- mean(abs(test_data - test_forecast$mean))
test_mape <- mean(abs((test_data - test_forecast$mean)/test_data)) * 100
test_r_squared <- 1 - sum((test_data - test_forecast$mean)^2) / sum((test_data - mean(test_data))^2)

# MASE calculation for test set
naive_errors <- abs(diff(train_data))
test_scaled_errors <- abs(test_data - test_forecast$mean) / mean(naive_errors)
test_mase <- mean(test_scaled_errors)

# Theil's U for test set
test_u_num <- sqrt(mean((test_data - test_forecast$mean)^2))
test_u_denom <- sqrt(mean(test_data^2)) + sqrt(mean(test_forecast$mean^2))
test_u_stat <- test_u_num / test_u_denom

# 2. Full data forecasting (for future predictions)
# Forecast for the next 30 days (future prediction)
forecast_horizon <- 30
full_forecast <- forecast(best_model, h=forecast_horizon)

# Calculate performance metrics on the full dataset (in-sample)
fitted_values <- fitted(best_model)
mse <- mean((ts_data - fitted_values)^2, na.rm = TRUE)
rmse <- sqrt(mse)
mae <- mean(abs(ts_data - fitted_values), na.rm = TRUE)

# Avoid division by zero in MAPE calculation
non_zero <- which(ts_data != 0)
mape <- ifelse(length(non_zero) > 0,
              mean(abs((ts_data[non_zero] - fitted_values[non_zero])/ts_data[non_zero]), na.rm = TRUE) * 100,
              NA)

# R-squared
r_squared <- 1 - sum((ts_data - fitted_values)^2, na.rm = TRUE) / sum((ts_data - mean(ts_data, na.rm = TRUE))^2, na.rm = TRUE)

# Calculate MASE (Mean Absolute Scaled Error)
naive_errors <- abs(diff(ts_data))
scaled_errors <- abs(ts_data[-1] - fitted_values[-1]) / mean(naive_errors, na.rm = TRUE)
mase <- mean(scaled_errors, na.rm = TRUE)

# Calculate Theil's U statistic
u_num <- sqrt(mean((ts_data - fitted_values)^2, na.rm = TRUE))
u_denom <- sqrt(mean(ts_data^2, na.rm = TRUE)) + sqrt(mean(fitted_values^2, na.rm = TRUE))
u_stat <- u_num / u_denom

# Create performance metrics tables
test_metrics <- data.frame(
  Metric = c("MSE", "RMSE", "MAE", "MAPE (%)", "R²", "MASE", "Theil's U"),
  TestValue = c(test_mse, test_rmse, test_mae, test_mape, test_r_squared, test_mase, test_u_stat)
)

full_metrics <- data.frame(
  Metric = c("MSE", "RMSE", "MAE", "MAPE (%)", "R²", "MASE", "Theil's U"),
  FullModelValue = c(mse, rmse, mae, mape, r_squared, mase, u_stat)
)

# Output performance metrics for both approaches
cat("\n--- Model Performance Metrics ---\n")
cat("\nTest Set Performance (Train-Test Split):\n")
print(test_metrics)

cat("\nFull Model Performance (In-Sample):\n")
print(full_metrics)

# Create visualizations with the full dataset + future forecasts
pdf("sarima_full_model_forecast.pdf", width=14, height=8)
# Set up the plot layout to show both full data and zoomed forecast
layout(matrix(c(1,2), nrow=2), heights=c(3,2))

# ---- Plot 1: Full time series with forecasts ----

# Create time series dataframe for easier plotting
full_ts_df <- data.frame(
  date = full_data$date,
  value = as.numeric(ts_data),
  fitted = as.numeric(fitted_values)
)

# Create data frame with forecast dates and values for plotting
forecast_df <- data.frame(
  date = forecast_dates, 
  forecast = as.numeric(full_forecast$mean),
  lower_80 = as.numeric(full_forecast$lower[,1]),
  upper_80 = as.numeric(full_forecast$upper[,1]),
  lower_95 = as.numeric(full_forecast$lower[,2]),
  upper_95 = as.numeric(full_forecast$upper[,2])
)

# Create the full time series plot
par(mar=c(3,4,3,2))
plot(full_ts_df$date, full_ts_df$value, type="l", col="black", lwd=1.5,
     main="Full Time Series with 30-Day SARIMA Forecast",
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

# ---- Plot 2: Zoomed view of forecast period and recent history ----
par(mar=c(4,4,3,2))

# Determine how many days of historical data to show (60 days)
recent_days <- 60
recent_start_date <- max(min(full_ts_df$date), max(full_ts_df$date) - recent_days)
recent_ts_df <- full_ts_df[full_ts_df$date >= recent_start_date, ]

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

# Create a markdown report
sink("sarima_full_model_report.md")

cat("# SARIMA Analysis Report - Full Dataset Model\n\n")

cat("## Data Overview\n\n")
cat("- Time series frequency:", attr(ts_data, "tsp")[3], "observations per cycle\n")
cat("- Time series length:", length(ts_data), "observations\n")
cat("- Date range:", as.character(min(data$date)), "to", as.character(max(data$date)), "\n\n")

cat("## Stationarity Analysis\n\n")
cat("- Augmented Dickey-Fuller Test p-value:", adf_test$p.value, "\n")
cat("- Interpretation:", ifelse(adf_test$p.value <= 0.05, "Series is stationary", "Series is non-stationary"), "\n\n")

cat("## Model Selection\n\n")
cat("- Best model type:", model_type, "\n")
cat("- Best seasonal period:", ifelse(model_type == "SARIMA", best_period, "N/A (non-seasonal model)"), "\n")
cat("- Model specification:", paste(capture.output(best_model)[2], collapse=" "), "\n")
cat("- AIC:", best_model$aic, "\n")
cat("- BIC:", best_model$bic, "\n\n")

cat("## Parameter Estimates & Significance\n\n")
# Format and print coefficient table
for (i in 1:nrow(param_table)) {
  param_name <- param_table$Parameter[i]
  estimate <- param_table$Estimate[i]
  p_value <- param_table$Pvalue[i]
  cat("- **", param_name, "**: ", round(estimate, 4), 
      " (p-value: ", format.pval(p_value, digits=3), ")\n", sep="")
}
cat("\n")

cat("## Model Diagnostics\n\n")
cat("- Ljung-Box Test for Autocorrelation: p-value =", round(ljung_box$p.value, 4), "\n")
cat("  - Interpretation:", ifelse(ljung_box$p.value > 0.05, "No significant autocorrelation in residuals", "Residuals show significant autocorrelation"), "\n\n")
cat("- Shapiro-Wilk Test for Normality: p-value =", round(shapiro$p.value, 4), "\n")
cat("  - Interpretation:", ifelse(shapiro$p.value > 0.05, "Residuals are normally distributed", "Residuals are not normally distributed"), "\n\n")
cat("- Box-Ljung Test for ARCH Effects: p-value =", round(arch_test$p.value, 4), "\n")
cat("  - Interpretation:", ifelse(arch_test$p.value > 0.05, "No significant ARCH effects in residuals", "ARCH effects present in residuals"), "\n\n")

cat("## Performance Metrics\n\n")
cat("### Test Set Performance (Train-Test Split)\n\n")
for (i in 1:nrow(test_metrics)) {
  cat("- **", as.character(test_metrics$Metric[i]), "**: ", 
      round(test_metrics$TestValue[i], 4), "\n", sep="")
}
cat("\n")

cat("### Full Model Performance (In-Sample)\n\n")
for (i in 1:nrow(full_metrics)) {
  cat("- **", as.character(full_metrics$Metric[i]), "**: ", 
      round(full_metrics$FullModelValue[i], 4), "\n", sep="")
}
cat("\n")

cat("## Forecast Approaches\n\n")
cat("This analysis implements two complementary forecasting approaches:\n\n")
cat("1. **Train-Test Split Evaluation**:\n")
cat("   - Uses the last", length(test_data), "days of data as a test set\n")
cat("   - Allows validation of model performance on known historical data\n")
cat("   - Provides confidence metrics for forecast accuracy\n\n")
cat("2. **Full Dataset Forecasting**:\n")
cat("   - Uses the complete historical dataset to build the model\n")
cat("   - Produces a true 30-day future forecast (from", as.character(forecast_dates[1]), "to", as.character(forecast_dates[30]), ")\n")
cat("   - Maximizes information utilization for optimal predictions\n\n")

cat("The forecast for the next 30 days (from ", as.character(forecast_dates[1]), 
    " to ", as.character(forecast_dates[30]), ") is visualized in the accompanying charts.\n\n", sep="")

cat("## Conclusion\n\n")
cat("The ", model_type, " model with ", 
    ifelse(model_type == "SARIMA", paste("period", best_period), "no seasonality"),
    " was identified as the best fit for the legislative changes data. ", sep="")

if (model_type == "SARIMA") {
  cat("The presence of weekly seasonality (period=7) confirms previous findings about cyclical patterns in legislative activities.\n\n")
} else {
  cat("While previous analyses found evidence of seasonality, this comprehensive model suggests the time series may be better represented without explicit seasonal components.\n\n")
}

if (r_squared > 0.3) {
  cat("With an R² of ", round(r_squared, 4), ", the model explains a significant portion of the variance in the legislative data. ", sep="")
} else if (r_squared > 0) {
  cat("With an R² of ", round(r_squared, 4), ", the model explains some of the variance in the legislative data, though substantial unexplained variation remains. ", sep="")
} else {
  cat("The negative R² value suggests that the complex nature of legislative activities may require additional external variables or different modeling approaches. ")
}

if (ljung_box$p.value > 0.05) {
  cat("The lack of autocorrelation in the residuals indicates that the model has successfully captured the time-dependent structure in the data.\n\n")
} else {
  cat("Some autocorrelation remains in the residuals, suggesting there may be additional temporal patterns not captured by the model.\n\n")
}

if (arch_test$p.value <= 0.05) {
  cat("The presence of ARCH effects in the residuals suggests that a GARCH-type model might be beneficial for capturing the volatility dynamics in legislative activities.\n\n")
}

cat("The 30-day forecast provides valuable insights for policy planning and resource allocation, leveraging the full historical dataset to make the most accurate possible predictions about future legislative activity levels.\n")

sink()

cat("\nComprehensive SARIMA analysis complete with both model validation and 30-day future forecast.\n")
cat("Model validation results can be found in 'forecast_vs_actual.pdf' and 'forecast_error.pdf'.\n")
cat("Full dataset forecast visualization is available in 'sarima_full_model_forecast.pdf'.\n")
cat("A comprehensive report has been saved to 'sarima_full_model_report.md'.\n")
