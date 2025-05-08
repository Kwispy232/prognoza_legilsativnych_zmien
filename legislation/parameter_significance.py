"""
Parameter significance testing for Holt-Winters models

This module provides functions for testing the statistical significance of parameters
in Holt-Winters exponential smoothing models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

def analyze_parameter_significance(full_series, model_params, fit_params, fitted_model, n_bootstrap=1000):
    """
    Analyze statistical significance of Holt-Winters model parameters using multiple methods:
    1. Bootstrap confidence intervals
    2. Model comparison tests
    3. Residual analysis
    
    Parameters:
    -----------
    full_series : pd.Series
        The time series data
    model_params : dict
        Parameters for the ExponentialSmoothing model
    fit_params : dict
        Parameters used for model fitting
    fitted_model : HoltWintersResults
        The fitted model object
    n_bootstrap : int
        Number of bootstrap iterations
    """
    print("\n" + "=" * 50)
    print("Statistical Significance Analysis of Model Parameters")
    print("=" * 50)
    
    # 1. Bootstrap method for parameter significance
    bootstrap_results = bootstrap_parameter_significance(full_series, model_params, fit_params, n_bootstrap)
    
    # 2. Model comparison approach
    comparison_results = test_parameter_significance_by_comparison(full_series, model_params, fit_params)
    
    # 3. Residual analysis
    residual_results = residual_significance_analysis(fitted_model, full_series)
    
    # Display results
    print("\n1. Bootstrap Analysis (95% Confidence Intervals):")
    print("-" * 50)
    for param, stats in bootstrap_results.items():
        param_name = param.replace('_', ' ').title()
        print(f"\n{param_name}:")
        print(f"  Estimate: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  95% CI: ({stats['CI_95%'][0]:.4f}, {stats['CI_95%'][1]:.4f})")
        print(f"  p-value: {stats['p_value']:.4f}")
        significance = "Yes" if stats['significant_at_5%'] else "No"
        print(f"  Significant at 5%: {significance}")
    
    print("\n2. Model Comparison Tests:")
    print("-" * 50)
    for param, stats in comparison_results.items():
        param_name = param.replace('_', ' ').title()
        print(f"\n{param_name}:")
        print(f"  LR Test p-value: {stats['p_value']:.4f}")
        print(f"  AIC Difference: {stats['AIC_difference']:.4f}")
        print(f"  Preferred Model: {stats['AIC_preferred_model']}")
    
    print("\n3. Residual Analysis:")
    print("-" * 50)
    print("\nLjung-Box Test (for autocorrelation in residuals):")
    print(f"  p-value: {residual_results['Ljung_Box_7']['p_value']:.4f}")
    print(f"  Residuals show autocorrelation: {'Yes' if residual_results['Ljung_Box_7']['significant'] else 'No'}")
    
    print("\nShapiro-Wilk Test (for normality of residuals):")
    print(f"  p-value: {residual_results['Shapiro_Normality']['p_value']:.4f}")
    print(f"  Residuals are normal: {'No' if residual_results['Shapiro_Normality']['significant'] else 'Yes'}")
    
    print("\nCorrelation with Fitted Values:")
    print(f"  Correlation: {residual_results['Correlation_with_Fitted']['correlation']:.4f}")
    print(f"  p-value: {residual_results['Correlation_with_Fitted']['p_value']:.4f}")
    print(f"  Significant correlation: {'Yes' if residual_results['Correlation_with_Fitted']['significant'] else 'No'}")
    
    print("\n" + "=" * 50)
    print("Summary of Parameter Significance")
    print("=" * 50)
    
    for param in bootstrap_results.keys():
        param_name = param.replace('_', ' ').title()
        bootstrap_sig = bootstrap_results[param]['significant_at_5%']
        
        # Check if parameter is in comparison results
        comparison_sig = "N/A"
        if param in comparison_results:
            comparison_sig = comparison_results[param]['p_value'] < 0.05
        
        print(f"{param_name}:")
        print(f"  Bootstrap method: {'Significant' if bootstrap_sig else 'Not significant'}")
        print(f"  Model comparison: {'Significant' if comparison_sig == True else ('Not significant' if comparison_sig == False else 'N/A')}")
        print(f"  Overall assessment: {'Significant' if bootstrap_sig or comparison_sig == True else 'Not significant'}")

def bootstrap_parameter_significance(full_series, model_params, fit_params, n_bootstrap=1000):
    """
    Test parameter significance using bootstrap method
    
    Parameters:
    -----------
    full_series : pd.Series
        Time series data
    model_params : dict
        Model parameters for ExponentialSmoothing
    fit_params : dict
        Fit parameters for model.fit()
    n_bootstrap : int
        Number of bootstrap iterations
        
    Returns:
    --------
    dict with statistical significance information for each parameter
    """
    # Parameters to analyze
    params_to_analyze = ['smoothing_level']
    if 'smoothing_trend' in fit_params:
        params_to_analyze.append('smoothing_trend')
    if 'smoothing_seasonal' in fit_params:
        params_to_analyze.append('smoothing_seasonal')
    
    # Store bootstrap parameters
    bootstrap_params = {param: [] for param in params_to_analyze}
    
    # Perform bootstrap
    for i in range(n_bootstrap):
        try:
            # Create bootstrap sample (sampling with replacement)
            indices = np.random.choice(len(full_series), size=len(full_series), replace=True)
            bootstrap_sample = full_series.iloc[indices].sort_index()
            
            # Fit model
            bootstrap_model = ExponentialSmoothing(bootstrap_sample, **model_params)
            bootstrap_fit = bootstrap_model.fit(**fit_params)
            
            # Extract parameters
            for param in params_to_analyze:
                if param in bootstrap_fit.params:
                    bootstrap_params[param].append(bootstrap_fit.params[param])
        except:
            # Skip failed fits
            continue
    
    # Calculate 95% confidence intervals and p-values
    results = {}
    for param in params_to_analyze:
        values = np.array(bootstrap_params[param])
        if len(values) > 0:
            lower_ci = np.percentile(values, 2.5)
            upper_ci = np.percentile(values, 97.5)
            
            # Calculate p-value (two-tailed test against null hypothesis that parameter = 0)
            # Method 1: Based on confidence interval
            significant = (lower_ci > 0) or (upper_ci < 0)
            
            # Method 2: Directly compute p-value (if parameter = 0 is the null hypothesis)
            t_stat = np.mean(values) / (np.std(values) / np.sqrt(len(values)))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(values)-1))
            
            results[param] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'CI_95%': (lower_ci, upper_ci),
                'significant_at_5%': significant,
                'p_value': p_value
            }
    
    # Visualization
    fig, axes = plt.subplots(len(params_to_analyze), 1, figsize=(10, 3*len(params_to_analyze)))
    if len(params_to_analyze) == 1:
        axes = [axes]
        
    for i, param in enumerate(params_to_analyze):
        if param in results:
            sns.histplot(bootstrap_params[param], kde=True, ax=axes[i])
            axes[i].axvline(x=0, color='red', linestyle='--', label='Parameter = 0')
            axes[i].set_title(f'Bootstrap Distribution: {param}')
            axes[i].axvline(x=results[param]['CI_95%'][0], color='green', linestyle=':', label='95% CI')
            axes[i].axvline(x=results[param]['CI_95%'][1], color='green', linestyle=':')
            axes[i].legend()
        
    plt.tight_layout()
    plt.savefig('parameter_significance_bootstrap.png')
    plt.close()
    
    return results

def test_parameter_significance_by_comparison(full_series, model_params, fit_params):
    """
    Test parameter significance by comparing models with and without each parameter
    """
    results = {}
    
    # Fit baseline model
    try:
        baseline_model = ExponentialSmoothing(full_series, **model_params)
        baseline_fit = baseline_model.fit(**fit_params)
        baseline_aic = baseline_fit.aic
        baseline_likelihood = baseline_fit.loglike
        
        # Test each parameter by setting it to its default/null value
        params_to_test = {
            'smoothing_level': 0.5,  # Default value to test against
            'smoothing_trend': 0.0,  # Testing no trend effect
            'smoothing_seasonal': 0.0  # Testing no seasonal effect
        }
        
        for param, null_value in params_to_test.items():
            if param not in fit_params:
                continue
                
            # Create modified parameters with current parameter set to null
            modified_fit_params = fit_params.copy()
            modified_fit_params[param] = null_value
            
            try:
                # Fit restricted model
                restricted_model = ExponentialSmoothing(full_series, **model_params)
                restricted_fit = restricted_model.fit(**modified_fit_params)
                
                # Calculate likelihood ratio test statistic
                # -2 * log(likelihood_ratio) ~ chi-squared with df=1
                lr_stat = -2 * (restricted_fit.loglike - baseline_likelihood)
                p_value = 1 - stats.chi2.cdf(lr_stat, 1)
                
                # Compare AIC
                aic_diff = restricted_fit.aic - baseline_aic
                
                results[param] = {
                    'original_value': fit_params[param],
                    'null_value': null_value,
                    'LR_statistic': lr_stat,
                    'p_value': p_value,
                    'significant_at_5%': p_value < 0.05,
                    'AIC_difference': aic_diff,
                    'AIC_preferred_model': 'Full' if aic_diff > 0 else 'Restricted'
                }
            except:
                # Skip if this comparison fails
                continue
    except:
        # If baseline model fails, we can't do comparison
        pass
        
    return results

def residual_significance_analysis(fit, full_series):
    """
    Analyze residuals to assess model specification and parameter significance
    """
    # Calculate residuals
    residuals = full_series - fit.fittedvalues
    
    # Statistical tests on residuals
    try:
        # Ljung-Box test for autocorrelation in residuals (use lag=7 for weekly data)
        lb_result = acorr_ljungbox(residuals, lags=[7])
        lb_stat = lb_result.iloc[0, 0] if hasattr(lb_result, 'iloc') else lb_result[0][0]
        lb_pvalue = lb_result.iloc[0, 1] if hasattr(lb_result, 'iloc') else lb_result[1][0]
        
        # Normality test
        shapiro_result = stats.shapiro(residuals)
        
        # Check for residual relationship with fitted values
        corr_with_fitted, p_value = stats.pearsonr(fit.fittedvalues, residuals)
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals time plot
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].axhline(y=0, color='r', linestyle='-')
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=20, density=True)
        x = np.linspace(min(residuals), max(residuals), 100)
        axes[0, 1].plot(x, stats.norm.pdf(x, np.mean(residuals), np.std(residuals)))
        axes[0, 1].set_title('Residuals Distribution')
        
        # QQ plot
        stats.probplot(residuals, plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # Residuals vs Fitted
        axes[1, 1].scatter(fit.fittedvalues, residuals)
        axes[1, 1].set_title('Residuals vs Fitted')
        axes[1, 1].axhline(y=0, color='r', linestyle='-')
        
        plt.tight_layout()
        plt.savefig('residual_analysis.png')
        plt.close()
        
        # Return statistical test results
        return {
            'Ljung_Box_7': {
                'statistic': lb_stat,
                'p_value': lb_pvalue,
                'significant': lb_pvalue < 0.05
            },
            'Shapiro_Normality': {
                'statistic': shapiro_result[0],
                'p_value': shapiro_result[1],
                'significant': shapiro_result[1] < 0.05
            },
            'Correlation_with_Fitted': {
                'correlation': corr_with_fitted,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        }
    except Exception as e:
        print(f"Error in residual analysis: {e}")
        # Return empty results if tests fail
        return {
            'Ljung_Box_7': {'statistic': None, 'p_value': None, 'significant': False},
            'Shapiro_Normality': {'statistic': None, 'p_value': None, 'significant': False},
            'Correlation_with_Fitted': {'correlation': None, 'p_value': None, 'significant': False}
        }
