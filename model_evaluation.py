"""
Model Evaluation and Visualization Functions
For evaluating time series forecasting models (pickups and dropoffs)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    median_absolute_error
)
from scipy import stats


def smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE).
    
    sMAPE is bounded between 0% and 200%, and handles zero values better than MAPE.
    Formula: 100 * mean(2 * |actual - predicted| / (|actual| + |predicted|))
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    float: sMAPE value (0-200%)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    denominator = np.abs(y_true) + np.abs(y_pred)
    # Avoid division by zero - if both actual and predicted are 0, error is 0
    mask = denominator != 0
    
    smape_value = np.zeros_like(y_true, dtype=float)
    smape_value[mask] = 2.0 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    
    return 100 * np.mean(smape_value)


def evaluate_basic_metrics(forecast_results_pickups, forecast_results_dropoffs, model_name="Model"):
    """
    Calculate and print basic evaluation metrics (MAE, RMSE, R²) for pickups and dropoffs.
    
    Parameters:
    -----------
    forecast_results_pickups : DataFrame
        DataFrame with 'actual' and 'predicted' columns for pickups
    forecast_results_dropoffs : DataFrame
        DataFrame with 'actual' and 'predicted' columns for dropoffs
    model_name : str
        Name of the model for display purposes
    """
    print("\n" + "="*70)
    print(f"{model_name.upper()} - BASIC EVALUATION METRICS")
    print("="*70)
    
    # Pickups metrics
    mae_pickups = mean_absolute_error(
        forecast_results_pickups['actual'],
        forecast_results_pickups['predicted']
    )
    rmse_pickups = np.sqrt(mean_squared_error(
        forecast_results_pickups['actual'],
        forecast_results_pickups['predicted']
    ))
    r2_pickups = r2_score(
        forecast_results_pickups['actual'],
        forecast_results_pickups['predicted']
    )
    
    print("\nPICKUPS:")
    print(f"  MAE:  {mae_pickups:.2f}")
    print(f"  RMSE: {rmse_pickups:.2f}")
    print(f"  R²:   {r2_pickups:.4f}")
    
    # Dropoffs metrics
    mae_dropoffs = mean_absolute_error(
        forecast_results_dropoffs['actual'],
        forecast_results_dropoffs['predicted']
    )
    rmse_dropoffs = np.sqrt(mean_squared_error(
        forecast_results_dropoffs['actual'],
        forecast_results_dropoffs['predicted']
    ))
    r2_dropoffs = r2_score(
        forecast_results_dropoffs['actual'],
        forecast_results_dropoffs['predicted']
    )
    
    print("\nDROPOFFS:")
    print(f"  MAE:  {mae_dropoffs:.2f}")
    print(f"  RMSE: {rmse_dropoffs:.2f}")
    print(f"  R²:   {r2_dropoffs:.4f}")
    
    print("\n" + "="*70)


def evaluate_extended_metrics(forecast_results_pickups, forecast_results_dropoffs, model_name="Model"):
    """
    Calculate and print extended evaluation metrics (MAPE, Median AE, Bias, Std) for pickups and dropoffs.
    
    Parameters:
    -----------
    forecast_results_pickups : DataFrame
        DataFrame with 'actual' and 'predicted' columns for pickups
    forecast_results_dropoffs : DataFrame
        DataFrame with 'actual' and 'predicted' columns for dropoffs
    model_name : str
        Name of the model for display purposes
    """
    print("\n" + "="*70)
    print(f"{model_name.upper()} - EXTENDED EVALUATION METRICS")
    print("="*70)
    
    # PICKUPS - Extended Metrics
    mae_pickups = mean_absolute_error(
        forecast_results_pickups['actual'],
        forecast_results_pickups['predicted']
    )
    rmse_pickups = np.sqrt(mean_squared_error(
        forecast_results_pickups['actual'],
        forecast_results_pickups['predicted']
    ))
    r2_pickups = r2_score(
        forecast_results_pickups['actual'],
        forecast_results_pickups['predicted']
    )
    smape_pickups = smape(
        forecast_results_pickups['actual'],
        forecast_results_pickups['predicted']
    )
    median_ae_pickups = median_absolute_error(
        forecast_results_pickups['actual'],
        forecast_results_pickups['predicted']
    )
    residuals_pickups = (forecast_results_pickups['actual'] - 
                         forecast_results_pickups['predicted'])
    bias_pickups = residuals_pickups.mean()
    std_residuals_pickups = residuals_pickups.std()
    
    print("\nPICKUPS:")
    print(f"  MAE:           {mae_pickups:.2f}")
    print(f"  RMSE:          {rmse_pickups:.2f}")
    print(f"  R²:            {r2_pickups:.4f}")
    print(f"  sMAPE:         {smape_pickups:.2f}%")
    print(f"  Median AE:     {median_ae_pickups:.2f}")
    print(f"  Bias (ME):     {bias_pickups:.2f}")
    print(f"  Std of errors: {std_residuals_pickups:.2f}")
    
    # DROPOFFS - Extended Metrics
    mae_dropoffs = mean_absolute_error(
        forecast_results_dropoffs['actual'],
        forecast_results_dropoffs['predicted']
    )
    rmse_dropoffs = np.sqrt(mean_squared_error(
        forecast_results_dropoffs['actual'],
        forecast_results_dropoffs['predicted']
    ))
    r2_dropoffs = r2_score(
        forecast_results_dropoffs['actual'],
        forecast_results_dropoffs['predicted']
    )
    smape_dropoffs = smape(
        forecast_results_dropoffs['actual'],
        forecast_results_dropoffs['predicted']
    )
    median_ae_dropoffs = median_absolute_error(
        forecast_results_dropoffs['actual'],
        forecast_results_dropoffs['predicted']
    )
    residuals_dropoffs = (forecast_results_dropoffs['actual'] - 
                          forecast_results_dropoffs['predicted'])
    bias_dropoffs = residuals_dropoffs.mean()
    std_residuals_dropoffs = residuals_dropoffs.std()
    
    print("\nDROPOFFS:")
    print(f"  MAE:           {mae_dropoffs:.2f}")
    print(f"  RMSE:          {rmse_dropoffs:.2f}")
    print(f"  R²:            {r2_dropoffs:.4f}")
    print(f"  sMAPE:         {smape_dropoffs:.2f}%")
    print(f"  Median AE:     {median_ae_dropoffs:.2f}")
    print(f"  Bias (ME):     {bias_dropoffs:.2f}")
    print(f"  Std of errors: {std_residuals_dropoffs:.2f}")
    
    print("\n" + "="*70)


def plot_actual_vs_predicted(forecast_results_pickups, forecast_results_dropoffs, model_name="Model"):
    """
    Create scatter plots of actual vs predicted values for pickups and dropoffs.
    
    Parameters:
    -----------
    forecast_results_pickups : DataFrame
        DataFrame with 'actual' and 'predicted' columns for pickups
    forecast_results_dropoffs : DataFrame
        DataFrame with 'actual' and 'predicted' columns for dropoffs
    model_name : str
        Name of the model for display purposes
    """
    # Calculate R² for titles
    r2_pickups = r2_score(
        forecast_results_pickups['actual'],
        forecast_results_pickups['predicted']
    )
    r2_dropoffs = r2_score(
        forecast_results_dropoffs['actual'],
        forecast_results_dropoffs['predicted']
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pickups
    axes[0].scatter(forecast_results_pickups['actual'], 
                    forecast_results_pickups['predicted'],
                    alpha=0.3, s=10)
    axes[0].plot([0, forecast_results_pickups['actual'].max()],
                 [0, forecast_results_pickups['actual'].max()],
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Pickups', fontsize=12)
    axes[0].set_ylabel('Predicted Pickups', fontsize=12)
    axes[0].set_title(f'{model_name} - Pickups: Actual vs Predicted\nR² = {r2_pickups:.4f}', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dropoffs
    axes[1].scatter(forecast_results_dropoffs['actual'], 
                    forecast_results_dropoffs['predicted'],
                    alpha=0.3, s=10)
    axes[1].plot([0, forecast_results_dropoffs['actual'].max()],
                 [0, forecast_results_dropoffs['actual'].max()],
                 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Dropoffs', fontsize=12)
    axes[1].set_ylabel('Predicted Dropoffs', fontsize=12)
    axes[1].set_title(f'{model_name} - Dropoffs: Actual vs Predicted\nR² = {r2_dropoffs:.4f}', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_residuals_distribution(forecast_results_pickups, forecast_results_dropoffs, model_name="Model"):
    """
    Create residuals distribution plots (histogram and Q-Q plot) for pickups and dropoffs.
    
    Parameters:
    -----------
    forecast_results_pickups : DataFrame
        DataFrame with 'actual' and 'predicted' columns for pickups
    forecast_results_dropoffs : DataFrame
        DataFrame with 'actual' and 'predicted' columns for dropoffs
    model_name : str
        Name of the model for display purposes
    """
    # Calculate residuals
    residuals_pickups = (forecast_results_pickups['actual'] - 
                         forecast_results_pickups['predicted'])
    residuals_dropoffs = (forecast_results_dropoffs['actual'] - 
                          forecast_results_dropoffs['predicted'])
    
    bias_pickups = residuals_pickups.mean()
    std_residuals_pickups = residuals_pickups.std()
    bias_dropoffs = residuals_dropoffs.mean()
    std_residuals_dropoffs = residuals_dropoffs.std()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Pickups - Residuals Histogram
    axes[0, 0].hist(residuals_pickups, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Residuals (Actual - Predicted)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title(f'{model_name} - Pickups: Residuals Distribution\nMean = {bias_pickups:.2f}, Std = {std_residuals_pickups:.2f}', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pickups - Q-Q Plot
    stats.probplot(residuals_pickups, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title(f'{model_name} - Pickups: Q-Q Plot (Normality Check)', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Dropoffs - Residuals Histogram
    axes[1, 0].hist(residuals_dropoffs, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Residuals (Actual - Predicted)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title(f'{model_name} - Dropoffs: Residuals Distribution\nMean = {bias_dropoffs:.2f}, Std = {std_residuals_dropoffs:.2f}', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Dropoffs - Q-Q Plot
    stats.probplot(residuals_dropoffs, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f'{model_name} - Dropoffs: Q-Q Plot (Normality Check)', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def evaluate_model_complete(forecast_results_pickups, forecast_results_dropoffs, model_name="Model"):
    """
    Complete evaluation: metrics and visualizations for pickups and dropoffs.
    Calls all evaluation and visualization functions.
    
    Parameters:
    -----------
    forecast_results_pickups : DataFrame
        DataFrame with 'actual' and 'predicted' columns for pickups
    forecast_results_dropoffs : DataFrame
        DataFrame with 'actual' and 'predicted' columns for dropoffs
    model_name : str
        Name of the model for display purposes
    """
    # Print basic metrics
    # evaluate_basic_metrics(forecast_results_pickups, forecast_results_dropoffs, model_name)
    
    # Print extended metrics
    evaluate_extended_metrics(forecast_results_pickups, forecast_results_dropoffs, model_name)
    
    # Show visualizations
    plot_actual_vs_predicted(forecast_results_pickups, forecast_results_dropoffs, model_name)
    plot_residuals_distribution(forecast_results_pickups, forecast_results_dropoffs, model_name)


def calculate_metrics_dict(forecast_results_pickups, forecast_results_dropoffs):
    """
    Calculate all metrics and return as a dictionary.
    
    Parameters:
    -----------
    forecast_results_pickups : DataFrame
        DataFrame with 'actual' and 'predicted' columns for pickups
    forecast_results_dropoffs : DataFrame
        DataFrame with 'actual' and 'predicted' columns for dropoffs
    
    Returns:
    --------
    dict: Dictionary with metrics for both pickups and dropoffs
    """
    metrics = {}
    
    # Pickups metrics
    metrics['pickups_mae'] = mean_absolute_error(
        forecast_results_pickups['actual'],
        forecast_results_pickups['predicted']
    )
    metrics['pickups_rmse'] = np.sqrt(mean_squared_error(
        forecast_results_pickups['actual'],
        forecast_results_pickups['predicted']
    ))
    metrics['pickups_r2'] = r2_score(
        forecast_results_pickups['actual'],
        forecast_results_pickups['predicted']
    )
    metrics['pickups_smape'] = smape(
        forecast_results_pickups['actual'],
        forecast_results_pickups['predicted']
    )
    
    # Dropoffs metrics
    metrics['dropoffs_mae'] = mean_absolute_error(
        forecast_results_dropoffs['actual'],
        forecast_results_dropoffs['predicted']
    )
    metrics['dropoffs_rmse'] = np.sqrt(mean_squared_error(
        forecast_results_dropoffs['actual'],
        forecast_results_dropoffs['predicted']
    ))
    metrics['dropoffs_r2'] = r2_score(
        forecast_results_dropoffs['actual'],
        forecast_results_dropoffs['predicted']
    )
    metrics['dropoffs_smape'] = smape(
        forecast_results_dropoffs['actual'],
        forecast_results_dropoffs['predicted']
    )
    
    return metrics


def compare_models(models_dict):
    """
    Compare multiple models and display results in a table with visualizations.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary where keys are model names and values are tuples of
        (forecast_results_pickups, forecast_results_dropoffs)
        
        Example:
        {
            'Baseline': (pickups_df, dropoffs_df),
            'SARIMAX': (pickups_df, dropoffs_df),
            'Prophet': (pickups_df, dropoffs_df)
        }
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    # Calculate metrics for all models
    all_metrics = {}
    for model_name, (pickups_df, dropoffs_df) in models_dict.items():
        all_metrics[model_name] = calculate_metrics_dict(pickups_df, dropoffs_df)
    
    # Create comparison DataFrames
    pickups_comparison = pd.DataFrame({
        model: {
            'MAE': metrics['pickups_mae'],
            'RMSE': metrics['pickups_rmse'],
            'R²': metrics['pickups_r2'],
            'sMAPE (%)': metrics['pickups_smape']
        }
        for model, metrics in all_metrics.items()
    }).T
    
    dropoffs_comparison = pd.DataFrame({
        model: {
            'MAE': metrics['dropoffs_mae'],
            'RMSE': metrics['dropoffs_rmse'],
            'R²': metrics['dropoffs_r2'],
            'sMAPE (%)': metrics['dropoffs_smape']
        }
        for model, metrics in all_metrics.items()
    }).T
    
    # Print comparison tables
    print("\n" + "-"*70)
    print("PICKUPS - MODEL COMPARISON")
    print("-"*70)
    print(pickups_comparison.to_string())
    
    # Highlight best model for each metric
    print("\nBest Performance (Pickups):")
    print(f"  Lowest MAE:   {pickups_comparison['MAE'].idxmin()} ({pickups_comparison['MAE'].min():.2f})")
    print(f"  Lowest RMSE:  {pickups_comparison['RMSE'].idxmin()} ({pickups_comparison['RMSE'].min():.2f})")
    print(f"  Highest R²:   {pickups_comparison['R²'].idxmax()} ({pickups_comparison['R²'].max():.4f})")
    print(f"  Lowest sMAPE: {pickups_comparison['sMAPE (%)'].idxmin()} ({pickups_comparison['sMAPE (%)'].min():.2f}%)")
    
    print("\n" + "-"*70)
    print("DROPOFFS - MODEL COMPARISON")
    print("-"*70)
    print(dropoffs_comparison.to_string())
    
    # Highlight best model for each metric
    print("\nBest Performance (Dropoffs):")
    print(f"  Lowest MAE:   {dropoffs_comparison['MAE'].idxmin()} ({dropoffs_comparison['MAE'].min():.2f})")
    print(f"  Lowest RMSE:  {dropoffs_comparison['RMSE'].idxmin()} ({dropoffs_comparison['RMSE'].min():.2f})")
    print(f"  Highest R²:   {dropoffs_comparison['R²'].idxmax()} ({dropoffs_comparison['R²'].max():.4f})")
    print(f"  Lowest sMAPE: {dropoffs_comparison['sMAPE (%)'].idxmin()} ({dropoffs_comparison['sMAPE (%)'].min():.2f}%)")
    
    print("\n" + "="*70)
    
    # Visualization 1: Bar charts comparing MAE and RMSE
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Pickups MAE
    pickups_comparison['MAE'].plot(kind='bar', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Pickups - MAE Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('MAE', fontsize=12)
    axes[0, 0].set_xlabel('Model', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Pickups R²
    pickups_comparison['R²'].plot(kind='bar', ax=axes[0, 1], color='seagreen')
    axes[0, 1].set_title('Pickups - R² Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('R²', fontsize=12)
    axes[0, 1].set_xlabel('Model', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim([0, 1])
    
    # Dropoffs MAE
    dropoffs_comparison['MAE'].plot(kind='bar', ax=axes[1, 0], color='coral')
    axes[1, 0].set_title('Dropoffs - MAE Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('MAE', fontsize=12)
    axes[1, 0].set_xlabel('Model', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Dropoffs R²
    dropoffs_comparison['R²'].plot(kind='bar', ax=axes[1, 1], color='mediumpurple')
    axes[1, 1].set_title('Dropoffs - R² Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('R²', fontsize=12)
    axes[1, 1].set_xlabel('Model', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()
    
    # Visualization 2: Grouped bar chart for all metrics
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Normalize metrics for better comparison (except R² which is already 0-1)
    pickups_normalized = pickups_comparison.copy()
    pickups_normalized['MAE'] = pickups_normalized['MAE'] / pickups_normalized['MAE'].max()
    pickups_normalized['RMSE'] = pickups_normalized['RMSE'] / pickups_normalized['RMSE'].max()
    pickups_normalized['sMAPE (%)'] = pickups_normalized['sMAPE (%)'] / pickups_normalized['sMAPE (%)'].max()
    
    dropoffs_normalized = dropoffs_comparison.copy()
    dropoffs_normalized['MAE'] = dropoffs_normalized['MAE'] / dropoffs_normalized['MAE'].max()
    dropoffs_normalized['RMSE'] = dropoffs_normalized['RMSE'] / dropoffs_normalized['RMSE'].max()
    dropoffs_normalized['sMAPE (%)'] = dropoffs_normalized['sMAPE (%)'] / dropoffs_normalized['sMAPE (%)'].max()
    
    # Pickups - normalized metrics
    pickups_normalized[['MAE', 'RMSE', 'R²', 'sMAPE (%)']].plot(kind='bar', ax=axes[0])
    axes[0].set_title('Pickups - Normalized Metrics Comparison\n(Lower is better, except R²)', 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Normalized Score (0-1)', fontsize=12)
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].legend(title='Metric', loc='best')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_ylim([0, 1.1])
    
    # Dropoffs - normalized metrics
    dropoffs_normalized[['MAE', 'RMSE', 'R²', 'sMAPE (%)']].plot(kind='bar', ax=axes[1])
    axes[1].set_title('Dropoffs - Normalized Metrics Comparison\n(Lower is better, except R²)', 
                     fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Normalized Score (0-1)', fontsize=12)
    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].legend(title='Metric', loc='best')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.show()
