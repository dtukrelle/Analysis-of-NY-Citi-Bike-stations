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
    mean_absolute_percentage_error,
    median_absolute_error
)
from scipy import stats


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
    mape_pickups = mean_absolute_percentage_error(
        forecast_results_pickups['actual'],
        forecast_results_pickups['predicted']
    ) * 100
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
    print(f"  MAPE:          {mape_pickups:.2f}%")
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
    mape_dropoffs = mean_absolute_percentage_error(
        forecast_results_dropoffs['actual'],
        forecast_results_dropoffs['predicted']
    ) * 100
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
    print(f"  MAPE:          {mape_dropoffs:.2f}%")
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
    evaluate_basic_metrics(forecast_results_pickups, forecast_results_dropoffs, model_name)
    
    # Print extended metrics
    evaluate_extended_metrics(forecast_results_pickups, forecast_results_dropoffs, model_name)
    
    # Show visualizations
    plot_actual_vs_predicted(forecast_results_pickups, forecast_results_dropoffs, model_name)
    plot_residuals_distribution(forecast_results_pickups, forecast_results_dropoffs, model_name)
