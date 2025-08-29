import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric, plot_forecast_component
import matplotlib.pyplot as plt
import os
import shutil
import sys

# --- Configuration ---
data_file_path = "/Users/ming/Library/CloudStorage/OneDrive-UniversityofBristol/bristol 毕设/original_data.csv"
output_base_dir = "/Users/ming/Library/CloudStorage/OneDrive-UniversityofBristol/bristol 毕设/GoGreen_Analysis/prophet_results"

# CV Parameters
CV_INITIAL = '180 days'
CV_PERIOD = '1 days'
CV_HORIZON = '1 days'

ASSETS_TO_ANALYZE = [
    'Aggregate load (kWh)',
    'Lift1',
    'Lift2_3',
    'Lift4_5',
    'outdoor_lights_platform_1_lighting_pole_single',
    'outdoor_lights_platform_2_lighting_pole_single',
    'outdoor_lights_platform_3_lighting_pole_single',
    'outdoor_lights_platform_4_5_lighting_pole_single'
]

def plot_specific_day_with_anomalies(df_cv, day_str, asset_name, save_path):
    """Plots the actual vs. forecast for a specific day from CV results and highlights anomalies."""
    day_df = df_cv[df_cv['ds'].dt.strftime('%Y-%m-%d') == day_str].copy()
    if day_df.empty:
        print(f"No data found for day: {day_str} for asset {asset_name}")
        return

    day_df['anomaly'] = (day_df['y'] > day_df['yhat_upper']) | (day_df['y'] < day_df['yhat_lower'])
    anomalies = day_df[day_df['anomaly']]

    plt.figure(figsize=(15, 7))
    plt.plot(day_df['ds'], day_df['y'], 'k', label='Actual Value')
    plt.plot(day_df['ds'], day_df['yhat'], 'b--', label='Forecast (yhat)')
    plt.fill_between(day_df['ds'], day_df['yhat_lower'], day_df['yhat_upper'], 
                     color='blue', alpha=0.2, label='Confidence Interval')
    plt.scatter(anomalies['ds'], anomalies['y'], color='red', s=50, zorder=5, label='Anomaly')

    plt.title(f'Daily Anomaly Detection for {asset_name} on {day_str}')
    plt.xlabel('Timestamp')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved specific day plot with anomalies to {save_path}")

def analyze_asset(asset_name, df):
    """Runs the full Prophet analysis for a single asset."""
    sanitized_asset_name = asset_name.replace(' (', '_').replace(')', '').replace(' ', '_')
    asset_results_dir = os.path.join(output_base_dir, sanitized_asset_name)
    
    if not os.path.exists(asset_results_dir):
        os.makedirs(asset_results_dir)

    cv_results_path = os.path.join(asset_results_dir, f'{sanitized_asset_name}_cv_results.csv')
    if not os.path.exists(cv_results_path):
        print(f"Cross-validation results not found for {asset_name}. Please run the full analysis first.")
        return

    print(f"--- Generating anomaly plot for: {asset_name} ---")
    df_cv = pd.read_csv(cv_results_path, parse_dates=['ds', 'cutoff'])
    
    specific_day_plot_path = os.path.join(asset_results_dir, f'{sanitized_asset_name}_specific_day_anomalies.png')
    plot_specific_day_with_anomalies(df_cv, '2025-02-26', asset_name, specific_day_plot_path)
    print(f"--- Finished plot generation for: {asset_name} ---\n")

def main():
    """Main function to run the analysis for all specified assets."""
    try:
        df = pd.read_csv(data_file_path, parse_dates=['Timestamps (UTC)'])
        df = df.rename(columns={'Timestamps (UTC)': 'ds'})
        if 'check' in df.columns:
            df = df.drop(columns=['check'])
    except Exception as e:
        print(f"FATAL: Could not load or prepare the data file: {e}")
        sys.exit(1)

    for asset in ASSETS_TO_ANALYZE:
        analyze_asset(asset, df)

    print("All analyses completed.")

if __name__ == '__main__':
    main()