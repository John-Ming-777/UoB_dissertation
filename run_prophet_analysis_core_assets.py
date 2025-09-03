
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Define paths
data_file_path = "/Users/ming/Library/CloudStorage/OneDrive-UniversityofBristol/bristol 毕设/original_data.csv"
output_base_dir = "/Users/ming/Library/CloudStorage/OneDrive-UniversityofBristol/bristol 毕设/GoGreen_Analysis"
prophet_output_dir = os.path.join(output_base_dir, "prophet_results")

# Create directories (if they don't exist, though they should from previous runs)
os.makedirs(prophet_output_dir, exist_ok=True)
csv_output_dir = os.path.join(prophet_output_dir, "csv_results")
os.makedirs(csv_output_dir, exist_ok=True)
plots_output_dir = os.path.join(prophet_output_dir, "plots")
os.makedirs(plots_output_dir, exist_ok=True)

# Load and prepare data
df = pd.read_csv(data_file_path, parse_dates=['Timestamps (UTC)'])
df = df.rename(columns={'Timestamps (UTC)': 'ds'})
if 'check' in df.columns:
    df = df.drop(columns=['check'])

# Core assets for analysis
core_assets = [
    'Aggregate load (kWh)',
    'Lift1',
    'Lift2_3',
    'Lift4_5',
    'outdoor_lights_platform_1_lighting_pole_single',
    'outdoor_lights_platform_2_lighting_pole_single',
    'outdoor_lights_platform_3_lighting_pole_single',
    'outdoor_lights_platform_4_5_lighting_pole_single'
]

print(f"Starting Prophet analysis for {len(core_assets)} core assets.")

# List to store performance metrics
performance_metrics = []

# Run Prophet for each core asset
for asset in core_assets:
    print(f"Running Prophet analysis for asset: {asset}")

    asset_df = df[['ds', asset]].copy()
    asset_df = asset_df.rename(columns={asset: 'y'})
    asset_df.dropna(subset=['y'], inplace=True)

    if asset_df.empty or len(asset_df) < 2:
        print(f"Skipping {asset} due to insufficient data.")
        continue

    # Initialize and fit the model
    model = Prophet(
        seasonality_mode='multiplicative',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    model.fit(asset_df)

    # Create future dataframe and predict
    future = model.make_future_dataframe(periods=0, freq='30min')
    forecast = model.predict(future)

    # Save forecast and components
    forecast_path = os.path.join(csv_output_dir, f'{asset}_prophet_forecast.csv')
    forecast.to_csv(forecast_path, index=False)

    components_fig = model.plot_components(forecast)
    components_path = os.path.join(plots_output_dir, f'{asset}_prophet_components.png')
    components_fig.savefig(components_path)
    plt.close(components_fig)

    # Merge forecast with actuals and identify anomalies
    results_df = pd.merge(asset_df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    results_df['anomaly'] = (results_df['y'] < results_df['yhat_lower']) | (results_df['y'] > results_df['yhat_upper'])

    # Save anomalies to a separate CSV for each asset
    asset_anomalies = results_df[results_df['anomaly'] == True]
    if not asset_anomalies.empty:
        asset_anomaly_path = os.path.join(csv_output_dir, f'{asset}_anomalies.csv')
        asset_anomalies.to_csv(asset_anomaly_path, index=False)
    
    # Calculate residuals
    results_df['residuals'] = results_df['y'] - results_df['yhat']

    # Save residuals data
    residuals_path = os.path.join(csv_output_dir, f'{asset}_prophet_residuals.csv')
    results_df[['ds', 'y', 'yhat', 'residuals']].to_csv(residuals_path, index=False)

    # Plot residuals distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['residuals'], bins=50, edgecolor='black')
    plt.title(f'Residuals Distribution for {asset}')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    residuals_plot_path = os.path.join(plots_output_dir, f'{asset}_residuals_distribution.png')
    plt.savefig(residuals_plot_path)
    plt.close()

    # Calculate MSE and MAE
    mse = mean_squared_error(results_df['y'], results_df['yhat'])
    mae = mean_absolute_error(results_df['y'], results_df['yhat'])
    
    print(f"  MSE: {mse:.4f}, MAE: {mae:.4f}")
    performance_metrics.append({'Asset': asset, 'MSE': mse, 'MAE': mae})

    # Plot a specific day
    if not results_df.empty:
        plot_day = results_df['ds'].max().date()
        day_df = results_df[results_df['ds'].dt.date == plot_day]

        if not day_df.empty:
            plt.figure(figsize=(15, 8))
            plt.plot(day_df['ds'], day_df['y'], 'ko', markersize=3, label='Actual')
            plt.plot(day_df['ds'], day_df['yhat'], 'b-', label='Forecast')
            plt.fill_between(day_df['ds'].dt.to_pydatetime(), day_df['yhat_lower'], day_df['yhat_upper'], color='blue', alpha=0.2, label='Confidence Interval')
            
            anomalies = day_df[day_df['anomaly']]
            plt.plot(anomalies['ds'], anomalies['y'], 'rX', markersize=8, label='Anomaly')

            plt.title(f'Forecast vs. Actuals for {asset} on {plot_day}')
            plt.xlabel('Time')
            plt.ylabel('Energy Consumption (kWh)')
            plt.legend()
            plt.grid(True)
            
            plot_path = os.path.join(plots_output_dir, f'{asset}_{plot_day}_forecast_vs_actual.png')
            plt.savefig(plot_path)
            plt.close()

# Save all performance metrics to a CSV
metrics_df = pd.DataFrame(performance_metrics)
metrics_csv_path = os.path.join(prophet_output_dir, "prophet_model_performance_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"\nAll model performance metrics saved to: {metrics_csv_path}")

print("Prophet analysis for core assets complete. Results are in GoGreen_Analysis/prophet_results.")
