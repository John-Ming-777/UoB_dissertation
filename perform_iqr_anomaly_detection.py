
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def perform_iqr_anomaly_detection(data_path='original_data.csv', output_dir='IQR_results'):
    """
    Performs anomaly detection based on Interquartile Range (IQR) for two profile types: Mon-Sat and Sun.

    Args:
        data_path (str): Path to the input CSV data.
        output_dir (str): Directory to save the results (CSV and plots).
    """
    print("Starting IQR anomaly detection process...")

    # 1. Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory '{output_dir}' created or already exists.")

    # 2. Load and preprocess data
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{data_path}'")
        return

    df.rename(columns={'Timestamps (UTC)': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    df['date'] = df.index.date
    df['time'] = df.index.time
    df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
    df['profile_type'] = df['day_of_week'].apply(lambda x: 'Sun' if x == 6 else 'Mon-Sat')

    asset_columns = [col for col in df.columns if col.startswith('indoor_lights')]
    print(f"Found {len(asset_columns)} assets to analyze.")

    all_anomalies = []

    # 3. Process each asset
    for i, asset in enumerate(asset_columns):
        print(f"Processing asset {i+1}/{len(asset_columns)}: {asset}...")
        
        asset_df = df[[asset, 'profile_type', 'time', 'date']].copy()
        asset_df.rename(columns={asset: 'value'}, inplace=True)

        # 4. Generate Standard Profiles (IQR)
        profiles = asset_df.groupby(['profile_type', 'time'])['value'].agg(
            Q1=lambda x: x.quantile(0.25),
            median=lambda x: x.quantile(0.5),
            Q3=lambda x: x.quantile(0.75)
        ).reset_index()
        
        # Merge profiles back to the main asset dataframe
        merged_df = pd.merge(asset_df, profiles, on=['profile_type', 'time'])

        # 5. Anomaly Detection
        merged_df['is_anomaly'] = (merged_df['value'] < merged_df['Q1']) | (merged_df['value'] > merged_df['Q3'])
        
        anomalies = merged_df[merged_df['is_anomaly']].copy()
        anomalies['asset'] = asset
        
        if not anomalies.empty:
            all_anomalies.append(anomalies[['asset', 'date', 'time', 'value', 'Q1', 'median', 'Q3', 'profile_type']])

        # 6. Generate Comparison Plot
        anomalous_days = merged_df[merged_df['is_anomaly']]['date'].unique()
        
        if len(anomalous_days) > 0:
            # Pick the first anomalous day for plotting
            day_to_plot_anomaly = anomalous_days[0]
            
            # Find a normal day of the same profile type
            anomaly_profile_type = merged_df[merged_df['date'] == day_to_plot_anomaly]['profile_type'].iloc[0]
            
            normal_days = merged_df[~merged_df['is_anomaly']].groupby('date').filter(lambda x: len(x) == 48) # Only full days
            normal_days_same_type = normal_days[normal_days['profile_type'] == anomaly_profile_type]['date'].unique()

            day_to_plot_normal = None
            if len(normal_days_same_type) > 0:
                # Find a normal day close to the anomalous day
                normal_day_candidates = [d for d in normal_days_same_type if d < day_to_plot_anomaly]
                if normal_day_candidates:
                    day_to_plot_normal = max(normal_day_candidates)
                else:
                    day_to_plot_normal = min(normal_days_same_type)


            # Plotting
            fig, ax = plt.subplots(figsize=(15, 7))

            profile_to_plot = profiles[profiles['profile_type'] == anomaly_profile_type]
            
            # Plot IQR band and median
            ax.fill_between(profile_to_plot['time'].astype(str), profile_to_plot['Q1'], profile_to_plot['Q3'], color='gray', alpha=0.3, label=f'Normal Range (IQR) - {anomaly_profile_type}')
            ax.plot(profile_to_plot['time'].astype(str), profile_to_plot['median'], color='gray', linestyle='--', label=f'Median - {anomaly_profile_type}')

            # Plot normal day if found
            if day_to_plot_normal:
                normal_day_data = merged_df[merged_df['date'] == day_to_plot_normal]
                ax.plot(normal_day_data['time'].astype(str), normal_day_data['value'], color='green', alpha=0.8, label=f'Normal Day ({day_to_plot_normal})')

            # Plot anomalous day
            anomalous_day_data = merged_df[merged_df['date'] == day_to_plot_anomaly]
            ax.plot(anomalous_day_data['time'].astype(str), anomalous_day_data['value'], color='orange', label=f'Anomalous Day ({day_to_plot_anomaly})')
            
            # Highlight anomaly points
            anomaly_points = anomalous_day_data[anomalous_day_data['is_anomaly']]
            ax.scatter(anomaly_points['time'].astype(str), anomaly_points['value'], color='red', s=50, zorder=5, label='Anomaly')

            ax.set_title(f'Anomaly Comparison for: {asset}')
            ax.set_xlabel('Time of Day')
            ax.set_ylabel('Energy Consumption (kWh)')
            ax.legend()
            plt.xticks(rotation=90)
            plt.gca().xaxis.set_major_locator(plt.MultipleLocator(4)) # Show every 4th label (every 2 hours)
            plt.tight_layout()
            
            plot_filename = os.path.join(output_dir, f'{asset}_anomaly_comparison.png')
            plt.savefig(plot_filename)
            plt.close(fig)
        else:
            print(f"No anomalies found for asset {asset}. Skipping plot.")

    # 7. Save all anomalies to a single CSV
    if all_anomalies:
        final_anomalies_df = pd.concat(all_anomalies)
        final_anomalies_df.sort_values(by=['asset', 'date', 'time'], inplace=True)
        csv_path = os.path.join(output_dir, 'detected_anomalies.csv')
        final_anomalies_df.to_csv(csv_path, index=False)
        print(f"Successfully saved {len(final_anomalies_df)} anomalies to '{csv_path}'")
    else:
        print("No anomalies were detected across any assets.")

    print("IQR anomaly detection process finished.")

if __name__ == '__main__':
    # Paths are relative to the project root directory where the script is executed from.
    data_file_path = 'original_data.csv'
    output_folder = 'GoGreen_Analysis/IQR_results'
    
    # To run this script, you would place it in the GoGreen_Analysis folder
    # and ensure original_data.csv is in the root of the project directory.
    perform_iqr_anomaly_detection(data_path=data_file_path, output_dir=output_folder)
