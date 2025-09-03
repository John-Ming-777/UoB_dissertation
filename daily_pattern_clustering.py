import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# --- Configuration ---
DATA_PATH = "/Users/ming/Library/CloudStorage/OneDrive-UniversityofBristol/bristol 毕设/original_data.csv"
OUTPUT_DIR = "/Users/ming/Library/CloudStorage/OneDrive-UniversityofBristol/bristol 毕设/GoGreen_Analysis/Clustering_Results"
ASSET_TO_ANALYZE = 'indoor_lights_office'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def perform_daily_pattern_clustering(data_path, output_dir, asset_name):

    # --- 1. Daily Profiling - Data Preparation ---
    df = pd.read_csv(data_path)
    df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])


    # Filter for the specific asset
    asset_df = df[['timestamp', asset_name]].copy()
    asset_df.set_index('timestamp', inplace=True)
    asset_df.columns = ['energy_kwh']

    # Resample to 30-minute intervals to ensure 48 points per day
    # Fill any missing 30-min intervals with 0 (assuming no consumption if missing)
    asset_df = asset_df.resample('30min').mean().fillna(0)

    # Create daily profiles (N x 48 matrix)
    daily_profiles = []
    dates = []
    for date, group in asset_df.groupby(asset_df.index.date):
        if len(group) == 48: # Ensure a full day of data
            daily_profiles.append(group['energy_kwh'].values)
            dates.append(date)


    daily_profiles_matrix = np.array(daily_profiles)
    
    # --- 2. Feature Scaling - Standardization ---
    scaler = StandardScaler()
    scaled_daily_profiles = scaler.fit_transform(daily_profiles_matrix)

    # --- 3. Finding Optimal K - Elbow Method ---
    wcss = []
    max_k = 10


    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(scaled_daily_profiles)
        wcss.append(kmeans.inertia_)

    # Plotting the Elbow Method results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), wcss, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS')
    elbow_plot_path = os.path.join(output_dir, f'{asset_name}_elbow_method.png')
    plt.savefig(elbow_plot_path)
    plt.close()


    optimal_k = 4 # Based on Elbow Method analysis

    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_daily_profiles)

    # Add cluster labels and original dates to a DataFrame
    clustered_df = pd.DataFrame({'date': dates, 'cluster': clusters})

    # Calculate and plot cluster centroids
    plt.figure(figsize=(12, 8))
    for i in range(optimal_k):
        cluster_profiles = scaled_daily_profiles[clusters == i]
        centroid = np.mean(cluster_profiles, axis=0)
        plt.plot(centroid, label=f'Cluster {i} (N={len(cluster_profiles)})')

    plt.title(f'Cluster Centroids for {asset_name}')
    plt.xlabel('Time Interval (0-47)')
    plt.ylabel('Scaled Energy Consumption')
    plt.legend()
    centroids_plot_path = os.path.join(output_dir, f'{asset_name}_cluster_centroids.png')
    plt.savefig(centroids_plot_path)
    plt.close()
    print(f"Cluster centroids plot saved to {centroids_plot_path}")

    # Interpret clusters based on date distribution
    print("--- Cluster Interpretation ---")
    for i in range(optimal_k):
        cluster_dates = clustered_df[clustered_df['cluster'] == i]['date']
        day_names = pd.to_datetime(cluster_dates).dt.day_name()
        day_counts = day_names.value_counts().sort_index()
        
        print(f"\nCluster {i} (N={len(cluster_dates)} days):")
        print(f"Dates: {cluster_dates.tolist()}")
        print(day_counts)


if __name__ == "__main__":
    perform_daily_pattern_clustering(DATA_PATH, OUTPUT_DIR, ASSET_TO_ANALYZE)
