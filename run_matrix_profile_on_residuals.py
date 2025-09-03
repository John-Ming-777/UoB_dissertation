
import pandas as pd
import numpy as np
import stumpy
import matplotlib.pyplot as plt
import os
import glob

BASE_DIR = '/Users/ming/Library/CloudStorage/OneDrive-UniversityofBristol/bristol 毕设/GoGreen_Analysis'
RESIDUALS_DIR = os.path.join(BASE_DIR, 'prophet_results', 'csv_results')
MP_OUTPUT_DIR = os.path.join(BASE_DIR, 'residual_matrix_profile_results')

# Matrix Profile window size (48 * 30 minutes = 24 hours)
MP_WINDOW_SIZE = 48

def analyze_residual_file(file_path):

    df = pd.read_csv(file_path)
    asset_name = os.path.basename(file_path).replace('_prophet_residuals.csv', '')

    # Prepare the time series (residuals)
    T = df['residuals'].values.astype(float)

    # Ensure the series is long enough for Matrix Profile
    if len(T) < MP_WINDOW_SIZE * 2:
        print(f"Skipping {asset_name} as series length ({len(T)}) is too short for the given window size.")
        return

    # --- Core Matrix Profile Calculation ---
    mp = stumpy.stump(T, m=MP_WINDOW_SIZE)

    # --- Discover Top 10 Discords Manually ---
    top_n_discords = 10
    discords = []
    # Create a copy of the matrix profile to modify
    mp_copy = mp[:, 0].copy()
    
    for _ in range(top_n_discords):
        discord_idx = np.argmax(mp_copy)
        nn_dist = mp_copy[discord_idx]
        
        # Stop if we've exhausted all meaningful discords
        if np.isinf(nn_dist):
            break
            
        discords.append((discord_idx, nn_dist))
        
        # Apply an exclusion zone around the found discord
        exclusion_start = max(0, discord_idx - MP_WINDOW_SIZE)
        exclusion_end = min(len(mp_copy), discord_idx + MP_WINDOW_SIZE)
        mp_copy[exclusion_start:exclusion_end] = -np.inf

    # --- Create asset-specific output directory ---
    asset_output_dir = os.path.join(MP_OUTPUT_DIR, asset_name)
    os.makedirs(asset_output_dir, exist_ok=True)

    discord_details = []

    # --- Plot and save each discord individually ---
    for i, (idx, nn_dist) in enumerate(discords):
        rank = i + 1
        
        # Plotting the specific discord subsequence
        fig, ax = plt.subplots(figsize=(10, 4))
        discord_subsequence = T[idx : idx + MP_WINDOW_SIZE]
        ax.plot(range(MP_WINDOW_SIZE), discord_subsequence, color='red')
        ax.set_title(f'Asset: {asset_name} - Discord Rank: {rank}')
        ax.set_xlabel('Time within Subsequence (30-min intervals)')
        ax.set_ylabel('Residual Value')
        plt.tight_layout()
        
        # Saving the plot
        discord_plot_path = os.path.join(asset_output_dir, f'{asset_name}_discord_rank_{rank}.png')
        plt.savefig(discord_plot_path)
        plt.close(fig)

        # Store discord info for CSV
        discord_details.append({
            'asset': asset_name,
            'rank': rank,
            'start_index': idx,
            'end_index': idx + MP_WINDOW_SIZE,
            'nearest_neighbor_distance': nn_dist,
            'start_timestamp': df.loc[idx, 'ds']
        })

    # --- Save discord details to CSV ---
    if discord_details:
        discord_df = pd.DataFrame(discord_details)
        csv_path = os.path.join(asset_output_dir, f'{asset_name}_top_{top_n_discords}_discords.csv')
        discord_df.to_csv(csv_path, index=False)
        print(f"  Saved discord details to: {csv_path}")

if __name__ == '__main__':

    residual_files = glob.glob(os.path.join(RESIDUALS_DIR, '*_prophet_residuals.csv'))
    
    for f in residual_files:
        analyze_residual_file(f)
        
