import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
SCALER_PATH = 'mlp_scaler.pkl'  # Path to your saved scaler
# The 10 channels we used (matches your 11-input CSV logic)
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz", "qw", "qx", "qy", "qz"]
# The 5 stats we calculated per channel
STATS = ["mean", "std", "min", "max", "msq"]


def inspect_scaler(path):
    try:
        # 1. Load the scaler
        scaler = joblib.load(path)

        # Generate the 50 feature names (10 channels * 5 stats)
        feature_names = [f"{c}_{s}" for c in CHANNELS for s in STATS]

        # 2. Extract metadata
        # .mean_ is the average value for each feature across your training set
        # .scale_ is the standard deviation (spread) for each feature
        means = scaler.mean_
        stds = scaler.scale_

        # 3. Create a summary DataFrame
        df_stats = pd.DataFrame({
            'Feature': feature_names,
            'Average_Value': means,
            'Standard_Deviation': stds
        })

        print(f"--- Scaler Inspection: {path} ---")
        print(df_stats.head(10))  # Print first 10 for a quick look

        # 4. Visualization
        plt.figure(figsize=(12, 8))

        # We'll plot the Standard Deviation per feature
        # High deviation usually means that feature is very dynamic during movements
        plt.subplot(2, 1, 1)
        plt.bar(range(len(stds)), stds, color='skyblue')
        plt.title("Feature Variance (Standard Deviation) stored in Scaler")
        plt.ylabel("Sigma (Spread)")
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # Plot the Mean per feature
        plt.subplot(2, 1, 2)
        plt.bar(range(len(means)), means, color='salmon')
        plt.title("Feature Centroids (Means) stored in Scaler")
        plt.ylabel("Average Value")
        plt.xticks(range(0, 50, 5), CHANNELS, rotation=45)  # Label by sensor groups
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

        return df_stats

    except FileNotFoundError:
        print(f"Error: {path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    scaler_df = inspect_scaler(SCALER_PATH)