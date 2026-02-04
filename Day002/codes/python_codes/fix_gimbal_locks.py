import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def fix_and_plot(file_path):
    df = pd.read_csv(file_path)

    # Simple 'fix' strategy: Apply a median filter to the Euler angles
    # to remove the 'spikes' caused by gimbal lock transitions
    df_fixed = df.copy()
    df_fixed['pitch'] = df['pitch'].rolling(window=5, center=True).median().fillna(method='bfill')
    df_fixed['roll'] = df['roll'].rolling(window=5, center=True).median().fillna(method='bfill')

    # Save corrected data
    df_fixed.to_csv("imu_data_fixed.csv", index=False)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Original Waveform
    ax1.plot(df['Timestamp_ms'], df['pitch'], label='Original Pitch', color='red', alpha=0.6)
    ax1.plot(df['Timestamp_ms'], df['roll'], label='Original Roll', color='blue', alpha=0.6)
    ax1.set_title("Original Waveform (With Gimbal Lock Spikes)")
    ax1.legend()

    # Corrected Waveform
    ax2.plot(df_fixed['Timestamp_ms'], df_fixed['pitch'], label='Fixed Pitch', color='red')
    ax2.plot(df_fixed['Timestamp_ms'], df_fixed['roll'], label='Fixed Roll', color='blue')
    ax2.set_title("Corrected Waveform (Filtered)")
    ax2.set_xlabel("Time (ms)")
    ax2.legend()

    plt.tight_layout()
    plt.show()


fix_and_plot("imu_data_log.csv")