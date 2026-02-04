import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick


def recalculate_without_lock(file_path):
    # 1. Load the data
    df = pd.read_csv(file_path)

    # Extract raw sensors (Madgwick expects Gyro in rad/s and Accel in m/s^2)
    # Most M5Core2 setups output Gyro in deg/s and Accel in Gs
    accel_data = df[['ax', 'ay', 'az']].values * 9.80665  # G to m/s^2
    gyro_data = np.deg2rad(df[['gx', 'gy', 'gz']].values)  # deg/s to rad/s

    # 2. Apply Madgwick Filter to generate Quaternions
    # Frequency should match your Arduino delay (20ms = 50Hz)
    madgwick = Madgwick(gyr=gyro_data, acc=accel_data, frequency=50.0)
    q = madgwick.Q  # This is an array of quaternions [qw, qx, qy, qz]

    # 3. Convert Quaternions to RPY (Euler)
    # Manual conversion to ensure we handle the math correctly
    def quat_to_euler(q):
        w, x, y, z = q
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

    rpy_fixed = np.array([quat_to_euler(qi) for qi in q])

    df['fixed_roll'] = rpy_fixed[:, 0]
    df['fixed_pitch'] = rpy_fixed[:, 1]
    df['fixed_yaw'] = rpy_fixed[:, 2]

    # 4. Save and Plot
    df.to_csv("imu_data_recalculated.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(df['Timestamp_ms'], df['pitch'], label='Original Pitch (Gimbal Lock)', alpha=0.5, linestyle='--')
    plt.plot(df['Timestamp_ms'], df['fixed_pitch'], label='Recalculated Pitch (Quat-based)', color='red')
    plt.title("Comparison: Recalculated Orientation via Quaternions")
    plt.xlabel("Time (ms)")
    plt.ylabel("Degrees")
    plt.legend()
    plt.show()


recalculate_without_lock("imu_data_log.csv")