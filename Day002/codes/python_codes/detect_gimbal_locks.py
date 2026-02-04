import pandas as pd


def detect_gimbal_lock(file_path):
    df = pd.read_csv(file_path)
    # Pitch is typically index 8 or column 'pitch'
    # Gimbal lock occurs at +/- 90 degrees
    threshold = 0.5
    locks = df[(df['pitch'] >= 90 - threshold) | (df['pitch'] <= -90 + threshold)]

    print(f"Total Gimbal Lock Occurrences: {len(locks)}")
    if len(locks) > 0:
        print("Timestamps (ms) of occurrence:")
        print(locks['Timestamp_ms'].values)

    return locks


detect_gimbal_lock("imu_data_recalculated.csv")