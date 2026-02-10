import serial
import csv
import pandas as pd
import matplotlib.pyplot as plt
import time

# --- Settings ---
PORT = '/dev/tty.wchusbserial53810043161'  # Update to your port
BAUD = 115200
OUT_FILE = "quaternion_log.csv"
RECORD_SECONDS = 10

data_list = []

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print(f"Recording {RECORD_SECONDS} seconds of data...")

    start_time = time.time()
    while (time.time() - start_time) < RECORD_SECONDS:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            parts = line.split(',')
            if len(parts) == 11:
                data_list.append(parts)

    ser.close()
except Exception as e:
    print(f"Error: {e}")

# Save to CSV
df = pd.DataFrame(data_list, columns=['ms', 'ax', 'ay', 'az', 'gz', 'gy', 'gz', 'qw', 'qx', 'qy', 'qz'])
df = df.apply(pd.to_numeric)
df.to_csv(OUT_FILE, index=False)
print(f"Saved to {OUT_FILE}")

# Plotting Results
plt.figure(figsize=(12, 6))
plt.plot(df['ms'], df['qw'], label='qw (Scalar)', color='black', linewidth=2)
plt.plot(df['ms'], df['qx'], label='qx', color='red')
plt.plot(df['ms'], df['qy'], label='qy', color='green')
plt.plot(df['ms'], df['qz'], label='qz', color='blue')

plt.title("IMU Quaternion Components (Gimbal Lock Free)")
plt.xlabel("Time (ms)")
plt.ylabel("Value (-1.0 to 1.0)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()