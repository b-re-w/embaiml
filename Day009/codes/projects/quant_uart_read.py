import serial
import pandas as pd
import matplotlib.pyplot as plt
import time

# --- Settings ---
PORT = '/dev/cu.wchusbserial537E0140361'  # Change to your M5Core2 port
BAUD = 115200
OUT_FILE = "quaternion_log.csv"
RECORD_SECONDS = 30

data_list = []

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    ser.reset_input_buffer()  # Clear startup noise
    print(f"Recording {RECORD_SECONDS} seconds of data...")

    start_time = time.time()
    while (time.time() - start_time) < RECORD_SECONDS:
        if ser.in_waiting > 0:
            try:
                # errors='ignore' prevents crashes on invalid startup bytes
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                parts = line.split(',')

                # We expect 5 values: ms, qw, qx, qy, qz
                if len(parts) == 5:
                    data_list.append(parts)
            except Exception as e:
                continue

    ser.close()
except Exception as e:
    print(f"Serial Error: {e}")

# Process and Plot
if data_list:
    df = pd.DataFrame(data_list, columns=['ms', 'qw', 'qx', 'qy', 'qz'])
    df = df.apply(pd.to_numeric)
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved to {OUT_FILE}")

    plt.figure(figsize=(12, 6))
    plt.plot(df['ms'], df['qw'], label='qw (Scalar)', color='black', linewidth=2)
    plt.plot(df['ms'], df['qx'], label='qx (i)', color='red')
    plt.plot(df['ms'], df['qy'], label='qy (j)', color='green')
    plt.plot(df['ms'], df['qz'], label='qz (k)', color='blue')

    plt.title("M5Core2 Manual Quaternion Orientation (30s Capture)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Component Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print("No valid data captured.")