import serial
import numpy as np
import collections
import joblib

# --- 1. LOAD MODEL AND SCALER ---
mlp = joblib.load('mlp_model.pkl')
scaler = joblib.load('mlp_scaler.pkl')

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'
WINDOW_SIZE = 100
STRIDE = 50
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz", "qw", "qx", "qy", "qz"]

data_window = collections.deque(maxlen=WINDOW_SIZE)


def extract_features(window_data):
    np_win = np.array(window_data)
    feats = []
    for i in range(len(CHANNELS)):
        x = np_win[:, i]
        feats.extend([x.mean(), x.std(), x.min(), x.max(), np.mean(x ** 2)])
    return np.array(feats).reshape(1, -1)


# --- 2. REAL-TIME LOOP ---
try:
    ser = serial.Serial(SERIAL_PORT, 115200, timeout=0.1)
    sample_count = 0

    while True:
        line = ser.readline().decode('utf-8').strip()
        if not line: continue

        try:
            values = [float(x) for x in line.split(',')]
            if len(values) == 11:
                data_window.append(values[1:])
                sample_count += 1

            if len(data_window) == WINDOW_SIZE and sample_count >= STRIDE:
                sample_count = 0

                # Step A: Extract raw features
                raw_features = extract_features(data_window)

                # Step B: SCALE features using the loaded scaler
                scaled_features = scaler.transform(raw_features)

                # Step C: PREDICT
                prediction = mlp.predict(scaled_features)[0]
                print(f"MLP Activity: {prediction.upper()}")

        except Exception:
            continue
except KeyboardInterrupt:
    print("\nClosing...")
finally:
    if 'ser' in locals(): ser.close()