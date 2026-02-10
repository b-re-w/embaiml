import serial
import numpy as np
import collections
import joblib
import time

# --- 1. LOAD MODEL ---
gb_model = joblib.load('gb_imu_model.pkl')

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
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


# --- 2. LIVE INFERENCE LOOP ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    print("Serial connected. Running Gradient Boosting Inference...")
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

                # Feature extraction
                features = extract_features(data_window)

                # Predict (No scaling required)
                prediction = gb_model.predict(features)[0]
                probs = gb_model.predict_proba(features)[0]
                confidence = np.max(probs)

                print(f"[{time.strftime('%H:%M:%S')}] Activity: {prediction.upper()} | Conf: {confidence:.2%}")

        except Exception:
            continue
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    if 'ser' in locals(): ser.close()