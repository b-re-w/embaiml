import serial
import numpy as np
import collections
import joblib
import time

# --- 1. LOAD MODEL AND SCALER ---
nn_model = joblib.load('nn_backprop_model.pkl')
scaler = joblib.load('nn_scaler.pkl')

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'  # Change to your port
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
    print("Serial connected. Listening for IMU data...")
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
                raw_feats = extract_features(data_window)

                # Apply the SAME scaling used during training
                scaled_feats = scaler.transform(raw_feats)

                # Predict
                prediction = nn_model.predict(scaled_feats)[0]
                probs = nn_model.predict_proba(scaled_feats)[0]
                confidence = np.max(probs)

                print(f"[{time.strftime('%H:%M:%S')}] Activity: {prediction.upper()} ({confidence:.2%})")

        except Exception:
            continue
except KeyboardInterrupt:
    print("\nExiting...")
finally:
    if 'ser' in locals(): ser.close()