import serial
import numpy as np
import collections
import time
import joblib  # Required for loading

# --- 1. LOAD THE PRE-TRAINED MODEL ---
model_path = 'rf_final_model.pkl'
try:
    # Load the model from the file
    rf_final = joblib.load(model_path)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file {model_path} was not found.")
    exit()

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
WINDOW_SIZE = 100
STRIDE = 50
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz", "qw", "qx", "qy", "qz"]

data_window = collections.deque(maxlen=WINDOW_SIZE)


def extract_features_from_window(window_data):
    # (Same function logic as provided previously)
    np_win = np.array(window_data)
    feats = []
    for i in range(len(CHANNELS)):
        x = np_win[:, i]
        feats.extend([x.mean(), x.std(), x.min(), x.max(), np.mean(x ** 2)])
    return np.array(feats).reshape(1, -1)


# --- START REAL-TIME LOOP ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
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
                features = extract_features_from_window(data_window)

                # Use the LOADED model to predict
                prediction = rf_final.predict(features)[0]
                conf = np.max(rf_final.predict_proba(features)[0])

                print(f"Activity: {prediction.upper()} | Confidence: {conf:.2%}")

        except Exception:
            continue
except KeyboardInterrupt:
    print("\nClosing...")
finally:
    if 'ser' in locals(): ser.close()