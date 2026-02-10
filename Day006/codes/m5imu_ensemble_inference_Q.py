import serial
import numpy as np
import collections
import joblib
import time
import sys

# -------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -------------------------------------------------------
MODEL_PATH = 'motion_ensemble_model.pkl'
SERIAL_PORT = 'COM3'  # Windows: 'COM3', Mac/Linux: '/dev/ttyACM0'
BAUD_RATE = 115200

# Windowing parameters must match training exactly
WINDOW_SIZE = 100  # 1.0 second at 100Hz
STRIDE = 50  # Update every 0.5 seconds
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz", "qw", "qx", "qy", "qz"]

# -------------------------------------------------------
# 2. LOAD THE TRAINED ENSEMBLE
# -------------------------------------------------------
print(f"Loading Ensemble Model from {MODEL_PATH}...")
try:
    # This loads the VotingClassifier containing RF, HGB, and the MLP Pipeline
    ensemble_model = joblib.load(MODEL_PATH)
    print("Model loaded successfully! Ready for inference.")
    print(f"Classes: {ensemble_model.classes_}")
except FileNotFoundError:
    print(f"Error: Could not find '{MODEL_PATH}'.")
    print("Make sure you ran the training script to generate the model file.")
    sys.exit(1)

# Initialize sliding window buffer
data_window = collections.deque(maxlen=WINDOW_SIZE)


# -------------------------------------------------------
# 3. FEATURE EXTRACTION FUNCTION
# -------------------------------------------------------
def extract_features(window_data):
    """
    Converts raw window (100x11) into feature vector (1x50).
    Stats: Mean, Std, Min, Max, Mean-Square-Energy
    """
    # Convert list of lists to numpy array
    np_win = np.array(window_data)  # Shape: (100, 10) excluding timestamp

    feats = []
    # Loop through the 10 sensor channels
    for i in range(len(CHANNELS)):
        col_data = np_win[:, i]

        # Calculate the 5 statistics
        mu = np.mean(col_data)
        std = np.std(col_data)
        mn = np.min(col_data)
        mx = np.max(col_data)
        msq = np.mean(col_data ** 2)

        feats.extend([mu, std, mn, mx, msq])

    # Reshape for Scikit-Learn (1 sample, 50 features)
    return np.array(feats).reshape(1, -1)


# -------------------------------------------------------
# 4. REAL-TIME SERIAL LOOP
# -------------------------------------------------------
def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"\nConnected to {SERIAL_PORT}. Waiting for IMU stream...")

        sample_count = 0

        while True:
            # Read a line from the M5Core2
            line = ser.readline().decode('utf-8', errors='ignore').strip()

            if not line:
                continue

            try:
                # Parse CSV: ts, ax, ay, az, gx, gy, gz, qw, qx, qy, qz
                parts = line.split(',')

                # Ensure we have all 11 columns
                if len(parts) == 11:
                    # Convert to floats
                    values = [float(x) for x in parts]

                    # Store only sensor data (indices 1-10), drop timestamp (index 0)
                    data_window.append(values[1:])
                    sample_count += 1

                    # Check if window is full and stride limit reached
                    if len(data_window) == WINDOW_SIZE and sample_count >= STRIDE:
                        sample_count = 0  # Reset stride counter

                        # A. Extract Features
                        features = extract_features(data_window)

                        # B. Predict (Ensemble Voting)
                        # The scaler pipeline inside the ensemble handles normalization automatically
                        prediction = ensemble_model.predict(features)[0]

                        # Get confidence probability
                        probs = ensemble_model.predict_proba(features)[0]
                        confidence = np.max(probs)

                        # C. Display Output
                        timestamp = time.strftime('%H:%M:%S')

                        # Optional: Add visual flair for different classes
                        marker = "●" if prediction == "static" else ">>"

                        print(f"[{timestamp}] {marker} Activity: {prediction.upper():<12} | Conf: {confidence:.2%}")
                        # Example Output: [10:05:22] >> Activity: LONGDMBL     | Conf: 98.50%

            except ValueError:
                continue  # Skip malformed lines

    except serial.SerialException as e:
        print(f"Serial Error: {e}")
        print("Check if the M5Core2 is plugged in and the port is correct.")
    except KeyboardInterrupt:
        print("\nStopping inference...")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial connection closed.")


if __name__ == "__main__":
    main()