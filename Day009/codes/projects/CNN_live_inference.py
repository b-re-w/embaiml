import serial
import serial.tools.list_ports
import numpy as np
import tensorflow as tf
from collections import deque
import time
import sys

# =========================================================
# CONFIGURATION (MUST MATCH YOUR TRAINING SETTINGS)
# =========================================================
MODEL_PATH = 'CNN_model.h5'

# The labels used during training (Alphabetical order is standard for Keras)
CLASSES = ["circ", "draw", "horiz", "static", "type", "verti", "write"]

# Window size: Must be EXACTLY what you used for training (e.g., 50 or 64)
WINDOW_SIZE = 50

# Baud rate must match Arduino Serial.begin()
BAUD_RATE = 115200

# Confidence Threshold: Only show prediction if confidence > 70%
CONFIDENCE_THRESHOLD = 0.7


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_serial_port():
    """Lists available ports and asks user to select."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found!")
        sys.exit()

    print("\nAvailable Serial Ports:")
    for i, port in enumerate(ports):
        print(f"[{i}] {port.device} - {port.description}")

    while True:
        try:
            choice = input("\nSelect port number: ")
            idx = int(choice)
            if 0 <= idx < len(ports):
                return ports[idx].device
        except ValueError:
            pass
        print("Invalid selection.")


# =========================================================
# MAIN INFERENCE LOOP
# =========================================================

def main():
    # 1. Load Model
    print(f"Loading model: {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit()

    # 2. Connect to Serial
    port_name = get_serial_port()
    try:
        ser = serial.Serial(port_name, BAUD_RATE, timeout=1)
        print(f"Connected to {port_name}. Listening for data...")
    except Exception as e:
        print(f"Serial error: {e}")
        sys.exit()

    # 3. Buffer Initialization
    # deque automatically handles the sliding window (pops old, appends new)
    data_buffer = deque(maxlen=WINDOW_SIZE)

    print("\n--- LIVE CLASSIFICATION STARTED ---\n")

    while True:
        try:
            if ser.in_waiting > 0:
                # Read line: "Timestamp, Ax, Ay, Az, Gx, Gy, Gz, Q0, Q1, Q2, Q3"
                raw_line = ser.readline().decode('utf-8', errors='ignore').strip()

                # Check for valid CSV
                if ',' in raw_line:
                    parts = raw_line.split(',')

                    # Ensure we have enough data points.
                    # Assuming format: [Timestamp, Q0, Q1, Q2, Q3]
                    # We need indices 1-4 for Quaternions (Q0-Q3)
                    if len(parts) >= 5:
                        try:
                            # Extract Quaternions (Indices 1-4)
                            # ADJUST INDICES HERE if your Arduino format is different!
                            q0 = float(parts[1])
                            q1 = float(parts[2])
                            q2 = float(parts[3])
                            q3 = float(parts[4])

                            # Add to buffer
                            data_buffer.append([q0, q1, q2, q3])

                            # Perform Inference only when buffer is full
                            if len(data_buffer) == WINDOW_SIZE:
                                # Prepare input: Reshape to (1, WINDOW_SIZE, 4)
                                input_data = np.array(data_buffer)
                                input_data = np.expand_dims(input_data, axis=0)

                                # Predict
                                prediction = model.predict(input_data, verbose=0)
                                class_idx = np.argmax(prediction)
                                confidence = prediction[0][class_idx]

                                label = CLASSES[class_idx]

                                # Output Result
                                if confidence > CONFIDENCE_THRESHOLD:
                                    # Print formatted output with overwrite (carriage return)
                                    print(f"\rAction: {label.upper()} \tConfidence: {confidence:.2f}   ", end="")
                                else:
                                    print(f"\rAction: ... \t\tConfidence: {confidence:.2f}   ", end="")

                        except ValueError:
                            continue  # distinct non-float data

        except KeyboardInterrupt:
            print("\n\nStopping...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            break

    ser.close()
    print("Serial closed.")


if __name__ == "__main__":
    main()