import serial
import csv
import threading
import time


# --- CONFIGURATION ---
SERIAL_PORT = 'COM7'  # Change to '/dev/ttyUSB0' on Linux or '/dev/tty.usbserial-...' on Mac
BAUD_RATE = 115200
FILE_NAME = "imu_data_log.csv"

logging = False
exit_program = False

def key_listener():
    global logging, exit_program
    print("Commands: [S] Start/Stop Logging | [Q] Quit")
    while not exit_program:
        user_input = input().upper()
        if user_input == 'S':
            logging = not logging
            status = "STARTED" if logging else "STOPPED"
            print(f"Logging {status}")
        elif user_input == 'Q':
            exit_program = True

# Initialize Serial
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) # Wait for connection to stabilize
except Exception as e:
    print(f"Error: {e}")
    exit()

# Start the keyboard thread
threading.Thread(target=key_listener, daemon=True).start()

print(f"Listening to {SERIAL_PORT}...")

with open(FILE_NAME, mode='w', newline='', buffering=1) as file:
    writer = csv.writer(file)
    # Write Header
    writer.writerow(["Timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz", "pitch", "roll", "yaw"])
    file.flush()

    try:
        while not exit_program:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='replace').strip()

                if logging:
                    data = line.split(',')
                    if len(data) == 10: # Ensure we got a full line
                        writer.writerow(data)
                        file.flush()
                        print(f"Logged: {data}")
                        
    except KeyboardInterrupt:
        print("\nClosing...")
    finally:
        ser.close()
        print("Data saved to", FILE_NAME)
