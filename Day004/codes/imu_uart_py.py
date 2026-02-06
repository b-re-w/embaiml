import serial
import csv
import time
import threading

# Configuration - Change 'COM3' or '/dev/ttyACM0' to your port
SERIAL_PORT = 'COM3' 
BAUD_RATE = 115200
FILE_NAME = "raw_imu_log.csv"

running = True
logging = False
data_file = None
writer = None

def keyboard_listener():
    global logging, running
    print("Commands: [S] Start/Stop Capture | [Q] Quit")
    while running:
        cmd = input().strip().upper()
        if cmd == 'S':
            logging = not logging
            status = "STARTED" if logging else "STOPPED"
            print(f"Data Capture {status}")
        elif cmd == 'Q':
            running = False
            print("Quitting...")

# Start keyboard thread
threading.Thread(target=keyboard_listener, daemon=True).start()

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    with open(FILE_NAME, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ts", "ax", "ay", "az", "gx", "gy", "gz"])
        
        print(f"Connected to {SERIAL_PORT}. Waiting for commands...")
        
        while running:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if logging and line:
                    try:
                        # Split string and convert to appropriate types
                        values = line.split(',')
                        if len(values) == 7:
                            writer.writerow(values)
                    except Exception as e:
                        print(f"Data error: {e}")
            time.sleep(0.001) # Small sleep to prevent CPU hogging

except serial.SerialException:
    print(f"Error: Could not open port {SERIAL_PORT}")
finally:
    if ser: ser.close()
    print("File saved and Serial closed.")