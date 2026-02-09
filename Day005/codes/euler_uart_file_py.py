import serial
import serial.tools.list_ports
import csv
import time
import threading
import sys
import os

# --- Global Flags for Thread Control ---
is_recording = False
keep_running = True
serial_connected = False

# --- Configuration ---
BAUD_RATE = 115200  # Match this with your Arduino Serial.begin()


def get_serial_port():
    """Lists available serial ports and asks user to select one."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found!")
        return None

    print("\nAvailable Serial Ports:")
    for i, port in enumerate(ports):
        print(f"[{i}] {port.device} - {port.description}")

    while True:
        try:
            choice = int(input("\nSelect port number (e.g., 0): "))
            if 0 <= choice < len(ports):
                return ports[choice].device
        except ValueError:
            pass
        print("Invalid selection. Try again.")


def serial_worker(ser, writer, file_obj):
    """Background thread to read serial data and write to CSV."""
    global is_recording, keep_running

    print(f"\nConnected to {ser.port}. Listening for data...")

    while keep_running:
        try:
            if ser.in_waiting > 0:
                # Read line from UART
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                except Exception as e:
                    print(f"Read Error: {e}")
                    continue

                # Check if data looks valid (simple check: contains commas)
                if ',' in line:
                    # Parse data to ensure it matches columns (Optional but good for safety)
                    data = line.split(',')

                    if is_recording:
                        # Write to CSV
                        writer.writerow(data)
                        file_obj.flush()  # Ensure data is written physically to disk immediately
                        print(f"\r[REC] Data: {line[:50]}...", end="")  # visual feedback
                    else:
                        # Just print raw stream so user knows it's alive
                        print(f"\r[Idle] Data: {line[:50]}...", end="")

        except serial.SerialException:
            print("\nSerial device disconnected!")
            keep_running = False
            break


def main():
    global is_recording, keep_running

    # 1. Setup Serial Port
    port_name = get_serial_port()
    if not port_name:
        return

    # 2. Get Filename
    action_name = input("\nEnter name of the action (will be used as filename): ").strip()
    # Sanitize filename (remove spaces or bad chars)
    filename = f"{action_name.replace(' ', '_')}_{int(time.time())}.csv"

    # 3. Initialize Serial and File
    try:
        ser = serial.Serial(port_name, BAUD_RATE, timeout=1)

        # Open CSV file
        with open(filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write Header
            # Format: Timestamp, Acc(3), Gyro(3), RPY(3)
            header = ["Timestamp", "AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", "Roll", "Pitch", "Yaw"]
            csv_writer.writerow(header)

            print(f"\nFile created: {filename}")
            print("-" * 40)
            print("COMMANDS:")
            print("  's' + Enter -> START / STOP Recording")
            print("  'q' + Enter -> QUIT")
            print("-" * 40)

            # 4. Start Background Thread
            t = threading.Thread(target=serial_worker, args=(ser, csv_writer, csv_file))
            t.daemon = True  # Thread dies if main program dies
            t.start()

            # 5. Main Loop (Command Listener)
            while keep_running:
                cmd = input().strip().lower()

                if cmd == 's':
                    is_recording = not is_recording
                    state = "STARTED" if is_recording else "STOPPED"
                    print(f"\n\n>>> RECORDING {state} <<<\n")

                elif cmd == 'q':
                    print("\nQuitting...")
                    keep_running = False
                    break

        ser.close()
        print("Serial connection closed. Bye!")

    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()