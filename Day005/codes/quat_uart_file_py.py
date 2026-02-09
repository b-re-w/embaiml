import serial
import serial.tools.list_ports
import csv
import time
import threading
import sys

# --- Global Flags ---
is_recording = False
keep_running = True

# --- Configuration ---
BAUD_RATE = 115200


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
            choice = input("\nSelect port number (e.g., 0): ")
            if choice.isdigit() and 0 <= int(choice) < len(ports):
                return ports[int(choice)].device
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
                except Exception:
                    continue

                # Check if data looks valid (contains commas)
                if ',' in line:
                    data = line.split(',')

                    # We expect roughly 11 columns: Time, 3xAcc, 3xGyro, 4xQuat
                    if len(data) >= 11:
                        if is_recording:
                            writer.writerow(data)
                            file_obj.flush()  # Save immediately
                            print(f"\r[REC] Data: {line[:60]}...", end="")
                        else:
                            print(f"\r[Idle] Data: {line[:60]}...", end="")
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
    filename = f"{action_name.replace(' ', '_')}_{int(time.time())}.csv"

    # 3. Initialize Serial and File
    try:
        ser = serial.Serial(port_name, BAUD_RATE, timeout=1)

        with open(filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Updated Header for Quaternions
            header = [
                "Timestamp",
                "AccX", "AccY", "AccZ",
                "GyroX", "GyroY", "GyroZ",
                "Q0", "Q1", "Q2", "Q3"  # Quaternion W, X, Y, Z
            ]
            csv_writer.writerow(header)

            print(f"\nFile created: {filename}")
            print("Press 's' + Enter to START/STOP recording.")
            print("Press 'q' + Enter to QUIT.")

            # 4. Start Background Thread
            t = threading.Thread(target=serial_worker, args=(ser, csv_writer, csv_file))
            t.daemon = True
            t.start()

            # 5. Command Loop
            while keep_running:
                cmd = input().strip().lower()
                if cmd == 's':
                    is_recording = not is_recording
                    state = "STARTED" if is_recording else "STOPPED"
                    print(f"\n\n>>> RECORDING {state} <<<\n")
                elif cmd == 'q':
                    keep_running = False
                    break

        ser.close()
        print("\nDone.")

    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()