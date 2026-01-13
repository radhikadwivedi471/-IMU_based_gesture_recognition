from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import *
import time
import csv
import os
import sys

# --- CONFIGURATION ---
DEVICE_MAC = "E3:EB:F8:DD:51:F7"  # Aapka Device MAC Address
GESTURES = ["UP", "DOWN", "LEFT", "RIGHT"]
SAMPLES_PER_GESTURE = 50
RECORDING_TIME = 1.5  # Seconds
OUTPUT_FOLDER = "dataset"

# --- GLOBAL VARIABLES FOR DATA COLLECTION ---
acc_data = []
gyro_data = []

def header_handler(column_names):
    # Helper to return CSV header
    return column_names

# --- DATA CALLBACKS ---
# Ye functions tab call honge jab sensor se data aayega
def acc_callback(ctx, data):
    val = parse_value(data)
    # Timestamp (epoch), X, Y, Z
    acc_data.append((time.time(), val.x, val.y, val.z))

def gyro_callback(ctx, data):
    val = parse_value(data)
    # Timestamp (epoch), X, Y, Z
    gyro_data.append((time.time(), val.x, val.y, val.z))

# Callback wrappers for C-library
acc_handler = FnVoid_VoidP_DataP(acc_callback)
gyro_handler = FnVoid_VoidP_DataP(gyro_callback)

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print(f"Connecting to {DEVICE_MAC}...")
    device = MetaWear(DEVICE_MAC)
    device.connect()
    print("Connected!")

    try:
        # 1. CONFIGURE SENSORS (100Hz, 8g, 1000dps)
        print("Configuring sensors...")

        # Accelerometer Setup
        libmetawear.mbl_mw_acc_set_odr(device.board, 100.0)
        libmetawear.mbl_mw_acc_set_range(device.board, 8.0)
        libmetawear.mbl_mw_acc_write_acceleration_config(device.board)

        # Gyroscope Setup (BMI160/BMI270 generic)
        libmetawear.mbl_mw_gyro_bmi160_set_odr(device.board, GyroBoschOdr._100Hz)
        libmetawear.mbl_mw_gyro_bmi160_set_range(device.board, GyroBoschRange._1000dps)
        libmetawear.mbl_mw_gyro_bmi160_write_config(device.board)

        # Get Signal Objects
        acc_signal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(device.board)
        gyro_signal = libmetawear.mbl_mw_gyro_bmi160_get_rotation_data_signal(device.board)

        print("\n--- READY TO RECORD ---")
        print("Instructions:")
        print("1. Relax your hand.")
        print(f"2. Press ENTER -> Immediately perform a quick FLICK (within {RECORDING_TIME}s).")
        print("3. Wait for 'Saved' message before relaxing.")

        # 2. RECORDING LOOP
        for gesture in GESTURES:
            print(f"\n>>> SWITCHING TO GESTURE: {gesture} <<<")

            for i in range(1, SAMPLES_PER_GESTURE + 1):
                filename = f"{OUTPUT_FOLDER}/{gesture.lower()}_{i}.csv"

                # Check if file already exists to avoid overwriting (optional)
                if os.path.exists(filename):
                    print(f"Skipping {filename} (Already exists)")
                    continue

                input(f"Press ENTER to record {gesture} #{i} ...")

                # Clear buffers
                acc_data.clear()
                gyro_data.clear()

                # Start Streams
                libmetawear.mbl_mw_datasignal_subscribe(acc_signal, None, acc_handler)
                libmetawear.mbl_mw_datasignal_subscribe(gyro_signal, None, gyro_handler)

                libmetawear.mbl_mw_acc_enable_acceleration_sampling(device.board)
                libmetawear.mbl_mw_acc_start(device.board)

                libmetawear.mbl_mw_gyro_bmi160_enable_rotation_sampling(device.board)
                libmetawear.mbl_mw_gyro_bmi160_start(device.board)

                # RECORDING WINDOW
                print(">> RECORDING... FLICK NOW!")
                time.sleep(RECORDING_TIME)
                print(">> STOP.")

                # Stop Streams
                libmetawear.mbl_mw_acc_stop(device.board)
                libmetawear.mbl_mw_acc_disable_acceleration_sampling(device.board)

                libmetawear.mbl_mw_gyro_bmi160_stop(device.board)
                libmetawear.mbl_mw_gyro_bmi160_disable_rotation_sampling(device.board)

                libmetawear.mbl_mw_datasignal_unsubscribe(acc_signal)
                libmetawear.mbl_mw_datasignal_unsubscribe(gyro_signal)

                # Save to CSV
                # Format: timestamp, sensor_type, x, y, z
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "sensor", "x", "y", "z"])

                    # Write Acc Data
                    for row in acc_data:
                        writer.writerow([row[0], "acc", row[1], row[2], row[3]])

                    # Write Gyro Data
                    for row in gyro_data:
                        writer.writerow([row[0], "gyro", row[1], row[2], row[3]])

                print(f"Saved: {filename} (Acc: {len(acc_data)}, Gyro: {len(gyro_data)} samples)")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        print("Resetting board and disconnecting...")
        libmetawear.mbl_mw_debug_reset(device.board)
        device.disconnect()

if __name__ == "__main__":
    main()