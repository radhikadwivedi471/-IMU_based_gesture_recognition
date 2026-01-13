from __future__ import print_function
from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import *
import pandas as pd
import numpy as np
import joblib
import pyautogui
import time
import os
import sys
import threading
from scipy.stats import entropy, skew, kurtosis
from threading import Lock
from collections import deque

# --- CONFIGURATION ---
DEVICE_MAC = "E3:EB:F8:DD:51:F7"  # Your MetaWear MAC
MODEL_FILE = "gesture_model.pkl"

# --- GAME SETTINGS ---
# Trigger on deviation from 1.0g (Gravity).
# LOWERED to 0.25 for higher sensitivity (Triggers at > 1.25g or < 0.75g)
TRIGGER_THRESHOLD = 0.25

# PREDICTION SETTINGS
# We capture this much data AFTER the trigger
# REDUCED from 0.15 to 0.10 to reduce "lag" (faster reaction time)
POST_TRIGGER_DURATION = 0.10
# We include this much data BEFORE the trigger (Crucial for robustness)
PRE_TRIGGER_DURATION = 0.15
# REDUCED from 0.25 to 0.20 for faster consecutive moves
COOLDOWN_TIME = 0.20

# --- KEYBOARD MAPPING ---
KEY_MAP = {
    'UP': 'up',
    'DOWN': 'down',
    'LEFT': 'left',
    'RIGHT': 'right'
}

# OPTIMIZATION: Instant PyAutoGUI
pyautogui.PAUSE = 0.0

# --- FEATURE EXTRACTION ---
def calculate_spectral_features(signal):
    if len(signal) == 0: return 0, 0
    fft_values = np.fft.rfft(signal)
    fft_magnitude = np.abs(fft_values)
    energy = np.sum(fft_magnitude[1:] ** 2) / len(fft_magnitude)
    sum_mag = np.sum(fft_magnitude[1:])
    if sum_mag == 0:
        spec_entropy = 0
    else:
        psd = fft_magnitude[1:] / sum_mag
        spec_entropy = entropy(psd)
    return energy, spec_entropy

def extract_features_from_axis(axis_data, prefix):
    if len(axis_data) < 2:
        return {f"{prefix}_{k}": 0 for k in ['mean', 'std', 'max', 'min', 'rms', 'skew', 'kurt', 'energy', 'entropy']}

    arr = np.array(axis_data)

    features = {
        f"{prefix}_mean": np.mean(arr),
        f"{prefix}_std": np.std(arr),
        f"{prefix}_max": np.max(arr),
        f"{prefix}_min": np.min(arr),
        f"{prefix}_rms": np.sqrt(np.mean(arr**2)),
        f"{prefix}_skew": skew(arr),
        f"{prefix}_kurt": kurtosis(arr),
    }
    eng, ent = calculate_spectral_features(arr)
    features[f"{prefix}_energy"] = eng
    features[f"{prefix}_entropy"] = ent
    return features

# --- MAIN CONTROLLER CLASS ---
class GameController:
    def __init__(self, model_path):
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        print("Model loaded!")

        # State & Buffers
        self.lock = Lock()

        # Continuous rolling history (for Pre-Trigger context)
        # Assuming ~100Hz, 15 samples is approx 0.15s
        self.history_len = int(PRE_TRIGGER_DURATION * 100)
        self.history_acc = deque(maxlen=self.history_len)
        self.history_gyro = deque(maxlen=self.history_len)

        # Active Recording Buffers
        self.acc_buffer = []
        self.gyro_buffer = []

        # State Machine
        self.is_predicting = False
        self.trigger_time = 0
        self.last_action_time = 0

        print("\n--- ROBUST CONTROLLER READY ---")
        print(f"Trigger Sensitivity: +/- {TRIGGER_THRESHOLD}g")
        print(f"Context Window: -{PRE_TRIGGER_DURATION}s to +{POST_TRIGGER_DURATION}s")
        print("--------------------------------")

    def process_data(self, data_type, x, y, z, timestamp):
        current_time = time.time()

        with self.lock:
            # 1. Always update history (Circular Buffer)
            if data_type == 'acc':
                self.history_acc.append({'t': timestamp, 'x': x, 'y': y, 'z': z})

                # Check Triggers if not already recording
                if not self.is_predicting:
                    if (current_time - self.last_action_time) > COOLDOWN_TIME:
                        # Robust Trigger: Deviation from 1.0g
                        mag = np.sqrt(x**2 + y**2 + z**2)
                        deviation = abs(mag - 1.0)

                        if deviation > TRIGGER_THRESHOLD:
                            print(">>> Flick!", end="\r")
                            self.start_prediction(current_time)

            elif data_type == 'gyro':
                self.history_gyro.append({'t': timestamp, 'x': x, 'y': y, 'z': z})

            # 2. If Predicting, accumulate data
            if self.is_predicting:
                if data_type == 'acc':
                    self.acc_buffer.append({'t': timestamp, 'x': x, 'y': y, 'z': z})
                elif data_type == 'gyro':
                    self.gyro_buffer.append({'t': timestamp, 'x': x, 'y': y, 'z': z})

                # Check if window is full
                if (current_time - self.trigger_time) >= POST_TRIGGER_DURATION:
                    # Offload prediction to thread to not block sensor stream
                    self.is_predicting = False
                    self.last_action_time = current_time

                    # Copy data for the worker thread
                    acc_copy = list(self.acc_buffer)
                    gyro_copy = list(self.gyro_buffer)

                    threading.Thread(target=self.run_inference, args=(acc_copy, gyro_copy)).start()

                    # Reset buffers
                    self.acc_buffer = []
                    self.gyro_buffer = []

    def start_prediction(self, current_time):
        self.is_predicting = True
        self.trigger_time = current_time
        # KEY ROBUSTNESS FIX:
        # Pre-fill the active buffer with history so we capture the START of the flick
        self.acc_buffer = list(self.history_acc)
        self.gyro_buffer = list(self.history_gyro)

    def run_inference(self, acc_data, gyro_data):
        try:
            df_acc = pd.DataFrame(acc_data)
            df_gyro = pd.DataFrame(gyro_data)

            if df_acc.empty: return

            row_features = {}
            # Acc Features
            for axis in ['x', 'y', 'z']:
                row_features.update(extract_features_from_axis(df_acc[axis], f"acc_{axis}"))

            # Gyro Features
            for axis in ['x', 'y', 'z']:
                if not df_gyro.empty and axis in df_gyro.columns:
                    row_features.update(extract_features_from_axis(df_gyro[axis], f"gyro_{axis}"))
                else:
                    row_features.update(extract_features_from_axis([], f"gyro_{axis}"))

            # Correlations
            if not df_acc.empty and len(df_acc) > 1:
                row_features['corr_acc_xy'] = df_acc['x'].corr(df_acc['y'])
                row_features['corr_acc_xz'] = df_acc['x'].corr(df_acc['z'])
                row_features['corr_acc_yz'] = df_acc['y'].corr(df_acc['z'])
            else:
                row_features['corr_acc_xy'] = 0; row_features['corr_acc_xz'] = 0; row_features['corr_acc_yz'] = 0

            # Clean NaNs
            for k, v in row_features.items():
                if pd.isna(v): row_features[k] = 0.0

            input_df = pd.DataFrame([row_features])

            # Predict
            prediction = self.model.predict(input_df)[0]
            probs = self.model.predict_proba(input_df)
            probability = np.max(probs)

            # LOWERED confidence threshold from 0.45 to 0.35 as requested
            if probability > 0.35:
                print(f"⚡ ACTION: {prediction} ({probability*100:.0f}%)    ")
                self._press(KEY_MAP.get(prediction))
            else:
                print(f"x Ignored (Conf: {probability*100:.0f}%)     ")

        except Exception as e:
            print(f"Inference Err: {e}")

    def _press(self, key):
        if not key: return
        pyautogui.keyDown(key)
        time.sleep(0.02)
        pyautogui.keyUp(key)

# --- SETUP SENSOR ---
def main():
    if not os.path.exists(MODEL_FILE):
        print(f"Error: {MODEL_FILE} not found. Train your model first!")
        return

    controller = GameController(MODEL_FILE)

    print(f"Connecting to {DEVICE_MAC}...")
    device = MetaWear(DEVICE_MAC)
    device.connect()
    print("Connected.")

    try:
        # High Speed Settings
        print("Configuring Sensors...")
        libmetawear.mbl_mw_acc_set_odr(device.board, 100.0)
        libmetawear.mbl_mw_acc_set_range(device.board, 8.0)
        libmetawear.mbl_mw_acc_write_acceleration_config(device.board)

        libmetawear.mbl_mw_gyro_bmi160_set_odr(device.board, GyroBoschOdr._100Hz)
        libmetawear.mbl_mw_gyro_bmi160_set_range(device.board, GyroBoschRange._1000dps)
        libmetawear.mbl_mw_gyro_bmi160_write_config(device.board)

        # Callbacks
        def acc_data_handler(ctx, data):
            pt = parse_value(data)
            controller.process_data('acc', pt.x, pt.y, pt.z, data.contents.epoch)

        def gyro_data_handler(ctx, data):
            pt = parse_value(data)
            controller.process_data('gyro', pt.x, pt.y, pt.z, data.contents.epoch)

        acc_callback = FnVoid_VoidP_DataP(acc_data_handler)
        gyro_callback = FnVoid_VoidP_DataP(gyro_data_handler)

        acc_signal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(device.board)
        gyro_signal = libmetawear.mbl_mw_gyro_bmi160_get_rotation_data_signal(device.board)

        libmetawear.mbl_mw_datasignal_subscribe(acc_signal, None, acc_callback)
        libmetawear.mbl_mw_datasignal_subscribe(gyro_signal, None, gyro_callback)

        libmetawear.mbl_mw_acc_enable_acceleration_sampling(device.board)
        libmetawear.mbl_mw_acc_start(device.board)
        libmetawear.mbl_mw_gyro_bmi160_enable_rotation_sampling(device.board)
        libmetawear.mbl_mw_gyro_bmi160_start(device.board)

        print("\n👇 GAME ON! Move hand to start. 👇")

        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Resetting board...")
        libmetawear.mbl_mw_debug_reset(device.board)
        device.disconnect()

if __name__ == "__main__":
    main()