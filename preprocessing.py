import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

# --- CONFIGURATION (Isse apne graph ke hisaab se change karein) ---
INPUT_FOLDER = "dataset"
OUTPUT_FOLDER = "dataset_clean"

# 1. Smoothing Settings
WINDOW_SIZE = 5  # 5 samples ka average (Smoothing ke liye)

# 2. Segmentation Settings (Most Important!)
# Apne visualizer graph ki Black Line dekhein.
# Agar Rest = 1.0g aur Peak = 3.0g hai, toh Threshold 1.3g ya 1.5g rakhein.
ENERGY_THRESHOLD = 1.3
BUFFER_SAMPLES = 20  # Gesture start hone se pehle aur baad ke 20 samples extra rakhna (padding)

def calculate_magnitude(df):
    # Energy = sqrt(x^2 + y^2 + z^2)
    return np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

def smooth_data(df, window):
    # Rolling Mean (Moving Average) filter
    # Hum sirf x, y, z columns ko smooth karenge, timestamp aur sensor type ko nahi
    df_smooth = df.copy()
    for col in ['x', 'y', 'z']:
        df_smooth[col] = df[col].rolling(window=window, center=True).mean()

    # Kinare (edges) par NaN values aa sakti hain, unhe fill karein
    df_smooth = df_smooth.fillna(method='bfill').fillna(method='ffill')
    return df_smooth

def segment_gesture(df_acc, df_gyro):
    # 1. Calculate Energy of Accelerometer
    energy = calculate_magnitude(df_acc)

    # 2. Find Active Region (Where Energy > Threshold)
    is_active = energy > ENERGY_THRESHOLD

    # Agar koi activity nahi mili (Threshold too high?)
    if not is_active.any():
        print("  -> Warning: No movement detected above threshold. Check file manually.")
        return None, None

    # 3. Get Start and End Indices
    active_indices = is_active[is_active].index
    start_idx = active_indices[0]
    end_idx = active_indices[-1]

    # 4. Add Buffer (Thoda pehle aur thoda baad ka data bhi rakho)
    start_idx = max(0, start_idx - BUFFER_SAMPLES)
    end_idx = min(len(df_acc), end_idx + BUFFER_SAMPLES)

    # 5. Crop Data using Time
    start_time = df_acc.loc[start_idx, 'timestamp']
    end_time = df_acc.loc[end_idx-1, 'timestamp'] # Safe end time

    # Accel aur Gyro dono ko same time window mein kaato
    seg_acc = df_acc[(df_acc['timestamp'] >= start_time) & (df_acc['timestamp'] <= end_time)]
    seg_gyro = df_gyro[(df_gyro['timestamp'] >= start_time) & (df_gyro['timestamp'] <= end_time)]

    return seg_acc, seg_gyro

def process_all_files():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Saari CSV files dhundo
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    print(f"Found {len(files)} files in {INPUT_FOLDER}...")

    for file_path in files:
        filename = os.path.basename(file_path)

        try:
            # Read CSV
            df = pd.read_csv(file_path)

            # Separate Accel and Gyro
            df_acc = df[df['sensor'] == 'acc'].reset_index(drop=True)
            df_gyro = df[df['sensor'] == 'gyro'].reset_index(drop=True)

            if df_acc.empty:
                print(f"Skipping {filename}: No accelerometer data.")
                continue

            # --- STEP 1: SMOOTHING ---
            df_acc_smooth = smooth_data(df_acc, WINDOW_SIZE)
            df_gyro_smooth = smooth_data(df_gyro, WINDOW_SIZE)

            # --- STEP 2: SEGMENTATION ---
            seg_acc, seg_gyro = segment_gesture(df_acc_smooth, df_gyro_smooth)

            if seg_acc is not None and not seg_acc.empty:
                # Combine back for saving
                seg_final = pd.concat([seg_acc, seg_gyro])

                # Save to Clean Folder
                output_path = os.path.join(OUTPUT_FOLDER, filename)
                seg_final.to_csv(output_path, index=False)
                print(f"Processed: {filename} (Rows: {len(df)} -> {len(seg_acc)})")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Is script ko run karne ke liye 'pandas' install karein: pip install pandas
    process_all_files()
    print("\nProcessing Complete! Check 'dataset_clean' folder.")