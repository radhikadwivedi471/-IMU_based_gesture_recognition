import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import entropy, skew, kurtosis

# --- CONFIGURATION ---
INPUT_FOLDER = "dataset_clean"
OUTPUT_FILE = "final_features.csv"

# --- HELPER FUNCTIONS ---

def calculate_spectral_features(signal):
    # Perform FFT (Fast Fourier Transform)
    # We use rfft for real-valued input (more efficient)
    fft_values = np.fft.rfft(signal)
    fft_magnitude = np.abs(fft_values)

    # 1. Spectral Energy
    # Sum of squared magnitudes (excluding DC component at index 0)
    energy = np.sum(fft_magnitude[1:] ** 2) / len(fft_magnitude)

    # 2. Spectral Entropy
    # Normalize magnitudes to get a probability distribution (PSD)
    # Avoid division by zero
    sum_mag = np.sum(fft_magnitude[1:])
    if sum_mag == 0:
        spec_entropy = 0
    else:
        psd = fft_magnitude[1:] / sum_mag
        spec_entropy = entropy(psd)

    return energy, spec_entropy

def extract_features_from_axis(axis_data, prefix):
    # Statistical Features (Time Domain)
    mean_val = np.mean(axis_data)
    std_val = np.std(axis_data)
    max_val = np.max(axis_data)
    min_val = np.min(axis_data)
    rms_val = np.sqrt(np.mean(axis_data**2))
    skew_val = skew(axis_data)
    kurt_val = kurtosis(axis_data)

    # Frequency Features (Frequency Domain)
    spec_energy, spec_entropy = calculate_spectral_features(axis_data)

    features = {
        f"{prefix}_mean": mean_val,
        f"{prefix}_std": std_val,
        f"{prefix}_max": max_val,
        f"{prefix}_min": min_val,
        f"{prefix}_rms": rms_val,
        f"{prefix}_skew": skew_val,
        f"{prefix}_kurt": kurt_val,
        f"{prefix}_energy": spec_energy,
        f"{prefix}_entropy": spec_entropy
    }
    return features

def extract_features_from_file(file_path):
    try:
        df = pd.read_csv(file_path)

        # Separate Accelerometer and Gyroscope
        df_acc = df[df['sensor'] == 'acc'].reset_index(drop=True)
        df_gyro = df[df['sensor'] == 'gyro'].reset_index(drop=True)

        if df_acc.empty or df_gyro.empty:
            print(f"Skipping {os.path.basename(file_path)}: Missing sensor data (Acc or Gyro empty)")
            return None

        row_features = {}

        # --- 1. ACCELEROMETER FEATURES (Ax, Ay, Az) ---
        for axis in ['x', 'y', 'z']:
            axis_feats = extract_features_from_axis(df_acc[axis], f"acc_{axis}")
            row_features.update(axis_feats)

        # --- 2. GYROSCOPE FEATURES (Gx, Gy, Gz) ---
        for axis in ['x', 'y', 'z']:
            axis_feats = extract_features_from_axis(df_gyro[axis], f"gyro_{axis}")
            row_features.update(axis_feats)

        # --- 3. CORRELATION FEATURES (Relationships between axes) ---
        # Tells us about the "directionality" of the movement
        row_features['corr_acc_xy'] = df_acc['x'].corr(df_acc['y'])
        row_features['corr_acc_xz'] = df_acc['x'].corr(df_acc['z'])
        row_features['corr_acc_yz'] = df_acc['y'].corr(df_acc['z'])

        # --- 4. EXTRACT LABEL ---
        # Filename example: "up_1.csv" -> label: "UP"
        filename = os.path.basename(file_path)
        # Assuming filename format is "gesture_number.csv"
        label = filename.split('_')[0].upper()
        row_features['label'] = label

        return row_features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Folder '{INPUT_FOLDER}' does not exist.")
        print("Please run 'process_dataset.py' first.")
        return

    files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    print(f"Found {len(files)} clean files. Starting feature extraction...")

    all_data = []

    for file_path in files:
        features = extract_features_from_file(file_path)
        if features:
            all_data.append(features)

    if all_data:
        # Convert to DataFrame
        df_features = pd.DataFrame(all_data)

        # Move 'label' to the last column for easier reading/training
        cols = [c for c in df_features.columns if c != 'label'] + ['label']
        df_features = df_features[cols]

        # Fill any NaNs (just in case) with 0
        df_features = df_features.fillna(0)

        # Save
        df_features.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSuccess! Extracted {len(df_features.columns)} features from {len(df_features)} files.")
        print(f"Features saved to: {OUTPUT_FILE}")
        print("You can now use this file to train your Machine Learning model.")
    else:
        print("No features extracted. Please check your data files.")

if __name__ == "__main__":
    main()