import matplotlib.pyplot as plt
import csv
import sys
import os
import math

def calculate_magnitude(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def plot_gesture(filename):
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        print("Make sure you recorded data first!")
        return

    # Data Containers
    t_acc, x_acc, y_acc, z_acc, mag_acc = [], [], [], [], []
    t_gyro, x_gyro, y_gyro, z_gyro = [], [], [], []

    print(f"Opening {filename}...")

    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)

            # Read first row to establish start time
            rows = list(reader)
            if not rows:
                print("Error: File is empty.")
                return

            start_time = float(rows[0]['timestamp'])

            for row in rows:
                ts = float(row['timestamp']) - start_time # Normalize time to start at 0
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])

                if row['sensor'] == 'acc':
                    t_acc.append(ts)
                    x_acc.append(x)
                    y_acc.append(y)
                    z_acc.append(z)
                    mag_acc.append(calculate_magnitude(x, y, z)) # Calculate Total Energy

                elif row['sensor'] == 'gyro':
                    t_gyro.append(ts)
                    x_gyro.append(x)
                    y_gyro.append(y)
                    z_gyro.append(z)

    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Accelerometer
    ax1.plot(t_acc, x_acc, label='X', color='r', alpha=0.6)
    ax1.plot(t_acc, y_acc, label='Y', color='g', alpha=0.6)
    ax1.plot(t_acc, z_acc, label='Z', color='b', alpha=0.6)
    # Plot Magnitude (The most important line for Segmentation)
    ax1.plot(t_acc, mag_acc, label='Total Energy (Magnitude)', color='black', linewidth=2, linestyle='--')

    ax1.set_title(f'Accelerometer Data (Look at the Black Line for Threshold)', fontsize=14)
    ax1.set_ylabel('Acceleration (g)')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper right')

    # Add a horizontal line at 1g (Gravity) for reference
    ax1.axhline(y=1.0, color='gray', linestyle=':', label='Gravity (1g)')

    # Plot 2: Gyroscope
    ax2.plot(t_gyro, x_gyro, label='X', color='r')
    ax2.plot(t_gyro, y_gyro, label='Y', color='g')
    ax2.plot(t_gyro, z_gyro, label='Z', color='b')

    ax2.set_title('Gyroscope Data (Wrist Rotation)', fontsize=14)
    ax2.set_ylabel('Angular Velocity (deg/s)')
    ax2.set_xlabel('Time (seconds)')
    ax2.grid(True)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    print("Graph opened! Check the popup window.")
    plt.show()

if __name__ == "__main__":
    # Check if user provided a filename
    if len(sys.argv) < 2:
        print("\n!!! MISSING FILE NAME !!!")
        print("Usage: python visualize_gesture.py <path_to_csv>")
        print("Example: python visualize_gesture.py dataset/up_1.csv")
        print("-" * 30)

        # Fallback: Try to find the first CSV in 'dataset' folder automatically
        if os.path.exists("dataset"):
            files = [f for f in os.listdir("dataset") if f.endswith(".csv")]
            if files:
                print(f"Auto-loading first file found: dataset/{files[0]}")
                plot_gesture(f"dataset/{files[0]}")
            else:
                print("No CSV files found in 'dataset' folder.")
    else:
        plot_gesture(sys.argv[1])