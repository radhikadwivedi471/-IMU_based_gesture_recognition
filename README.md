# 🎮 IMU Gesture Control for Gaming (MetaWear)

Control PC games like **Subway Surfers** or **Temple Run** using **hand gestures** captured from a **MetaWear IMU sensor**.  
The system records accelerometer and gyroscope data, processes it using **Machine Learning (Random Forest)**, and converts gestures into **real-time keyboard inputs** (⬆️ ⬇️ ⬅️ ➡️).

---

##  Project Structure
├── record.py # Records raw IMU data for training
├── visualisation.py # Visualizes signals & energy thresholds
├── preprocessing.py # Cleans & segments active gesture data
├── feature_extraction.py # Extracts statistical features
├── train_model.py # Trains Random Forest model
├── play_gamee.py # Real-time gesture-to-keyboard controller
├── dataset/ # Raw recorded data
├── dataset_clean/ # Preprocessed gesture segments
├── final_features.csv # Extracted feature set
└── gesture_model.pkl # Trained ML model

---

##  Hardware Requirements

- **MbientLab MetaWear Sensor** (MetaMotion R / RL)
- Bluetooth Low Energy (BLE) compatible PC or USB dongle

---

##  Installation & Dependencies

Ensure **Python 3.8+** is installed.

```bash

pip install metawear pandas numpy scikit-learn scipy matplotlib pyautogui joblib

---

## Configuration

Open record.py and play_gamee.py and update your device MAC address:

DEVICE_MAC = "E3:EB:F8:DD:51:F7"  # Replace with your MetaWear MAC

## 1️. Data Collection

Record gesture samples: python record.py

Recording Protocol
Relax your hand
Press Enter
Immediately perform the gesture
Wait until "Saved"

 Record 20–30 samples per gesture
(UP, DOWN, LEFT, RIGHT)

## 2️. Visualization (Optional but Recommended)

Visualize and verify signal quality:
python visualisation.py dataset/up_1.csv


Black dotted line → Total Energy
Resting energy ≈ 1.0 g
Gesture peaks help tune thresholds

## 3️. Preprocessing

Clean and segment active gesture regions: python preprocessing.py
If gestures are not detected, adjust: ENERGY_THRESHOLD = 1.3  # Default
This step generates the dataset_clean/ folder.

## 4️. Feature Extraction

Convert segmented signals into statistical features: python feature_extraction.py
Output: final_features.csv

## 5️. Train the Model

Train the Random Forest classifier: python train_model.py
Check the accuracy printed in the terminal
Aim for ≥ 85% accuracy
Retrain with more data if accuracy is low
Output: gesture_model.pkl

## 6️. Play the Game 🎮
Start real-time gesture control: python play_gamee.py
Wait until "GAME ON"
Focus the game window
Perform gestures to control movement

Fine-Tuning the Controller
Adjust these values in play_gamee.py if needed:

Variable	Default	Description
TRIGGER_THRESHOLD	0.25	Lower = more sensitive
POST_TRIGGER_DURATION	0.10	Lower = faster response
PRE_TRIGGER_DURATION	0.15	Captures motion start
COOLDOWN_TIME	0.20	Min time between actions
Probability Threshold	0.35	Accepts lower confidence gestures
