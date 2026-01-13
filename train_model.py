import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# --- CONFIGURATION ---
INPUT_FILE = "final_features.csv"
MODEL_FILE = "gesture_model.pkl"

def train():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run extract_features.py first.")
        return

    # 1. Load Data
    print("Loading data...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Separate Features (X) and Labels (y)
    # 'label' column contains "UP", "DOWN", etc.
    X = df.drop('label', axis=1)
    y = df['label']

    print(f"Dataset shape: {df.shape} (Rows, Cols)")
    print(f"Classes found: {y.unique()}")

    # 2. Split Data (80% Train, 20% Test)
    # stratify=y ensures we have equal number of Up/Down/Left/Right in the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Initialize and Train Classifier
    # Random Forest is chosen because:
    # - It handles high-dimensional data well (58 features).
    # - It is resistant to overfitting.
    # - It works well without complex scaling.
    print("\nTraining Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 4. Evaluate
    print("Evaluating model...")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n>>> Model Accuracy: {acc * 100:.2f}% <<<")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # 5. Confusion Matrix
    # (Shows if the model is confusing Up with Down, etc.)
    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
    print(confusion_matrix(y_test, y_pred, labels=clf.classes_))

    # 6. Save Model
    print(f"\nSaving model to {MODEL_FILE}...")
    joblib.dump(clf, MODEL_FILE)
    print("Done! You can now use 'gesture_model.pkl' for the game.")

if __name__ == "__main__":
    train()