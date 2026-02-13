import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.utils import to_categorical

# --- CONFIGURATION ---
WINDOW_SIZE = 50  # 50 samples (approx 1 sec at 50Hz)
STEP_SIZE = 25  # 50% overlap
CLASSES = ["static", "verti", "horiz", "circ", "type", "draw", "write"]


def load_and_preprocess_data(root_dir="imu_class_data"):
    X = []
    y = []

    print(f"Loading data from {root_dir}...")

    for label in CLASSES:
        folder_path = os.path.join(root_dir, label)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{label}' not found.")
            continue

        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        for file in files:
            file_path = os.path.join(folder_path, file)
            try:
                # Read CSV. Assuming format: Timestamp, Q0, Q1, Q2, Q3
                df = pd.read_csv(file_path)

                # Select only Quaternion columns (Indices 1, 2, 3, 4)
                # Adjust if your CSV has headers or different column order
                data = df.iloc[:, 1:5].values

                # Sliding Window Segmentation
                for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
                    window = data[i: i + WINDOW_SIZE, :]
                    if window.shape[0] == WINDOW_SIZE:
                        X.append(window)
                        y.append(label)
            except Exception as e:
                print(f"Error reading {file}: {e}")

    X = np.array(X)
    y = np.array(y)

    # Encoder Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    print(f"Data Loaded. Shape: {X.shape}")
    return X, y_categorical, le


def plot_metrics(model, history, X_test, y_test, class_names, model_name):
    # 1. Predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # 2. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.show()

    # 3. ROC-AUC (Micro-Average)
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred_prob.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Micro-Average ROC (area = {roc_auc:.2f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{model_name}_roc_curve.png')
    plt.show()

    # 4. Log Loss (Training History)
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Model Log Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_log_loss.png')
    plt.show()

    # 5. Text Report
    print(f"\n--- {model_name} Classification Report ---")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))