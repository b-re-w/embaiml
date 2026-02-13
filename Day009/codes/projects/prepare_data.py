import os
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
# --- CONFIGURATION ---
DATA_PATH = "./uav_commands/"
CLASSES = ["up", "down", "left", "right", "stop", "go", "forward", "backward"]
SAMPLE_RATE = 16000
DURATION = 1.0  # 1 second
N_MFCC = 13  # Number of coefficients
MAX_LEN = 44  # Expected time steps (approx 1 sec / hop_length)


def extract_features(file_path):
    try:
        # Load audio
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        # Pad or truncate to 1 sec
        if len(audio) < SAMPLE_RATE:
            audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)), 'constant')
        else:
            audio = audio[:SAMPLE_RATE]

        # MFCC Extraction
        mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

        # Transpose to (Time, Feat) -> (44, 13)
        mfccs = mfccs.T

        # Ensure fixed length
        if mfccs.shape[0] < MAX_LEN:
            pad_width = MAX_LEN - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfccs = mfccs[:MAX_LEN, :]

        return mfccs
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def load_data():
    X = []
    y = []

    print("Loading audio files...")
    for label in CLASSES:
        folder = os.path.join(DATA_PATH, label)
        if not os.path.isdir(folder):
            print(f"Warning: Folder '{label}' not found.")
            continue

        for file in os.listdir(folder):
            if file.endswith('.wav'):
                file_path = os.path.join(folder, file)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Encode Labels
    le = LabelEncoder()
    y_encoded = to_categorical(le.fit_transform(y))

    print(f"Data Loaded. Shape: {X.shape}")
    return X, y_encoded, le.classes_


# Helper to reshape for CNN (add channel dim)
def prepare_for_cnn(X):
    # (Samples, Time, Feat) -> (Samples, Time, Feat, 1)
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)


def plot_evaluation(model, X_test_data, y_test_data, history, model_name):
    # Predictions
    y_pred_prob = model.predict(X_test_data)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test_data, axis=1)

    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # 2. ROC-AUC (Micro-Average)
    fpr, tpr, _ = roc_curve(y_test_data.ravel(), y_pred_prob.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Micro-Average ROC (area = {roc_auc:.2f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # 3. Log Loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Log Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 4. Detailed Metrics
    print(f"\n--- {model_name} Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=CLASSES))


# Execute Plots
#plot_evaluation(model_cnn, X_test_cnn, y_test, history_cnn, "CNN")
#plot_evaluation(model_lstm, X_test, y_test, history_lstm, "LSTM")