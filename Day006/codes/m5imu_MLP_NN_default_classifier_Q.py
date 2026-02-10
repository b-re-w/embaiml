import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# --- 1) Load and Window Data (Same logic as before) ---
paths = {"static": "imu_smooth_m0.csv", "shortdmbl": "imu_smooth_m1.csv", "longdmbl": "imu_smooth_m2.csv"}
colnames = ["timestamp","ax","ay","az","gx","gy","gz","qw","qx","qy","qz"]
channels = ["ax","ay","az","gx","gy","gz","qw","qx","qy","qz"]

def window_featurize(df, label, win=100, stride=50):
    X, y = [], []
    for start in range(0, len(df) - win + 1, stride):
        w = df.iloc[start:start+win]
        feats = []
        for c in channels:
            x = w[c].astype(float).to_numpy()
            feats.extend([x.mean(), x.std(), x.min(), x.max(), np.mean(x**2)])
        X.append(feats)
        y.append(label)
    return np.array(X), np.array(y)

X_list, y_list = [], []
for label, p in paths.items():
    df = pd.read_csv(p, header=None, names=colnames)
    Xw, yw = window_featurize(df, label)
    X_list.append(Xw)
    y_list.append(yw)

X = np.vstack(X_list)
y = np.concatenate(y_list)

# --- 2) Scaling and Splitting ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# CRITICAL: MLP requires feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3) Train MLP Neural Network ---
# Hidden layers: (100, 50) means two layers with 100 and 50 neurons respectively
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, activation='relu',
                    solver='adam', random_state=42)
mlp.fit(X_train_scaled, y_train)

# --- 4) Save Model AND Scaler ---
joblib.dump(mlp, 'mlp_model.pkl')
joblib.dump(scaler, 'mlp_scaler.pkl') # You must save the scaler to use it in real-time!

print(f"MLP Accuracy: {accuracy_score(y_test, mlp.predict(X_test_scaled)):.4f}")