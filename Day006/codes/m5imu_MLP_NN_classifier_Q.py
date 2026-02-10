import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- 1) Setup Data Structures ---
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
            # Features: mean, std, min, max, mean-square
            feats.extend([x.mean(), x.std(), x.min(), x.max(), np.mean(x**2)])
        X.append(feats)
        y.append(label)
    return np.array(X), np.array(y)

# Load and process CSVs
X_list, y_list = [], []
for label, p in paths.items():
    df = pd.read_csv(p, header=None, names=colnames)
    Xw, yw = window_featurize(df, label)
    X_list.append(Xw)
    y_list.append(yw)

X = np.vstack(X_list)
y = np.concatenate(y_list)

# --- 2) Feature Scaling (Essential for Neural Networks) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3) Train NN with Backpropagation ---
# hidden_layer_sizes: 2 layers (64 neurons, 32 neurons)
# solver='adam': The backpropagation optimization algorithm
# activation='relu': The rectified linear unit function
nn_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    verbose=True # Shows the loss decreasing during backpropagation
)

nn_model.fit(X_train_scaled, y_train)

# --- 4) Save Model and Scaler ---
joblib.dump(nn_model, 'nn_backprop_model.pkl')
joblib.dump(scaler, 'nn_scaler.pkl')

print(f"\nTraining Complete. Accuracy: {accuracy_score(y_test, nn_model.predict(X_test_scaled)):.4f}")