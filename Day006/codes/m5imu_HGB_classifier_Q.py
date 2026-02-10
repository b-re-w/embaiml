import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------------------------
# 1) Load and Window Data (11-input structure)
# -------------------------------------------------------
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
            # 5 stats: mean, std, min, max, mean-square energy
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

# Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# -------------------------------------------------------
# 2) Train HistGradientBoosting
# -------------------------------------------------------
# Note: Gradient Boosting does NOT require a StandardScaler
gb_model = HistGradientBoostingClassifier(
    max_iter=100,          # Number of boosting rounds (trees)
    learning_rate=0.1,     # Step size shrinkage
    max_depth=5,           # Depth of individual trees
    l2_regularization=1.5, # Prevents overfitting
    random_state=42
)

print("Training Gradient Boosting model...")
gb_model.fit(X_train, y_train)

# -------------------------------------------------------
# 3) Save the Model
# -------------------------------------------------------
joblib.dump(gb_model, 'gb_imu_model.pkl')
print("Model saved as gb_imu_model.pkl")

# -------------------------------------------------------
# 4) Evaluation
# -------------------------------------------------------
y_pred = gb_model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=gb_model.classes_))