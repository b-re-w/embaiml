import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, f1_score,
    confusion_matrix, roc_curve, auc, log_loss
)
from sklearn.preprocessing import label_binarize

# -------------------------------------------------------
# 1) Load your three datasets and assign class labels
# -------------------------------------------------------
paths = {
    "static":    "imu_smooth_m0.csv",
    "shortdmbl": "imu_smooth_m1.csv",
    "longdmbl":  "imu_smooth_m2.csv",
}

# Mapping for 11 columns: ts + 3 Accel + 3 Gyro + 4 Quaternions (qw, qx, qy, qz)
colnames = ["timestamp","ax","ay","az","gx","gy","gz","qw","qx","qy","qz"]
dfs = {label: pd.read_csv(p, header=None, names=colnames) for label, p in paths.items()}

# Features used for classification
channels = ["ax","ay","az","gx","gy","gz","qw","qx","qy","qz"]

# -------------------------------------------------------
# 2) Convert time-series into windowed feature vectors
# -------------------------------------------------------
FS = 100
WIN = 100
STRIDE = 50

def window_featurize(df, label, win=WIN, stride=STRIDE):
    X, y = [], []
    for start in range(0, len(df) - win + 1, stride):
        w = df.iloc[start:start+win]
        feats = []
        for c in channels:
            x = w[c].astype(float).to_numpy()
            feats.extend([
                x.mean(),
                x.std(ddof=0),
                x.min(),
                x.max(),
                np.mean(x**2),
            ])
        X.append(feats)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)

X_list, y_list = [], []
for label, df in dfs.items():
    Xw, yw = window_featurize(df, label)
    X_list.append(Xw)
    y_list.append(yw)

X = np.vstack(X_list)
y = np.concatenate(y_list)

feature_names = [f"{c}_{stat}" for c in channels for stat in ["mean","std","min","max","msq"]]

# -------------------------------------------------------
# 3) Split data (60/20/20)
# -------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# -------------------------------------------------------
# 4) Train Random Forest with Hyperparameter Tuning
# -------------------------------------------------------
# We tune 'n_estimators' (number of trees) using the validation set
n_trees_options = [10, 50, 100, 200]
val_scores = []

for n in n_trees_options:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    val_scores.append(rf.score(X_val, y_val))

best_n = n_trees_options[np.argmax(val_scores)]
print(f"Best number of trees: {best_n}")

# Final Forest Model
rf_final = RandomForestClassifier(n_estimators=best_n, max_depth=10, random_state=42, n_jobs=-1)
rf_final.fit(X_train, y_train)

# -------------------------------------------------------
# 5) Visualizations
# -------------------------------------------------------

# Plot 1: Feature Importance

plt.figure(figsize=(10, 8))
importances = rf_final.feature_importances_
indices = np.argsort(importances)[-20:]  # Top 20 features
plt.title("Random Forest: Top 20 Feature Importances")
plt.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Gini Importance")
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 6) Final Evaluation
# -------------------------------------------------------
y_pred = rf_final.predict(X_test)
print("\n=== Random Forest Test Metrics ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred, average='macro'):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=rf_final.classes_))

# -------------------------------------------------------
# 7) Model Store
# -------------------------------------------------------

# Assuming 'rf_final' is your trained RandomForestClassifier object
model_filename = 'rf_final_model.pkl'

# Save the model to a file
joblib.dump(rf_final, model_filename)

print(f"Model saved successfully to {model_filename}")