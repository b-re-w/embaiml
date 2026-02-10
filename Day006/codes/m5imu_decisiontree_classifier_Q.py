import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, f1_score,
    confusion_matrix, roc_curve, auc, log_loss
)
from sklearn.preprocessing import label_binarize

# -------------------------------------------------------
# 1) Load your three datasets and assign class labels
# -------------------------------------------------------
paths = {
    "static":    "Qimu_smooth_m0.csv",
    "shortdmbl": "Qimu_smooth_m1.csv",
    "longdmbl":  "Qimu_smooth_m2.csv",
}

# UPDATED: Mapping for 11 columns (ts + 3 Accel + 3 Gyro + 4 Quaternion)
colnames = ["timestamp","ax","ay","az","gx","gy","gz","qw","qx","qy","qz"]
dfs = {label: pd.read_csv(p, header=None, names=colnames) for label, p in paths.items()}

# UPDATED: Channels used for classification (10 total features excluding timestamp)
channels = ["ax","ay","az","gx","gy","gz","qw","qx","qy","qz"]

# -------------------------------------------------------
# 2) Convert time-series into windowed feature vectors
# -------------------------------------------------------
FS = 100          # 100 Hz sampling rate
WIN = 100         # 1.0 second window
STRIDE = 50       # 0.5 second hop (50% overlap)

def window_featurize(df, label, win=WIN, stride=STRIDE):
    """
    Computes 5 statistics per channel: mean, std, min, max, and mean-square energy.
    """
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

# Generate feature names for plots (10 channels * 5 stats = 50 features)
feature_names = [f"{c}_{stat}" for c in channels for stat in ["mean","std","min","max","msq"]]

print("Total windows:", X.shape[0], "Feature dim:", X.shape[1])
print("Windows per class:", dict(zip(*np.unique(y, return_counts=True))))

# -------------------------------------------------------
# 3) Split into 60% train, 20% val, 20% test (stratified)
# -------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# -------------------------------------------------------
# 4) Train DT with depth tuning
# -------------------------------------------------------
depths = list(range(1, 21))
val_ll = []

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    val_ll.append(log_loss(y_val, clf.predict_proba(X_val), labels=clf.classes_))

best_depth = depths[int(np.argmin(val_ll))]
clf = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
clf.fit(X_train, y_train)

# -------------------------------------------------------
# 5) Visualizations: Feature Importance & Tree Structure
# -------------------------------------------------------

# Plot 1: Feature Importance (Top 15)
plt.figure(figsize=(10, 6))
importances = clf.feature_importances_
indices = np.argsort(importances)[-15:] # Get top 15 features
plt.title("Top 15 Feature Importances (Gini)")
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# Plot 2: Decision Tree Structure
plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=feature_names,
    class_names=list(clf.classes_),
    filled=True,
    rounded=True,
    fontsize=7,
    max_depth=3  # Limited depth for better visibility
)
plt.title(f"Decision Tree Structure (Best Depth: {best_depth})")
plt.show()

# -------------------------------------------------------
# 6) Performance Evaluation
# -------------------------------------------------------
y_pred = clf.predict(X_test)
print("\n=== Final Test Metrics ===")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred, average="macro"))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=clf.classes_))