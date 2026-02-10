import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------------------------------
# 1) Load Data and Featurize (11-input structure)
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# -------------------------------------------------------
# 2) Define the Ensemble Members
# -------------------------------------------------------

# Model A: Random Forest (Bagging)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Model B: HistGradientBoosting (Boosting)
# (Naturally handles large feature sets and is very fast)
hgb = HistGradientBoostingClassifier(random_state=42)

# Model C: MLP Neural Network (requires a Pipeline to include Scaling)
mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
])

# -------------------------------------------------------
# 3) Create and Train the Voting Ensemble
# -------------------------------------------------------
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('hgb', hgb),
        ('mlp', mlp_pipeline)
    ],
    voting='soft' # 'soft' uses weighted probabilities for higher accuracy
)

print("Training Ensemble (RF + Gradient Boosting + MLP)...")
ensemble.fit(X_train, y_train)

# -------------------------------------------------------
# 4) Save the Ensemble
# -------------------------------------------------------
# Note: The 'mlp_pipeline' inside the ensemble already contains the scaler!
joblib.dump(ensemble, 'motion_ensemble_model.pkl')

# -------------------------------------------------------
# 5) Evaluate
# -------------------------------------------------------
y_pred = ensemble.predict(X_test)
print("\n=== Ensemble Test Metrics ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))