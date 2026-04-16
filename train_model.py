import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

print("Loading datasets...")

X = pd.read_csv("X_train.csv")
y = pd.read_csv("y_train.csv").values.ravel()

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

# ---------------- USE SAME FEATURES ----------------
print("Loading selected features...")

top_features = pd.read_csv("selected_features.csv").iloc[:, 0].tolist()

X = X[top_features]
X_test = X_test[top_features]

print("Number of features used:", len(top_features))

# ---------------- TRAIN-VALIDATION SPLIT ----------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- MODEL TRAINING ----------------
print("Training optimized Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,          # reduced (faster + better generalization)
    max_depth=25,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- THRESHOLD OPTIMIZATION (VALIDATION) ----------------
print("Optimizing threshold...")

val_probs = model.predict_proba(X_val)[:, 1]

best_thresh = 0.5
best_f1 = 0

for t in np.arange(0.2, 0.8, 0.02):
    preds = (val_probs >= t).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"Best Threshold (Validation): {best_thresh:.2f} | F1: {best_f1:.4f}")

# ---------------- FINAL TEST EVALUATION ----------------
test_probs = model.predict_proba(X_test)[:, 1]
y_pred = (test_probs >= best_thresh).astype(int)

print("\n===== FINAL RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------- SAVE MODEL ----------------
joblib.dump(model, "intrusion_model_rf.pkl")
joblib.dump(top_features, "feature_names.pkl")
joblib.dump(best_thresh, "rf_threshold.pkl")

print("Model + threshold saved!")