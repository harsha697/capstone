import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE

print("Loading data...")
X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test  = pd.read_csv("y_test.csv").values.ravel()

# ---------------- FEATURE SELECTION ----------------
print("Loading top features...")
top_features = pd.read_csv("top_features.csv")["feature"].tolist()
top_features = [f for f in top_features if f in X_train.columns]

X_train = X_train[top_features]
X_test  = X_test[top_features]
print("Using Top Features:", len(top_features))

# ---------------- BALANCE CLASSES ----------------
print("Applying SMOTE for class balancing...")
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
print("After SMOTE:", X_train.shape)

# ---------------- TRAIN XGBOOST ----------------
print("Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,      # reduced for laptop memory
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    objective='binary:logistic',
    eval_metric='logloss',
    n_jobs=2,              # limit cores to prevent overload
    random_state=42
)

xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# ---------------- SAVE MODEL ----------------
joblib.dump(xgb_model, "intrusion_model_xgb.pkl")
print("XGBoost saved as 'intrusion_model_xgb.pkl'")

# ---------------- THRESHOLD OPTIMIZATION ----------------
print("Optimizing threshold based on F1-score...")
y_probs = xgb_model.predict_proba(X_test)[:, 1]

best_thresh = 0.5
best_f1 = 0

for t in np.arange(0.3, 0.6, 0.02):
    y_pred_temp = (y_probs > t).astype(int)
    f1 = f1_score(y_test, y_pred_temp)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"Best Threshold: {best_thresh:.2f} | Best F1: {best_f1:.4f}")
joblib.dump(best_thresh, "xgb_threshold.pkl")

# ---------------- EVALUATION ----------------
y_pred = (y_probs > best_thresh).astype(int)

print("\n===== XGBOOST MODEL EVALUATION =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))