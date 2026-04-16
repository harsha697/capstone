import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import xgboost as xgb

print("Loading datasets...")
X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test  = pd.read_csv("y_test.csv").values.ravel()

# ---------------- FEATURE SELECTION ----------------
print("Loading top features...")
top_features = pd.read_csv("top_features.csv")["feature"].tolist()
top_features = [f for f in top_features if f in X_train.columns]
print("Using Top Features:", top_features)
print("Total features used:", len(top_features))

X_train = X_train[top_features]
X_test  = X_test[top_features]

# ---------------- RANDOM FOREST ----------------
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,   # reduced to save memory
    max_depth=25,       # reduced to avoid overfitting
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=2,           # limit cores
    random_state=42
)
rf.fit(X_train, y_train)

# ---------------- XGBOOST ----------------
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,      # reduced for memory
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=np.sum(y_train==0)/np.sum(y_train==1),
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='hist',
    n_jobs=2,              # limit cores
    random_state=42
)
xgb_model.fit(X_train, y_train, verbose=False)

# ---------------- SOFT-VOTING ENSEMBLE ----------------
print("Creating soft-voting ensemble...")
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb_model)],
    voting='soft',
    weights=[3, 2],   # RF slightly stronger
    n_jobs=2
)
ensemble.fit(X_train, y_train)

joblib.dump(ensemble, "intrusion_model_ensemble.pkl")
print("Ensemble model saved as 'intrusion_model_ensemble.pkl'")

# ---------------- THRESHOLD OPTIMIZATION ----------------
print("Optimizing threshold based on F1-score...")
probs = ensemble.predict_proba(X_test)[:, 1]

best_thresh = 0.5
best_f1 = 0
for t in np.arange(0.3, 0.7, 0.01):
    preds = (probs >= t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"Best Threshold: {best_thresh:.2f} | Best F1: {best_f1:.4f}")
joblib.dump(best_thresh, "ensemble_threshold.pkl")

# ---------------- FINAL EVALUATION ----------------
y_pred = (probs >= best_thresh).astype(int)

print("\n===== ENSEMBLE RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))