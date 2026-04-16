import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("="*60)
print("      Model Comparison: Ensemble vs CNN+LSTM (20 Features)")
print("="*60)

# ---------------- LOAD TEST DATA ----------------
if os.path.exists("X_test.csv") and os.path.exists("y_test.csv"):
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv").values.ravel()
else:
    raise FileNotFoundError("Test data not found. Make sure X_test.csv and y_test.csv exist.")

# ---------------- USE TOP 20 FEATURES ----------------
top_features = ['sttl', 'ct_state_ttl', 'dload', 'sload', 'rate', 'sbytes', 'smean',
                'ct_srv_dst', 'dmean', 'dbytes', 'ct_dst_src_ltm', 'dttl', 'ct_srv_src',
                'dur', 'ackdat', 'tcprtt', 'dinpkt', 'sinpkt']

X_test = X_test[top_features]

results = {}

# ---------------- ENSEMBLE MODEL ----------------
if os.path.exists("intrusion_model_ensemble.pkl") and os.path.exists("ensemble_threshold.pkl"):
    print("\n[1/2] Evaluating Ensemble Model (RF + XGBoost)...")
    ensemble = joblib.load("intrusion_model_ensemble.pkl")
    thresh_ens = joblib.load("ensemble_threshold.pkl")

    y_prob_ens = ensemble.predict_proba(X_test)[:, 1]
    y_pred_ens = (y_prob_ens >= thresh_ens).astype(int)

    results["Ensemble (RF+XGB)"] = {
        "accuracy": accuracy_score(y_test, y_pred_ens),
        "precision": precision_score(y_test, y_pred_ens),
        "recall": recall_score(y_test, y_pred_ens),
        "f1": f1_score(y_test, y_pred_ens),
        "threshold": thresh_ens
    }

    print(f"   Accuracy : {results['Ensemble (RF+XGB)']['accuracy']*100:.2f}%")
    print(f"   F1       : {results['Ensemble (RF+XGB)']['f1']:.4f}")
else:
    print("   [SKIP] Ensemble model or threshold not found.")

# ---------------- CNN+LSTM MODEL ----------------
if os.path.exists("cnn_lstm_model_20features.keras") and os.path.exists("scaler_cnn_lstm_20features.pkl"):
    print("\n[2/2] Evaluating CNN+LSTM Model...")
    cnn_lstm = tf.keras.models.load_model("cnn_lstm_model_20features.keras")
    scaler_dl = joblib.load("scaler_cnn_lstm_20features.pkl")

    # Scale and reshape
    X_dl = scaler_dl.transform(X_test)
    X_dl = X_dl.reshape(X_dl.shape[0], X_dl.shape[1], 1)

    # Load threshold if exists
    thresh_file = "cnn_lstm_threshold_20features.npy"
    thresh_dl = float(np.load(thresh_file)[0]) if os.path.exists(thresh_file) else 0.5

    y_prob_dl = cnn_lstm.predict(X_dl, batch_size=512, verbose=0).ravel()
    y_pred_dl = (y_prob_dl >= thresh_dl).astype(int)

    results["CNN+LSTM"] = {
        "accuracy": accuracy_score(y_test, y_pred_dl),
        "precision": precision_score(y_test, y_pred_dl),
        "recall": recall_score(y_test, y_pred_dl),
        "f1": f1_score(y_test, y_pred_dl),
        "threshold": thresh_dl
    }

    print(f"   Accuracy : {results['CNN+LSTM']['accuracy']*100:.2f}%")
    print(f"   F1       : {results['CNN+LSTM']['f1']:.4f}")
else:
    print("   [SKIP] CNN+LSTM model or scaler not found.")

# ---------------- FINAL COMPARISON ----------------
if results:
    print("\n" + "="*60)
    print("   FINAL COMPARISON TABLE")
    print("="*60)
    print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Threshold':>10}")
    print("-"*60)
    for name, m in results.items():
        print(f"{name:<22} {m['accuracy']*100:>8.2f}% {m['precision']*100:>9.2f}% "
              f"{m['recall']*100:>7.2f}% {m['f1']*100:>7.2f}% {m['threshold']:>10.2f}")
    print("="*60)

    # Best model by F1-score
    best = max(results, key=lambda k: results[k]["f1"])
    print(f"\n🏆 Best model by F1-score: {best}")
    print(f"   → Recommended for real-time detection")
else:
    print("[INFO] No models evaluated.")

print("\n[DONE] Evaluation complete!")