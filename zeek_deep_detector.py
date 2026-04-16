import os
import time
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from datetime import datetime

# ---------------- CONFIG ----------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ZEEK_LOG_PATH  = os.path.join(PROJECT_DIR, "zeek_logs", "conn.log")
ALERT_LOG_PATH = os.path.join(PROJECT_DIR, "ensemble_attack_alerts.log")
POLL_INTERVAL  = 1.0   # seconds between log checks

# ---------------- LOAD MODELS ----------------
rf_model     = joblib.load(os.path.join(PROJECT_DIR, "intrusion_model_rf.pkl"))
rf_threshold = float(joblib.load(os.path.join(PROJECT_DIR, "rf_threshold.pkl")))

xgb_model     = joblib.load(os.path.join(PROJECT_DIR, "intrusion_model_xgb.pkl"))
xgb_threshold = float(joblib.load(os.path.join(PROJECT_DIR, "xgb_threshold.pkl")))

cnn_model     = tf.keras.models.load_model(os.path.join(PROJECT_DIR, "cnn_lstm_model.keras"))
cnn_threshold = float(np.load(os.path.join(PROJECT_DIR, "cnn_lstm_threshold.npy"))[0])

# Load scaler and label encoders
scaler         = joblib.load(os.path.join(PROJECT_DIR, "scaler.pkl"))
label_encoders = joblib.load(os.path.join(PROJECT_DIR, "label_encoders.pkl"))

# Load selected features
top_features = pd.read_csv(os.path.join(PROJECT_DIR, "selected_features.csv"))["feature"].tolist()
categorical_cols = list(label_encoders.keys())

print("[INFO] Ensemble detector loaded successfully")
print("[INFO] Features used:", top_features)

# ---------------- HELPER FUNCTIONS ----------------
def safe_float(v, default=0.0):
    try:
        return float(v) if v not in ("-", "") else default
    except:
        return default

def extract_features_zeek(fields):
    # Use a safe getter to avoid IndexError
    def get_field(idx):
        return safe_float(fields[idx]) if idx < len(fields) else 0

    feat = {
        "sttl": get_field(10),
        "ct_state_ttl": get_field(11),
        "dload": get_field(9),
        "rate": get_field(12),
        "sload": get_field(8),
        "dttl": get_field(13),
        "dmean": get_field(16),
        "ackdat": get_field(20),
        "smean": get_field(15),
        "ct_srv_dst": get_field(21),
        "sbytes": get_field(6),
        "synack": get_field(22),
        "dinpkt": get_field(17),
        "tcprtt": get_field(23),
        "ct_srv_src": get_field(18),
        "dbytes": get_field(7),
        "dur": get_field(8),
        "ct_dst_src_ltm": get_field(19),
        "sinpkt": get_field(14),
        "sjit": get_field(24)
    }
    return pd.Series(feat)

def encode_and_scale(df_row):
    for col in categorical_cols:
        if col in df_row:
            le = label_encoders[col]
            df_row[col] = df_row[col].astype(str)
            if "unknown" not in le.classes_:
                le.classes_ = np.append(le.classes_, "unknown")
            df_row[col] = df_row[col].apply(lambda x: x if x in le.classes_ else "unknown")
            df_row[col] = le.transform(df_row[col])
    return scaler.transform(df_row.values.reshape(1, -1))

def predict_cnn(vec):
    x = vec.reshape(1, vec.shape[1], 1)
    prob = float(cnn_model.predict(x, verbose=0)[0][0])
    label = 1 if prob >= cnn_threshold else 0
    return label, prob

# ---------------- ALERT LOGGING ----------------
alert_count = 0
def log_alert(meta, model_name, prob):
    global alert_count
    alert_count += 1
    alert = {
        "alert_id": alert_count,
        "time": datetime.now().isoformat(),
        "src_ip": meta.get("src_ip", ""),
        "src_port": meta.get("src_port", ""),
        "dst_ip": meta.get("dst_ip", ""),
        "dst_port": meta.get("dst_port", ""),
        "model": model_name,
        "confidence": round(prob * 100, 2)
    }
    with open(ALERT_LOG_PATH, "a") as f:
        f.write(str(alert) + "\n")
    print(f"[ALERT #{alert_count}] {meta.get('src_ip')}:{meta.get('src_port')} -> "
          f"{meta.get('dst_ip')}:{meta.get('dst_port')} [{model_name}] confidence={prob*100:.1f}%")

# ---------------- MAIN LOOP ----------------
processed = 0
print(f"[INFO] Monitoring Zeek logs at {ZEEK_LOG_PATH}")
try:
    while True:
        if not os.path.exists(ZEEK_LOG_PATH):
            time.sleep(POLL_INTERVAL)
            continue

        with open(ZEEK_LOG_PATH, "r") as f:
            lines = f.readlines()

        new_lines = lines[processed:]
        for line in new_lines:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            df_row = extract_features_zeek(fields)
            X_scaled = encode_and_scale(df_row)

            # --- Ensemble Predictions ---
            rf_prob = rf_model.predict_proba(X_scaled)[:,1][0]
            rf_label = int(rf_prob >= rf_threshold)

            xgb_prob = xgb_model.predict_proba(X_scaled)[:,1][0]
            xgb_label = int(xgb_prob >= xgb_threshold)

            cnn_label, cnn_prob = predict_cnn(X_scaled)

            votes = rf_label + xgb_label + cnn_label
            ensemble_label = 1 if votes >= 2 else 0
            ensemble_prob = np.mean([rf_prob, xgb_prob, cnn_prob])

            if ensemble_label == 1:
                log_alert(meta={"src_ip": fields[2], "src_port": fields[3], "dst_ip": fields[4], "dst_port": fields[5]},
                          model_name="Ensemble", prob=ensemble_prob)

        processed += len(new_lines)
        if new_lines:
            print(f"[INFO] Processed {len(new_lines)} new lines | Total alerts: {alert_count}", end="\r")

        time.sleep(POLL_INTERVAL)

except KeyboardInterrupt:
    print(f"\n[INFO] Stopped. Total alerts: {alert_count}")
    print(f"[INFO] Alerts saved to: {ALERT_LOG_PATH}")