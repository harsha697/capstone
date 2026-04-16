import os
import time
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ---------------- CONFIG ----------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ZEEK_LOG_DIR = os.path.join(PROJECT_DIR, "zeek_logs")
LOG_FILE = "conn.log"
ALERT_LOG = os.path.join(PROJECT_DIR, "attack_alerts.log")

os.makedirs(ZEEK_LOG_DIR, exist_ok=True)

# ---------------- LOAD MODEL PIPELINE ----------------
# Load trained Random Forest model
model = joblib.load(os.path.join(PROJECT_DIR, "intrusion_model_rf.pkl"))

# Load threshold
threshold = joblib.load(os.path.join(PROJECT_DIR, "rf_threshold.pkl"))

# Load scaler and label encoders
scaler = joblib.load(os.path.join(PROJECT_DIR, "scaler.pkl"))
label_encoders = joblib.load(os.path.join(PROJECT_DIR, "label_encoders.pkl"))

# Load selected features used during training
top_features = pd.read_csv(os.path.join(PROJECT_DIR, "selected_features.csv"))["feature"].tolist()
categorical_cols = list(label_encoders.keys())

print("[INFO] Random Forest detector loaded successfully")
print("[INFO] Features used:", top_features)

# ---------------- HANDLER ----------------
class ZeekLogHandler(FileSystemEventHandler):

    def on_modified(self, event):
        if event.src_path.endswith(LOG_FILE):
            self.process_log(event.src_path)

    def process_log(self, file_path):
        try:
            # Load Zeek log
            df = pd.read_csv(file_path, low_memory=False)
            if df.empty:
                return

            # Align columns and ensure correct DataFrame with column names
            X = pd.DataFrame(df.reindex(columns=top_features, fill_value=0), columns=top_features)

            # Encode categorical features safely
            for col in categorical_cols:
                if col in X.columns:
                    le = label_encoders[col]
                    X[col] = X[col].astype(str)

                    # Add "unknown" class if not present
                    if "unknown" not in le.classes_:
                        le.classes_ = np.append(le.classes_, "unknown")

                    # Replace unseen labels with "unknown"
                    X[col] = X[col].apply(lambda x: x if x in le.classes_ else "unknown")
                    X[col] = le.transform(X[col])

            # Ensure all numeric
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

            # Scale features (keep as DataFrame)
            X_scaled = pd.DataFrame(scaler.transform(X), columns=top_features)

            # Predict with Random Forest and threshold
            probs = model.predict_proba(X_scaled)[:, 1]
            preds = (probs >= threshold).astype(int)

            df["prediction"] = preds
            attacks = df[df["prediction"] == 1]

            # --------- LOG SUMMARY ---------
            print(f"\n[INFO] Records processed: {len(df)}")
            print(f"[INFO] Attacks detected: {len(attacks)}")

            # --------- ALERT OUTPUT ---------
            if not attacks.empty:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(ALERT_LOG, "a") as f:
                    for _, row in attacks.iterrows():
                        alert = {"timestamp": timestamp, **row.to_dict()}
                        print(f"[ALERT] Attack detected! {alert}")
                        f.write(str(alert) + "\n")

        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")

# ---------------- WATCHER ----------------
observer = Observer()
observer.schedule(ZeekLogHandler(), path=ZEEK_LOG_DIR, recursive=False)
observer.start()

print(f"🚨 Monitoring Zeek logs in {ZEEK_LOG_DIR} for attacks...")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping detector...")
    observer.stop()

observer.join()