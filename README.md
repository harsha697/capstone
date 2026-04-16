# 🚨 Real-Time Cyberattack Detection using Machine Learning & Zeek

## 📌 Overview

This project presents a **Real-Time Network Intrusion Detection System (NIDS)** that combines **machine learning models** with **Zeek network monitoring logs** to detect cyberattacks as they occur.

Trained on the **UNSW-NB15 dataset**, the system is designed to identify malicious traffic with high accuracy while maintaining strong recall — a critical requirement in cybersecurity.

---

## 🎯 Objectives

- Detect cyberattacks from network traffic with high accuracy
- Minimize **false negatives** (missed attacks)
- Enable **real-time detection using Zeek logs**
- Compare multiple ML models and improve performance using ensemble learning
- Build a modular and scalable IDS pipeline

---

## 🧠 Models Used

### 🌲 Random Forest
- Strong baseline model
- Handles non-linear relationships
- Robust against overfitting

### ⚡ XGBoost
- Gradient boosting algorithm
- High performance on structured data
- Handles class imbalance effectively

### 🤝 Ensemble Model (Soft Voting)
- Combines **Random Forest + XGBoost**
- Uses probability-based predictions
- Improves overall generalization and detection capability

---

## 📊 Dataset

- **Dataset:** UNSW-NB15  
- **Type:** Network intrusion dataset  
- **Classes:**
  - `0` → Normal Traffic
  - `1` → Attack Traffic  

---

## ⚙️ Preprocessing Steps

- Handling missing values
- Label encoding categorical features
- Feature scaling using StandardScaler
- Feature selection (Top 20 important features)
- Class imbalance handling using **SMOTE**

---

## 🧪 Model Performance

### 🔹 Ensemble Model Results

| Metric        | Value |
|--------------|------|
| Accuracy     | **~92.47%** |
| Precision    | ~92% |
| Recall       | ~94% |
| F1-score     | **~93%** |

### 📌 Confusion Matrix

- True Positives (Attack detected): **42,651**
- False Negatives (Missed attacks): **2,681**
- False Positives: **3,516**
- True Negatives: **33,484**

✅ The model achieves **high recall**, making it suitable for real-world cybersecurity applications where missing attacks is costly.

---

## 📈 ROC Curve

- **AUC Score:** ~0.98  
- Indicates excellent model separability between attack and normal traffic.

---

## 🎯 Threshold Optimization

Instead of using a default threshold (0.5):

- Tested multiple probability thresholds
- Selected optimal threshold based on **F1-score**
- Final threshold improved detection sensitivity

---

## 🛰️ Real-Time Detection with Zeek

### 🔄 Workflow

1. Zeek captures live network traffic
2. Logs are converted into feature vectors
3. Trained ML model performs predictions
4. Alerts are generated for detected attacks

### 💻 Script

```bash
zeek_realtime_detector_ubuntu.py

## 👨‍💻 Author

SIVAKOTI HARSHAVARDHAN

---

## ⭐ Acknowledgements

* UNSW Canberra Cybersecurity Research Group
* Zeek (formerly Bro) Network Security Monitor
* Open-source ML community

---

⭐ *If you find this project useful, consider starring the repository!*
