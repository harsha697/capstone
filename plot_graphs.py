import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib

# -------------------------------
# Load data
# -------------------------------
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

model = joblib.load("intrusion_model_rf.pkl")
threshold = joblib.load("rf_threshold.pkl")

# Load correct features
features = pd.read_csv("selected_features.csv")["feature"].tolist()
X_test = X_test[features]

# -------------------------------
# Predictions
# -------------------------------
probs = model.predict_proba(X_test)[:, 1]
y_pred = (probs >= threshold).astype(int)

# -------------------------------
# Confusion Matrix (SAVE)
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i][j], ha='center', va='center')

plt.colorbar()
plt.savefig("confusion_matrix.png")  # 🔥 SAVE
plt.close()

# -------------------------------
# ROC Curve (SAVE)
# -------------------------------
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png")  # 🔥 SAVE
plt.close()

print("Graphs saved: confusion_matrix.png, roc_curve.png")