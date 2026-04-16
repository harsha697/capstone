import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

print("Loading processed datasets...")

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()

# ---------------- MODEL ----------------
print("Training optimized Random Forest for Feature Importance...")

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=40,
    min_samples_leaf=3,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- FEATURE IMPORTANCE ----------------
print("Extracting feature importance...")

importances = model.feature_importances_
feature_names = list(X_train.columns)

feature_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# Save full importance
feature_df.to_csv("feature_importance.csv", index=False)

# Save top features separately
top_features = feature_df.head(30)
top_features.to_csv("top_features.csv", index=False)

print("\nTop 20 Important Features:")
print(feature_df.head(20))

# ---------------- VISUALIZATION ----------------
plt.figure()
plt.barh(top_features["feature"], top_features["importance"])
plt.gca().invert_yaxis()
plt.title("Top 30 Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

print("\nSaved:")
print(" - feature_importance.csv")
print(" - top_features.csv")