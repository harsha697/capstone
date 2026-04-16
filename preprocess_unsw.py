import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

print("Loading datasets...")

train_path = "/home/robotics/MLCyberProject/capstone_project/UNSW_NB15_training-set(in).csv"
test_path  = "/home/robotics/MLCyberProject/capstone_project/UNSW_NB15_testing-set(in).csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Train:", train.shape, "Test:", test.shape)


# REMOVE ID COLUMN 

if 'id' in train.columns:
    train = train.drop(columns=['id'])
    test = test.drop(columns=['id'])


# Handle missing values

train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# Encode categorical columns safely

categorical_cols = ['proto', 'service', 'state']
encoders = {}

print("Encoding categorical columns...")

for col in categorical_cols:
    le = LabelEncoder()
    
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    
    encoders[col] = le

print("Encoding done!")

# -------------------------------
# Split features and labels
# -------------------------------
X_train = train.drop(['label', 'attack_cat'], axis=1)
y_train = train['label']

X_test = test.drop(['label', 'attack_cat'], axis=1)
y_test = test['label']

# -------------------------------
# FEATURE SELECTION (CLEAN 🔥)
# -------------------------------
print("Selecting important features...")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importance = pd.Series(rf.feature_importances_, index=X_train.columns)

# REMOVE any accidental leakage feature again
importance = importance.drop(labels=['label'], errors='ignore')

top_features = importance.sort_values(ascending=False).head(20).index

print("Top features selected:", list(top_features))

# Apply feature selection
X_train = X_train[top_features]
X_test = X_test[top_features]

# Save feature names properly
pd.DataFrame(top_features, columns=["feature"]).to_csv("selected_features.csv", index=False)

# -------------------------------
# Scaling
# -------------------------------
print("Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_df = pd.DataFrame(X_train_scaled, columns=top_features)
X_test_df = pd.DataFrame(X_test_scaled, columns=top_features)

# -------------------------------
# SMOTE (CONTROLLED ⚠️)
# -------------------------------
print("Applying SMOTE...")

smote = SMOTE(sampling_strategy=0.6, random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_df, y_train)

print("After SMOTE:", X_train_bal.shape)

# -------------------------------
# Save everything
# -------------------------------
print("Saving processed data...")

X_train_bal.to_csv("X_train.csv", index=False)
X_test_df.to_csv("X_test.csv", index=False)
pd.DataFrame(y_train_bal, columns=["label"]).to_csv("y_train.csv", index=False)
pd.DataFrame(y_test, columns=["label"]).to_csv("y_test.csv", index=False)

joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "label_encoders.pkl")

print("Preprocessing Completed Successfully!")