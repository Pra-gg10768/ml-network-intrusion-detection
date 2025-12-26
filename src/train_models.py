import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# =========================
# Path Configuration
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models")

os.makedirs(MODEL_PATH, exist_ok=True)

# =========================
# Load Processed Data
# =========================

print("Loading processed datasets...")

X = pd.read_csv(os.path.join(DATA_PATH, "X_processed.csv"))
y = pd.read_csv(os.path.join(DATA_PATH, "y_labels.csv")).values.ravel()

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

# =========================
# Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# =========================
# Logistic Regression (Baseline)
# =========================

print("\nTraining Logistic Regression...")

lr = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    class_weight="balanced"
)

lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Save model
joblib.dump(lr, os.path.join(MODEL_PATH, "logistic_regression.pkl"))

# =========================
# Random Forest
# =========================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Save model
joblib.dump(rf, os.path.join(MODEL_PATH, "random_forest.pkl"))

# =========================
# Confusion Matrix
# =========================

print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))

results = {
    "logistic_regression": {
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "classification_report": classification_report(y_test, y_pred_lr, output_dict=True)
    },
    "random_forest": {
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "classification_report": classification_report(y_test, y_pred_rf, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist()
    }
}

REPORTS_PATH = os.path.join(BASE_DIR, "..", "reports")
os.makedirs(REPORTS_PATH, exist_ok=True)

with open(os.path.join(REPORTS_PATH, "day5_results.json"), "w") as f:
    json.dump(results, f, indent=4)

print("Evaluation metrics saved to reports/day5_results.json")

print("\nDay 5 model training complete.")
