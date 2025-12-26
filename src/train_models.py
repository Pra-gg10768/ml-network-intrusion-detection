import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models")
REPORTS_PATH = os.path.join(BASE_DIR, "..", "reports")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)

# =========================
# Load Data
# =========================
print("Loading processed datasets...")
X = pd.read_csv(os.path.join(DATA_PATH, "X_processed.csv"))
y = pd.read_csv(os.path.join(DATA_PATH, "y_labels.csv")).values.ravel()
print("Feature shape:", X.shape, "| Label shape:", y.shape)

# =========================
# Train / Validation / Test Split
# =========================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print("Train size:", X_train.shape, "| Validation size:", X_val.shape, "| Test size:", X_test.shape)

# =========================
# Logistic Regression (Baseline)
# =========================
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced")
lr.fit(X_train, y_train)

y_val_pred_lr = lr.predict(X_val)
y_test_pred_lr = lr.predict(X_test)

print("\nLogistic Regression Results (Validation Set)")
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred_lr))
print(classification_report(y_val, y_val_pred_lr))

joblib.dump(lr, os.path.join(MODEL_PATH, "logistic_regression.pkl"))

# =========================
# Random Forest
# =========================
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
rf.fit(X_train, y_train)

y_val_pred_rf = rf.predict(X_val)
y_test_pred_rf = rf.predict(X_test)

print("\nRandom Forest Results (Validation Set)")
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred_rf))
print(classification_report(y_val, y_val_pred_rf))

joblib.dump(rf, os.path.join(MODEL_PATH, "random_forest.pkl"))

# =========================
# Confusion Matrix (Test Set)
# =========================
print("\nConfusion Matrix (Random Forest - Test Set):")
print(confusion_matrix(y_test, y_test_pred_rf))

# =========================
# Save Metrics to JSON
# =========================
results = {
    "logistic_regression": {
        "validation_accuracy": accuracy_score(y_val, y_val_pred_lr),
        "test_accuracy": accuracy_score(y_test, y_test_pred_lr),
        "validation_report": classification_report(y_val, y_val_pred_lr, output_dict=True),
        "test_report": classification_report(y_test, y_test_pred_lr, output_dict=True)
    },
    "random_forest": {
        "validation_accuracy": accuracy_score(y_val, y_val_pred_rf),
        "test_accuracy": accuracy_score(y_test, y_test_pred_rf),
        "validation_report": classification_report(y_val, y_val_pred_rf, output_dict=True),
        "test_report": classification_report(y_test, y_test_pred_rf, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_test_pred_rf).tolist()
    }
}

with open(os.path.join(REPORTS_PATH, "day5_results.json"), "w") as f:
    json.dump(results, f, indent=4)

print("\nDay 5 model training and validation complete!")
print("Evaluation metrics saved to reports/day5_results.json")
