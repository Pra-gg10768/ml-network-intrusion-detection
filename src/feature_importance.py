import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models")
REPORTS_PATH = os.path.join(BASE_DIR, "..", "reports")

os.makedirs(REPORTS_PATH, exist_ok=True)

# =========================
# Load Data
# =========================
print("Loading processed datasets and Random Forest model...")

X = pd.read_csv(os.path.join(DATA_PATH, "X_processed.csv"))
y = pd.read_csv(os.path.join(DATA_PATH, "y_labels.csv")).values.ravel()

# Train / Validation Split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load trained Random Forest
rf_model = joblib.load(os.path.join(MODEL_PATH, "random_forest.pkl"))

print("Random Forest loaded. Computing feature importances...")

# =========================
# Feature Importance
# =========================
importances = rf_model.feature_importances_
feature_names = X.columns

feat_imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Save feature importance CSV
feat_imp_df.to_csv(os.path.join(REPORTS_PATH, "feature_importances.csv"), index=False)
print("Feature importances saved to reports/feature_importances.csv")

# =========================
# Top 20 Features Plot
# =========================
plt.figure(figsize=(12,8))
sns.barplot(x="Importance", y="Feature", data=feat_imp_df.head(20), palette="viridis")
plt.title("Top 20 Feature Importances - Random Forest IDS")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_PATH, "top20_feature_importances.png"))
plt.show()

# =========================
# Cumulative Importance Plot
# =========================
feat_imp_df["Cumulative"] = np.cumsum(feat_imp_df["Importance"])
plt.figure(figsize=(10,6))
plt.plot(range(len(feat_imp_df)), feat_imp_df["Cumulative"])
plt.xlabel("Number of Features")
plt.ylabel("Cumulative Importance")
plt.title("Cumulative Feature Importance")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_PATH, "cumulative_feature_importance.png"))
plt.show()

# =========================
# Validation-aware Analysis
# =========================
print("Evaluating Random Forest on validation set...")

y_val_pred = rf_model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
val_report = classification_report(y_val, y_val_pred, output_dict=True)

print(f"Validation Accuracy: {val_acc:.6f}")
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred))

# Save validation metrics to JSON
import json

results_path = os.path.join(REPORTS_PATH, "feature_importance_validation.json")
val_results = {
    "validation_accuracy": val_acc,
    "validation_classification_report": val_report
}

with open(results_path, "w") as f:
    json.dump(val_results, f, indent=4)

print(f"Validation metrics saved to {results_path}")

print("\nDay 6 feature importance and validation-aware analysis complete!")
