import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# =========================
# Path Configuration
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "processed")

os.makedirs(OUTPUT_PATH, exist_ok=True)

FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]

# =========================
# Data Loading
# =========================

def load_and_combine(files):
    dfs = []
    for file in files:
        file_path = os.path.join(RAW_DATA_PATH, file)
        print(f"Loading file: {file_path}")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print("Combined dataset shape:", combined_df.shape)
    return combined_df

# =========================
# Data Cleaning
# =========================

def clean_data(df):
    print("Cleaning data...")

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with missing values
    initial_shape = df.shape
    df.dropna(inplace=True)
    print(f"Dropped {initial_shape[0] - df.shape[0]} rows containing NaN or Inf values")

    # Drop non-feature / leakage columns
    drop_cols = [
        "Flow ID", "Source IP", "Destination IP",
        "Timestamp", "Source Port", "Destination Port"
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    print("Shape after cleaning:", df.shape)
    return df

# =========================
# Encoding & Scaling
# =========================

def encode_and_scale(df):
    print("Encoding labels and scaling features...")

    X = df.drop("Label", axis=1)
    y = df["Label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    print("Feature matrix shape:", X_scaled.shape)
    print("Number of classes:", len(label_encoder.classes_))

    return X_scaled, y_encoded, label_encoder

# =========================
# Main Execution
# =========================

if __name__ == "__main__":
    print("Starting preprocessing pipeline...")

    df = load_and_combine(FILES)
    df = clean_data(df)
    X, y, label_encoder = encode_and_scale(df)

    # Save processed outputs
    X_path = os.path.join(OUTPUT_PATH, "X_processed.csv")
    y_path = os.path.join(OUTPUT_PATH, "y_labels.csv")

    X.to_csv(X_path, index=False)
    pd.Series(y, name="Label").to_csv(y_path, index=False)

    print("Preprocessing complete.")
    print("Saved files:")
    print(" -", X_path)
    print(" -", y_path)
