# ML-Based Network Intrusion Detection System (IDS)

This project implements an end-to-end **Machine Learning–based Network Intrusion Detection System (IDS)** using real-world network traffic data. The goal is to automatically identify malicious activities in network traffic by learning patterns from labeled data, thereby enhancing network security and incident response capabilities.

The system follows a structured machine learning pipeline, starting from raw data ingestion and preprocessing to model training, evaluation, and result documentation.

---

## 📌 Motivation

With the increasing complexity and volume of cyberattacks, traditional rule-based intrusion detection systems struggle to scale and adapt. Machine learning–based IDS solutions can learn from historical traffic patterns and generalize to detect unseen or evolving attack behaviors.

This project is designed to:
- Gain hands-on experience with **real-world cybersecurity datasets**
- Apply **machine learning techniques** to intrusion detection
- Build a **reproducible and research-oriented ML pipeline**
- Prepare a **Masters-level project** suitable for academic and research applications

---

## 📂 Dataset

The project uses the **CICIDS2017 dataset**, a widely adopted benchmark dataset for intrusion detection research. It contains realistic network traffic captured in a controlled environment, including both benign and malicious activities such as DDoS attacks.

Key characteristics:
- Large-scale dataset (750K+ network flows)
- High-dimensional feature space (network flow statistics)
- Labeled traffic (Benign vs Attack)

---

## ⚙️ Project Pipeline

The project follows a modular and reproducible workflow:

1. **Data Ingestion**
   - Load and combine raw network traffic files
   - Standardize column names and formats

2. **Data Preprocessing**
   - Handle missing and infinite values
   - Remove non-informative and leakage-prone features (IPs, ports, timestamps)
   - Encode class labels
   - Normalize numerical features
   - Persist processed datasets for reproducibility

3. **Exploratory Data Analysis (EDA)**
   - Class distribution analysis
   - Feature-level inspection
   - Initial understanding of attack vs normal traffic patterns

4. **Model Training**
   - Logistic Regression as a baseline IDS model
   - Random Forest as a stronger non-linear classifier

5. **Model Evaluation**
   - Accuracy, precision, recall, and F1-score
   - Confusion matrix analysis
   - Emphasis on attack recall and false positive rates

6. **Result Documentation**
   - Persist evaluation metrics to files
   - Summarize results for reproducibility and reporting

---

## 📊 Current Results (Baseline)

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression | ~98.6%   |
| Random Forest       | ~99.99%  |

- Logistic Regression achieves high recall for attack detection, making it suitable as a baseline IDS.
- Random Forest demonstrates near-perfect classification with minimal false positives and false negatives.

Detailed metrics and confusion matrices are stored in the `reports/` directory.

---

## 🗂️ Project Structure

ml-network-intrusion-detection/
│
├── data/
│ ├── raw/ # Original CICIDS2017 CSV files
│ └── processed/ # Cleaned and scaled datasets
│
├── src/
│ ├── preprocessing.py
│ └── train_models.py
│
├── models/ # Trained ML models
├── reports/ # Evaluation metrics and results
└── README.md

yaml
Copy code

---

## 🚧 Project Status

**Week 1 – Completed**
- Dataset exploration and preprocessing
- Baseline and ensemble model training
- Evaluation and result documentation

**Upcoming**
- Feature importance and explainability
- ROC-AUC and threshold analysis
- Advanced models and anomaly-based detection
- Research-oriented performance analysis

---

## 🎯 Learning Outcomes

Through this project, I aim to strengthen my understanding of:
- Applied machine learning for cybersecurity
- Large-scale data preprocessing and evaluation
- Intrusion detection system design
- Reproducible ML pipelines for research

---

## 📌 Notes

This project is part of my preparation for **Masters-level studies** in Machine Learning and Cybersecurity, with a focus on practical, research-driven system development.