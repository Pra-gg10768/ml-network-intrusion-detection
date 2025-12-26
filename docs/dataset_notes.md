1️⃣ Dataset Overview

The CICIDS2017 dataset contains flow-based network traffic records, where each row represents a bidirectional network flow rather than individual packets. The dataset is designed to simulate real-world enterprise network traffic, including both benign activity and multiple types of cyber attacks.

Each flow is described using approximately 78 numerical features, capturing statistical properties such as packet counts, byte rates, flow duration, and TCP flag information.

2️⃣ Files Used in This Project

For initial experimentation, the following files were selected:

Monday-WorkingHours.pcap_ISCX.csv
This file primarily contains benign network traffic, representing normal user behavior during working hours.

Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
This file contains a high concentration of DDoS attack traffic, making it suitable for studying intrusion patterns and class imbalance.

These two files provide a clear contrast between normal and malicious network behavior and allow faster iteration during early development.

3️⃣ Target Variable (Label Column)

The Label column indicates the class of each network flow.

Typical values include:

BENIGN

DDoS (in Friday dataset)

This confirms that the problem can initially be treated as a binary classification task (Benign vs Attack), with scope for later extension to multi-class classification.

4️⃣ Data Quality Observations

Initial inspection of the dataset revealed:

Presence of missing (NaN) values in certain flow features

Occasional infinite values (e.g., division by zero in rate-based features)

Potential duplicate flows

Significant class imbalance, especially in the DDoS dataset

These issues indicate the necessity of a robust data cleaning and preprocessing pipeline before model training.

5️⃣ Feature Characteristics

The features represent statistical summaries of network flows, including:

Flow duration and inter-arrival times

Packet length statistics (mean, max, variance)

Bytes per second and packets per second

TCP flag counts (SYN, ACK, FIN, etc.)

These features are numerical and well-suited for machine learning models, but require normalization due to varying scales.

6️⃣ Preliminary Insights

Benign and malicious traffic are expected to differ significantly in packet rates, flow durations, and flag patterns.

Class imbalance suggests that evaluation metrics such as precision, recall, and F1-score will be more informative than accuracy.

Flow-based features make the dataset suitable for both supervised classification and anomaly detection approaches.

7️⃣ Next Steps

Perform exploratory data analysis (EDA) to visualize class distributions and feature correlations.

Identify and remove problematic features or records.

Prepare a cleaned dataset for baseline machine learning models.