# Credit Card Fraud Detection Using Deep Learning and Graph Neural Networks (GNN)

A robust credit card fraud detection system built using **Graph Neural Networks (GNNs)** and **deep learning** techniques. This project leverages graph-based modeling to identify hidden relationships between transactions, enabling more accurate detection of fraudulent activity in highly imbalanced datasets.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Objective](#objective)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Future Work](#future-work)
- [License](#license)


---

## 🧠 Project Overview

Traditional machine learning models often fail to detect sophisticated fraud patterns due to their inability to capture relationships between data points. This project transforms transactional data into a **graph-based structure**, enabling **Graph Convolutional Networks (GCNs)** to analyze structural similarities among transactions.

---

## 🎯 Objective

To design and implement a deep learning model capable of:
- Detecting fraudulent credit card transactions.
- Modeling transaction relationships using graphs.
- Handling class imbalance effectively.

---

## 🔍 Methodology

### 1. **Data Preprocessing**
- Used a real-world credit card transaction dataset.
- Performed **stratified sampling** to select 10,000 records while maintaining fraud ratio.
- Applied **feature scaling** using `StandardScaler`.

### 2. **Graph Construction**
- Constructed graphs using **K-Nearest Neighbors (KNN)** with **cosine similarity**.
- Each transaction is a **node**, with edges connecting similar transactions.

### 3. **Model Architecture**
- Implemented a **Graph Convolutional Network (GCN)** using **PyTorch Geometric**:
  - Two GCN layers with **ReLU** activation.
  - A final **Softmax** layer for binary classification.

### 4. **Training & Evaluation**
- **Train/Test split**: 80/20
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - Confusion Matrix & Classification Report

---

## 📊 Results

The GCN-based model delivered strong performance in detecting credit card fraud. Here are the highlights from training and evaluation:

- **Test Accuracy by Epochs**:  
  - Epoch 20: 95.95%  
  - Epoch 100: 99.65%  
  - Epoch 200: **99.75%**

- **Classification Report**:
  ```
               precision    recall  f1-score   support
    accuracy                         0.9975      2000
  ```

- **ROC-AUC Score**: **1.0000**

These metrics indicate excellent generalization to unseen data and confirm the model's effectiveness in identifying both common and subtle fraud patterns, even under class imbalance.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Aparna-analyst/Credit-card-fraud-detection.git
cd Credit-card-fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
----



## 🔮 Future Work

- Explore **semi-supervised GNNs**
- Apply **dynamic graph learning** for real-time fraud detection
- Integrate with live financial transaction APIs

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.


