# 🤖 Trained Models

This folder contains the machine learning artifacts used in the Telecom Customer Churn Prediction project.

---

## 📁 Files

| File | Description |
|------|-------------|
| `Scaler.pkl` | StandardScaler used to normalize numerical features before prediction. |

---

## 📊 Model Information

The project was trained using multiple machine learning algorithms, including:

- Logistic Regression
- Decision Tree
- Random Forest
- K Nearest Neighbors
- Support Vector Machine
- Naive Bayes

After evaluation, the **Random Forest Classifier** achieved the best overall performance and was selected as the final model.

**Final Model Accuracy:** **86.77%**

---

## ⚠️ Why isn't the trained model included?

The trained Random Forest model (`telecom_churn_Project.pkl`) is approximately **75 MB**, which exceeds GitHub's web upload limit.

The model can be regenerated at any time by running the notebook available in the **notebooks** folder.

---

## 🔄 Model Pipeline

Dataset

⬇️

Data Cleaning

⬇️

Feature Engineering

⬇️

Feature Scaling (StandardScaler)

⬇️

Model Training

⬇️

Model Evaluation

⬇️

Customer Churn Prediction

---

## 📌 Note

This repository focuses on demonstrating the complete machine learning workflow. The preprocessing scaler is included because it is required during prediction, while the trained model can be recreated using the provided notebook.
