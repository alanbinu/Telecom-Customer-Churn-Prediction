# 🤖 Trained Models

This folder contains the machine learning artifacts used in the **Telecom Customer Churn Prediction** project.

---

## 📁 Available Files

| File | Description |
|------|-------------|
| `Scaler.pkl` | StandardScaler used to normalize numerical features before model prediction. |

---

## 🏆 Final Model

Multiple machine learning algorithms were trained and evaluated to identify the best performing model.

### Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest
- K Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Naive Bayes

After comparing performance metrics, the **Random Forest Classifier** was selected as the final model.

### Performance

| Metric | Value |
|---------|------:|
| Accuracy | **86.77%** |
| Selected Model | Random Forest Classifier |

---

## 🔄 Model Development Pipeline

```text
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Feature Engineering
   │
   ▼
Feature Scaling (StandardScaler)
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Customer Churn Prediction
```

---

## ⚠️ Model Availability

The trained Random Forest model (`telecom_churn_Project.pkl`) is approximately **75 MB**, which exceeds GitHub's web upload limit.

The model can be regenerated at any time by executing the notebook located in the **notebooks** folder.

---

## 📌 Notes

- The `Scaler.pkl` file is included because it is required during prediction.
- The complete training pipeline is fully documented in the Jupyter Notebook.
- All preprocessing, feature engineering, model training, and evaluation steps can be reproduced using the notebook provided in this repository.
