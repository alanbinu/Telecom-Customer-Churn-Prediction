# Model Summary — Telecom Customer Churn Prediction

## Selected Model: Random Forest Classifier

| Metric | Value |
|---|---|
| Test Accuracy | 87.5617% |
| ROC-AUC | 0.9084 |
| Precision (Churn class) | 0.7675 |
| Recall (Churn class) | 0.5436 |
| F1-Score (Churn class) | 0.6364 |
| Precision (No-Churn class) | 0.8935 |
| Recall (No-Churn class) | 0.9588 |
| F1-Score (No-Churn class) | 0.9250 |
| Weighted Avg F1 | 0.8672 |
| Macro Avg F1 | 0.7807 |

## All Models Evaluated

| Model | Test Accuracy |
|---|---|
| **Random Forest** | **87.56%** |
| Logistic Regression | 87.42% |
| SVM | 87.25% |
| KNN | 83.89% |
| Decision Tree | 81.72% |
| Naive Bayes | 77.56% |

## Why Random Forest Was Selected

Random Forest, Logistic Regression, and SVM are within a 0.31-point accuracy band —
accuracy alone does not separate them meaningfully. Random Forest was selected because:

- It provides a native, interpretable feature-importance ranking that a linear
  model's coefficients do not communicate as intuitively to a non-technical
  business audience.
- Its precision/recall balance on the minority (churn) class is usable for
  targeted retention outreach without excessive false positives.

## Known Limitation

Recall on the churn class is 54.36% — the model misses roughly 46 of every 100
customers who actually churn. This is the model's most honest weakness and the
clearest target for improvement (see Future Improvements: class-imbalance
handling via SMOTE or class-weighting).

## Top 10 Feature Importances

| Rank | Feature | Importance |
|---|---|---|
| 1 | Complaint_Count | 0.1141 |
| 2 | Satisfaction_Score | 0.0853 |
| 3 | NPS_Score | 0.0815 |
| 4 | Customer_Service_Calls | 0.0529 |
| 5 | Dropped_Call_Rate | 0.0365 |
| 6 | Reward_Points | 0.0313 |
| 7 | Discount_Received | 0.0312 |
| 8 | Customer_Lifetime_Value | 0.0293 |
| 9 | Total_Bill_Amount | 0.0291 |
| 10 | Network_Outages_Experienced | 0.0284 |

## Data Notes

- 538 duplicate rows were detected in the raw dataset (`df.duplicated().sum()`)
  but were never removed in the training pipeline — the deployed model was
  trained on data that still contains them. This is preserved intentionally
  per the constraint not to retrain or change preprocessing; flagged here as
  a known gap for a future retraining pass, not a claim that it's been handled.
- Missing values were imputed (median for numerical, mode for categorical),
  not dropped.
