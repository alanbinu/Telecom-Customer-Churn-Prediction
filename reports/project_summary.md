# Project Summary — Telecom Customer Churn Prediction

## Objective

Predict which telecom customers are likely to churn, using service, billing,
and satisfaction signals, to support proactive retention decisions instead of
reactive, post-hoc revenue reporting.

## Dataset

- 74,983 customer records, 42 features (29 numerical, 11 categorical, plus
  a dropped identifier column).
- Target: `Churn_Flag` — 80.05% retained / 19.95% churned.
- 538 duplicate rows detected, not removed (see reports/model_summary.md,
  Data Notes).

## Pipeline

1. Load raw data
2. Drop `Customer_ID`
3. Impute missing values (median / mode)
4. One-hot encode categorical features (`drop_first=True`)
5. Train/test split (70/30, `random_state=42`)
6. Scale features (`StandardScaler`, fit on train only)
7. Train six classifiers: Logistic Regression, Decision Tree, Random Forest,
   KNN, SVM, Naive Bayes
8. Evaluate and compare on test-set accuracy
9. Select Random Forest as the production model
10. Deploy via an interactive Streamlit application

## Result

Random Forest reaches **87.56% test accuracy** and **0.9084 ROC-AUC**, with
77% precision and 54% recall on the churn class. See `model_summary.md` for
the full breakdown and an honest accounting of where the model currently
underperforms (churn-class recall).

## Business Insights

- Service experience (complaints, satisfaction, NPS, support-call volume)
  outweighs billing and plan-type variables as a churn driver.
- Network reliability (dropped calls, outages) has a measurable, quantified
  effect on churn — this gives infrastructure spend a retention-ROI argument.
- The model's precision profile supports targeted retention outreach rather
  than blanket discount campaigns.

## Next Steps

See the README's Future Improvements section — the highest-leverage next step
is addressing churn-class recall (54%) through class-imbalance handling
(SMOTE / class-weighting), not further hyperparameter tuning on the current
class balance.
