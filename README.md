# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn (cancel their service), using account details, subscribed services, and billing information. Built end-to-end — from data cleaning to a deployed, interactive web app.



## Problem Statement

Telecom companies lose a significant portion of customers every year, and acquiring a new customer costs 5–25x more than retaining an existing one. This project builds a model that flags customers likely to churn *before* they leave, so a retention team can intervene early with targeted offers or support.

This is framed as a **binary classification** problem: given a customer's profile, predict `Churn` (Yes/No) along with a probability score.

---

## Dataset

- **Source:** [Telco Customer Churn Dataset (Kaggle / IBM Sample Data)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 rows, 21 columns
- **Target:** Churn (Yes/No) — ~26% churn rate (imbalanced)
- **Features:** Customer demographics, account details (tenure, contract type), subscribed services (internet, phone, streaming, security add-ons), and billing information

---

## Approach

1. **Data Cleaning** — Fixed TotalCharges type issue, handled missing values tied to new customers, simplified redundant categorical labels
2. **EDA** — Explored churn patterns across tenure, contract type, internet service, and payment method
3. **Feature Engineering** — Encoded categorical variables, addressed class imbalance using both **SMOTE** and **class-weighting**
4. **Modeling** — Compared Logistic Regression, Random Forest, and XGBoost, each trained with both imbalance-handling strategies (6 models total)
5. **Evaluation** — Prioritized **Recall** and **F1-score** over accuracy, since missing an actual churner is more costly to the business than a false alarm
6. **Hyperparameter Tuning** — GridSearchCV with 5-fold cross-validation, optimizing for recall
7. **Model Interpretation** — Used both Logistic Regression coefficients and **SHAP values** to explain which features drive predictions
8. **Deployment** — Built an interactive Streamlit app and deployed it on Hugging Face Spaces

---

## Results

**Best Model: Logistic Regression (`class_weight='balanced'`)**

| Metric      | Score  |
|-------------|--------|
| Recall      | 0.789  |
| Precision   | 0.504  |
| F1-Score    | 0.615  |
| ROC-AUC     | 0.842  |

Logistic Regression outperformed both Random Forest and XGBoost on recall and ROC-AUC — across all model types, class-weighting consistently beat SMOTE for this dataset.

### Key Churn Drivers (from SHAP & coefficients)
- **Contract type** — month-to-month customers churn far more than those on 1-year/2-year contracts
- **Tenure** — churn is heavily concentrated among newer customers
- **Internet Service** — fiber optic customers show higher churn than DSL
- **Payment Method** — electronic check users churn more than automatic payment methods

---

## Tech Stack

- **Language:** Python
- **Data Handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Interpretability:** SHAP
- **Deployment:** Streamlit, Hugging Face Spaces

---

## Repository Structure

```
├── Customer_Churn_Prediction.ipynb         # Full analysis: EDA, modeling, evaluation, interpretation
├── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Dataset
├── streamlit_app.py                        # Deployed web app
├── churn_model.pkl                         # Trained model
├── scaler.pkl                              # Fitted StandardScaler
├── model_columns.pkl                       # Column structure used for inference
├── requirements.txt                        # Dependencies
└── README.md
```

## Author 
**Nikhil Kumar**
