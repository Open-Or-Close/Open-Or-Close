
# Healthcare Data Analytics and Machine Learning for Cardiovascular Risk Assessment

# 1. Executive Summary

Cardiovascular disease is a major global health challenge, and early identification of patients at increased risk can support preventive healthcare strategies and clinical decision-making. This project applies a complete medical data science workflow to develop predictive machine learning models for heart disease classification using patient clinical information.

The study analysed a structured healthcare dataset containing demographic, physiological, and diagnostic variables. The complete analytical workflow included:

- Data quality assessment
- Exploratory data analysis (EDA)
- Statistical analysis
- Feature preprocessing
- Machine learning model development
- Model validation
- Hyperparameter optimisation
- Feature importance analysis

Two supervised machine learning approaches were developed and evaluated:

### 1. Logistic Regression
- Selected due to its high interpretability and suitability for clinical applications.
- Provides insight into the relationship between individual clinical variables and heart disease risk.

### 2. Random Forest Classifier
- Selected due to its ability to capture non-linear relationships and complex interactions between clinical variables.
- Provides feature importance measurements to identify the most influential predictors.

Model performance was evaluated using multiple metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix analysis
- Five-fold cross-validation

The results demonstrated that both models successfully identified patterns associated with heart disease. Logistic Regression achieved stronger performance on the independent test dataset, while Random Forest demonstrated a higher cross-validation ROC-AUC score, indicating strong generalisation capability and robustness across different data subsets.

---

# 2. Clinical Objective

The objective of this project was to develop machine learning models capable of predicting the presence of heart disease based on patient clinical characteristics.

The target variable represented heart disease status:

- **1 = Heart disease present**
- **0 = Heart disease absent**

The primary aim was to investigate whether machine learning approaches could support cardiovascular risk assessment by identifying important clinical predictors associated with heart disease.

---

# 3. Dataset Description

The dataset consisted of clinical records containing demographic, physiological, and diagnostic information related to cardiovascular health.

| Dataset Characteristic | Value |
|---|---:|
| Number of records | 1025 |
| Number of variables | 14 |
| Target variable | Heart disease status |
| Missing values | 0 |
| Duplicate records | 723 |

The dataset contained **14 variables**, including:

- **13 input features (predictor variables)**
- **1 target variable (heart disease outcome)**

The predictor variables represent patient characteristics and clinical measurements used to estimate the likelihood of heart disease.


This project demonstrates an end-to-end medical data science workflow for cardiovascular risk assessment using patient clinical data. The analysis includes healthcare data quality assessment, exploratory analysis, statistical testing, predictive modelling, validation, and interpretation of clinically relevant predictors.

The objective is to demonstrate how healthcare data can be analysed to identify risk patterns and support evidence-based decision-making.

Machine learning models can support identification of potential risk patterns; however, they are not intended to replace clinical judgement. Further validation using larger and diverse clinical populations would be required before deployment.


![Alt Image](https://github.com/Open-Or-Close/Open-Or-Close/blob/59d5b7c3e5b59eb1940b398dd36a66570d1ce6fa/Heart-Disease-Clinical-Analytics/Data/Dataset.jpg)
