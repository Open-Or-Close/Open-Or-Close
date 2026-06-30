
# Healthcare Data Analytics and Machine Learning for Cardiovascular Risk Assessment

# 1. Executive Summary

Cardiovascular disease is a major global health challenge, and early identification of patients at increased risk can support preventive healthcare strategies and clinical decision-making. This project applies a complete medical data science workflow to develop predictive machine learning models for heart disease classification using patient clinical information.

The study analysed a structured healthcare dataset containing demographic, physiological, and diagnostic variables. The complete analytical workflow included:

- Data quality assessment
- Exploratory data analysis (EDA)
- Statistical analysis (hypothesis test)
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
- ** Duplicates were identified and removed prior to model training to prevent data leakage and inflated performance estimates. **
- 
The predictor variables represent patient characteristics and clinical measurements used to estimate the likelihood of heart disease.


![Dataset Overview](https://github.com/Open-Or-Close/Open-Or-Close/blob/59d5b7c3e5b59eb1940b398dd36a66570d1ce6fa/Heart-Disease-Clinical-Analytics/Data/Dataset.jpg)


# 4. Data Quality Assessment

- Missing Data
A complete missing value assessment was performed.
Result:
-	No missing values were identified.
Therefore, no imputation strategy was required.
-   Duplicate Detection
Duplicate records were identified and removed.
This step was important because repeated observations can artificially increase model performance and reduce reliability.
After cleaning:
-	Dataset contained unique patient observations.


# 5. Exploratory Data Analysis
Exploratory analysis was performed to understand patient characteristics and identify relationships between clinical variables.
The analysis included:
-	Feature distributions
-	Target class distribution
-	Correlation analysis
-	Outlier detection
The target variable distribution was examined to ensure appropriate model evaluation.
A stratified train-test split was applied to maintain similar disease/non-disease proportions between training and testing datasets.



# 6. Correlation Analysis
Correlation analysis was conducted for numerical variables.
The purpose was to:
•	Identify relationships between clinical measurements
•	Detect possible redundancy between variables
•	Understand cardiovascular risk patterns
The analysis included:
•	Age
•	Blood pressure
•	Cholesterol
•	Maximum heart rate
•	ST depression

<img width="975" height="531" alt="image" src="https://github.com/user-attachments/assets/d83415e1-5b7e-4fed-8957-76c6069483fa" />

Figure.2-Correlation among the numerical variables 

# 7. Statistical Analysis
Statistical hypothesis testing was performed to evaluate relationships between clinical variables and heart disease outcomes.
### Numerical Variables
Independent t-tests were used.
Hypothesis:
	H0: There is no difference between patients with and without heart disease.
	H1: The clinical feature differs significantly between groups.
Significance level: α = 0.05

Table 1- P-value of each categorical variable associated with heart disease in the dataset

<img width="1033" height="213" alt="image" src="https://github.com/user-attachments/assets/5c887f58-0a55-42b8-9d7b-fa5fefcf4066" />

Table 1 presents the results of the t-test, indicating which numerical variables are significantly associated with heart disease and which are not. The significance threshold for this study was set at 0.05. After conducting an independent t-test, the results showed that cholesterol levels were not significantly associated with heart disease in this dataset, with a p-value of 0.1553. In contrast, age was found to be statistically significantly associated with heart disease because the mean age differed significantly between the two groups. Since the p-value was less than 0.05 (p < 0.05), the null hypothesis was rejected.
Although age is a well-established cardiovascular risk factor, this dataset does not fully demonstrate the expected population-level pattern. This may be because the dataset consists of selected patients who have already undergone clinical evaluation for cardiac disease rather than representing a general population sample.

### Categorical Variables
Chi-square tests were applied to determine whether categorical variables were associated with heart disease.
This approach evaluates whether variables such as chest pain type or exercise-induced angina are statistically related to disease status.
The hypothesis tested was:
	H₀ (Null hypothesis):
There is no association between the categorical variable and heart disease status.
	H₁ (Alternative hypothesis):
The categorical variable is associated with heart disease status.

Significance level: α=0.05

Table 2- P-value of each categorical variable associated with heart disease in the dataset
<img width="975" height="299" alt="image" src="https://github.com/user-attachments/assets/fdde56a4-9d96-4f4c-bcc1-28a5698ce584" />


The p-values of the variables presented in Table 2 are less than 0.05, indicating that these variables are statistically significantly associated with heart disease in this dataset. For example, sex is significantly associated with heart disease because the distribution of heart disease cases differs substantially between males and females. In contrast, fasting blood sugar is not significantly associated with heart disease in this dataset, as its p-value is greater than 0.05. This suggests that sex may represent an important cardiovascular risk factor, as biological, hormonal, and lifestyle-related differences between males and females can influence cardiovascular risk.




# 8. Data Preprocessing
Feature Scaling
Standardisation was applied for Logistic Regression.
The transformation ensured:
•	Mean = 0
•	Standard deviation = 1
Scaling was applied because Logistic Regression is sensitive to feature magnitude.
Random Forest was trained using the original feature values because tree-based algorithms do not require scaling.


# 9. Machine Learning Models

### 9.1 Logistic Regression
Logistic Regression was selected because it provides:
•	Interpretability
•	Probability-based predictions
•	Clinical transparency
This is valuable in healthcare where understanding decision factors is important.
________________________________________
### 9.2 Random Forest
Random Forest was selected because it:
•	Captures non-linear patterns
•	Models interactions between variables
•	Provides feature importance ranking


# 10. Model Evaluation Results
Independent Test Set Performance
Model	Accuracy	Precision	Recall	F1-score	ROC-AUC
Logistic Regression	80.3%	80.0%	84.8%	82.4%	0.871
Random Forest	75.4%	76.5%	78.8%	77.6%	0.861


# 11. Confusion Matrix Analysis

### Logistic Regression
The model correctly identified 28 out of 33 disease cases.
The false negative rate was relatively low, which is important in healthcare because missed disease cases may delay treatment.

<img width="1010" height="614" alt="image" src="https://github.com/user-attachments/assets/aac14b53-d3db-42c5-af55-d6044e2aae92" />

Figure.3-Confusion matrix for Logistic Regression

### Random Forest
Random Forest also demonstrated good disease detection capability.

<img width="975" height="589" alt="image" src="https://github.com/user-attachments/assets/8bac1254-270c-4a2a-9110-ef800ade1f71" />


Figure.4-Confusion matrix for Random Forest



# 12. Cross-Validation Results
    
Five-fold cross-validation was performed using ROC-AUC.
Model	Mean ROC-AUC
Logistic Regression	0.894
Random Forest	0.911
Random Forest achieved a higher cross-validation score, suggesting stronger predictive stability across different data splits. Furthermore, Random Forest achieved a higher cross-validation ROC-AUC compared with Logistic Regression, indicating superior discriminative performance. These results demonstrate that Random Forest can capture patterns within the dataset more effectively and maintain consistent performance across different subsets of the data. The strong cross-validation performance suggests that the learned patterns are reproducible and that Random Forest has stronger generalisation ability compared with Logistic Regression.



# 13. Hyperparameter Optimisation
GridSearchCV was used to identify optimal model configurations.
### Logistic Regression
Best parameters:
Parameter	Value
C	0.1
Maximum iterations	2000

Best cross-validation accuracy:
82.6%
________________________________________
### Random Forest
Best parameters:
Parameter	Value
Number of trees	200
Maximum depth	3

Best cross-validation accuracy:
85.5%


# 14. Feature Importance Analysis
Random Forest feature importance identified the strongest contributors to heart disease prediction.
Rank	Feature	Importance
1	Chest pain type (cp)	0.174
2	Maximum heart rate (thalach)	0.132
3	Number of vessels (ca)	0.106
4	ST depression (oldpeak)	0.097
5	Thalassemia	0.090
6	Age	0.083
The results indicate that clinical indicators related to cardiac symptoms, exercise response, and physiological measurements contributed strongly to prediction.


# 15. Clinical Interpretation
The developed machine learning models demonstrate the potential of artificial intelligence for cardiovascular risk prediction.
Key findings:
•	Logistic Regression provided strong predictive performance with high interpretability.
•	Random Forest provided strong generalisation capability through cross-validation.
•	Clinical variables related to chest pain, heart rate response, and exercise-related measurements were important predictors.
These models could potentially support clinicians by providing additional risk information during patient assessment.
However, machine learning predictions should be considered decision-support tools rather than replacements for clinical judgement.
________________________________________
# 16. Limitations
Several limitations should be considered:
1.	The dataset size is relatively limited.
2.	The study requires external validation using independent clinical populations.
3.	Dataset characteristics may not represent all demographic groups.
4.	Before clinical deployment, additional evaluation is required, including:
o	Model calibration
o	Bias assessment
o	Prospective validation
________________________________________
# 17. Conclusion
This project demonstrates a complete medical data science workflow for heart disease prediction.
The analysis successfully integrated:
•	Healthcare data cleaning
•	Statistical investigation
•	Machine learning modelling
•	Model validation
•	Explainability analysis
Both Logistic Regression and Random Forest demonstrated the ability to predict heart disease from patient clinical information.
Future work should focus on external clinical validation, explainable AI methods, and integration into clinical decision-support systems.


