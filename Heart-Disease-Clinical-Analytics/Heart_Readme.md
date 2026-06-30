
# Medical Data Science Workflow for Cardiovascular Risk Prediction Using Clinical Patient Data

# 1. Executive Summary

Cardiovascular disease is a major global health challenge, and early identification of patients at increased risk can support preventive healthcare strategies and clinical decision-making. This project applies a complete medical data science workflow to develop predictive analytics models to analyse clinical risk factors, identify important patient characteristics, and support evidence-based cardiovascular risk assessment.

The study analysed a structured healthcare dataset containing demographic, physiological, and diagnostic variables. The complete analytical workflow included:

1. Clinical dataset assessment
2. Data quality review and validation
3. Exploratory analysis of patient characteristics
4. Statistical evaluation of clinical variables
5. Predictive model development
6. Model performance assessment
7. Feature importance and clinical interpretation
8. Identification of limitations and validation requirements

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



# 2. Clinical Objective

The objective of this project was to develop machine learning models capable of predicting the presence of heart disease based on patient clinical characteristics.

The target variable represented heart disease status:

- **1 = Heart disease present**
- **0 = Heart disease absent**

The primary aim was to investigate whether machine learning approaches could support cardiovascular risk assessment by identifying important clinical predictors associated with heart disease.



# 3. Dataset Description

The dataset consisted of clinical records containing demographic, physiological, and diagnostic information related to cardiovascular health.

| Dataset Characteristic | Value |
|---|---:|
| Number of records | 1025 |
| Number of variables | 14 |
| Target variable | Heart disease status |
| Missing values | 0 |
| Duplicate observations identified during quality assessment | 723
  Duplicate observations removed before modelling to reduce risk of biased performance estimates.

The dataset contained **14 variables**, including:

- **13 input features (predictor variables)**
- **1 target variable (heart disease outcome)**

  
The predictor variables represent patient characteristics and clinical measurements used to estimate the likelihood of heart disease.


### Dataset Variables Description

| Variable | Type | Meaning |
|---|---|---|
| Age | Numerical | Patient age in years. Age is an important cardiovascular risk factor because heart disease risk generally increases with age. |
| Sex | Categorical | Patient sex (typically coded as male/female). Sex-related biological and lifestyle differences can influence cardiovascular risk. |
| Chest Pain Type (cp) | Categorical | Type of chest pain experienced by the patient. Different chest pain patterns can indicate different levels of cardiac risk. |
| Resting Blood Pressure (trestbps) | Numerical | Resting blood pressure measured in mm Hg. High blood pressure is a major cardiovascular risk factor. |
| Cholesterol (chol) | Numerical | Serum cholesterol level measured in mg/dL. High cholesterol can contribute to atherosclerosis and cardiovascular risk. |
| Fasting Blood Sugar (fbs) | Categorical | Indicates whether fasting blood sugar is greater than 120 mg/dL (1 = yes, 0 = no). High blood sugar may indicate diabetes-related cardiovascular risk. |
| Resting ECG Results (restecg) | Categorical | Results of the resting electrocardiogram (ECG), showing possible abnormalities in heart electrical activity. |
| Maximum Heart Rate Achieved (thalach) | Numerical | The maximum heart rate achieved during exercise testing. Lower maximum heart rate may indicate reduced cardiac function. |
| Exercise-Induced Angina (exang) | Categorical | Indicates whether exercise causes chest pain (angina). Exercise-induced symptoms can suggest underlying coronary artery disease. |
| ST Depression (oldpeak) | Numerical | Measures ST-segment depression induced by exercise compared with rest. It is an indicator of possible myocardial ischemia. |
| Slope of Peak Exercise ST Segment (slope) | Categorical | Describes the slope pattern of the ST segment during peak exercise, which provides information about cardiac stress response. |
| Number of Major Vessels (ca) | Numerical/Categorical | Number of major blood vessels visualised by fluoroscopy (0–3). Higher values may indicate greater coronary artery involvement. |
| Thalassemia (thal) | Categorical | A blood disorder-related feature recorded in the dataset, representing different thalassemia states associated with heart risk assessment. |
| Heart Disease (target) | Binary Target Variable | Indicates whether the patient has heart disease (1 = presence of disease, 0 = absence of disease). This is the outcome predicted by the machine learning models. |


# 4. Data Quality Assessment

### Missing Data

A complete missing value assessment was performed.

Result:
- No missing values were identified.
- Therefore, no imputation strategy was required.



### Duplicate Detection

Duplicate records were identified and removed.

This step was important because repeated observations can artificially inflate model performance and reduce reliability.

After cleaning:
- The dataset contained only unique patient observations.


# 5. Exploratory Data Analysis

Exploratory analysis was performed to understand patient characteristics and identify relationships between clinical variables.

The analysis included:

- Feature distributions (e.g., Cholesterol levels)
- Target class distribution
- Correlation analysis
- Outlier detection

The target variable distribution was examined to ensure appropriate and unbiased model evaluation.

A stratified train-test split was applied to maintain consistent proportions of disease and non-disease cases across both training and testing datasets.


<img width="975" height="517" alt="image" src="https://github.com/user-attachments/assets/43a161d9-bede-4a47-b578-9ef469983489" />

                        
<p align="center">
  <em>Figure.1: Distribution of Cholesterol (Chol). The cholesterol distribution shows that most patients have cholesterol values concentrated around the middle range, while fewer patients have very high cholesterol levels. The distribution indicates variability in cholesterol levels across the study population, with some extreme values suggesting potential outliers.</em>
</p>

### Outlier Detection
In many machine learning projects, outliers are removed because they may result from measurement errors, data entry mistakes, or sensor-related problems. However, in healthcare datasets, extreme values may represent genuine physiological conditions rather than errors. For example, a high cholesterol value may indicate:

- A patient with dyslipidaemia
- Increased cardiovascular risk

Removing such values could eliminate clinically meaningful patient information from the dataset. Therefore, in this clinical project, outliers were not removed because they may represent important variations in patient health characteristics.




# 6. Correlation Analysis


### Correlation Analysis

Correlation analysis was conducted for numerical variables.

**Objectives:**
- Identify relationships between clinical measurements
- Detect possible redundancy between variables
- Understand cardiovascular risk patterns

**Variables analysed:**
- Age
- Blood pressure
- Cholesterol
- Maximum heart rate
- ST depression



<img width="975" height="531" alt="image" src="https://github.com/user-attachments/assets/d83415e1-5b7e-4fed-8957-76c6069483fa" />

                            

<p align="center">
  <em>Figure.2: Correlation among the numerical variables.</em>
</p>


# 7. Statistical Analysis
Statistical hypothesis testing was performed to evaluate relationships between clinical variables and heart disease outcomes.

### T-test (Numerical Variables)

Independent t-tests were used.

**Hypothesis:**

- **H0 (Null Hypothesis):** There is no difference between patients with and without heart disease.
- **H1 (Alternative Hypothesis):** The clinical feature differs significantly between the two groups.

**Significance level:**
- α = 0.05

Table 1- P-value of each categorical variable associated with heart disease in the dataset

<img width="1033" height="213" alt="image" src="https://github.com/user-attachments/assets/5c887f58-0a55-42b8-9d7b-fa5fefcf4066" />

Table 1 presents the results of the t-test, indicating which numerical variables are significantly associated with heart disease and which are not. The significance threshold for this study was set at 0.05. After conducting an independent t-test, the results showed that cholesterol levels were not significantly associated with heart disease in this dataset, with a p-value of 0.1553. In contrast, age was found to be statistically significantly associated with heart disease because the mean age differed significantly between the two groups. Since the p-value was less than 0.05 (p < 0.05), the null hypothesis was rejected.
Although age is a well-established cardiovascular risk factor, this dataset does not fully demonstrate the expected population-level pattern. This may be because the dataset consists of selected patients who have already undergone clinical evaluation for cardiac disease rather than representing a general population sample.


### Chi-Square Test (Categorical Variables)

Chi-square tests were applied to determine whether categorical variables were associated with heart disease.

This approach evaluates whether variables such as chest pain type or exercise-induced angina are statistically related to disease status.

**Hypothesis:**

- **H₀ (Null hypothesis):** There is no association between the categorical variable and heart disease status.
- **H₁ (Alternative hypothesis):** The categorical variable is associated with heart disease status.

**Significance level:**
- α = 0.05


Table 2- P-value of each categorical variable associated with heart disease in the dataset
<img width="975" height="299" alt="image" src="https://github.com/user-attachments/assets/fdde56a4-9d96-4f4c-bcc1-28a5698ce584" />


The p-values of the variables presented in Table 2 are less than 0.05, indicating that these variables are statistically significantly associated with heart disease in this dataset. For example, sex is significantly associated with heart disease because the distribution of heart disease cases differs substantially between males and females. In contrast, fasting blood sugar is not significantly associated with heart disease in this dataset, as its p-value is greater than 0.05. This suggests that sex may represent an important cardiovascular risk factor, as biological, hormonal, and lifestyle-related differences between males and females can influence cardiovascular risk.




# 8. Data Preprocessing
Feature Scaling
Standardisation was applied for Logistic Regression.
The transformation ensured:
-	Mean = 0
-	Standard deviation = 1
Scaling was applied because Logistic Regression is sensitive to feature magnitude.
Random Forest was trained using the original feature values because tree-based algorithms do not require scaling.


# 9. Machine Learning Models

### 9.1 Logistic Regression
Logistic Regression was selected because it provides:
-	Interpretability
-	Probability-based predictions
-	Clinical transparency
   This is valuable in healthcare where understanding decision factors is important.


### 9.2 Random Forest
Random Forest was selected because it:
-	Captures non-linear patterns
-	Models interactions between variables
-	Provides feature importance ranking


# 10. Model Evaluation Results

### Independent Test Set Performance

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 80.3% | 80.0% | 84.8% | 82.4% | 0.871 |
| Random Forest | 75.4% | 76.5% | 78.8% | 77.6% | 0.861 |

Logistic Regression achieved slightly higher performance on the independent test dataset, while maintaining strong recall and ROC-AUC. This highlights its effectiveness and interpretability for clinical prediction tasks.

# 11. Confusion Matrix Analysis

### Logistic Regression
The model correctly identified 28 out of 33 disease cases.
The false negative rate was relatively low, which is important in healthcare because missed disease cases may delay treatment.

<img width="1010" height="614" alt="image" src="https://github.com/user-attachments/assets/aac14b53-d3db-42c5-af55-d6044e2aae92" />


<p align="center">
  <em>Figure.3-Confusion matrix for Logistic Regression.</em>
</p>

### Confusion Matrix Interpretation (Logistic Regression)

The confusion matrix provides insight into the model’s ability to correctly identify patients with and without
**True Positive (TP)**
Patients who actually had heart disease and were correctly identified by Logistic Regression.
These are successful detections.
The model correctly recognised high-risk patients who may require:
-	Further cardiac investigation 
-	Preventive intervention 
-	Clinical follow-up 
A higher TP count contributes to higher sensitivity (recall).

**True Negative (TN)**
Patients without heart disease who were correctly classified as healthy.
The model successfully avoided unnecessary concern or additional testing for low-risk patients.
A good TN value indicates the model can distinguish healthy patients from diseased patients.

**False Positive (FP)**
Patients without heart disease who were incorrectly predicted as having disease.
These patients may experience:
-	Additional diagnostic testing 
-	Increased healthcare costs 
-	Possible anxiety 
However, in cardiovascular screening, false positives are generally considered less harmful than false negatives because further investigation can confirm the diagnosis.

**False Negative (FN)**
Patients who had heart disease but were incorrectly predicted as healthy.
This is the most clinically important error.
A false negative may result in:
-	Delayed diagnosis 
-	Missed treatment opportunity 
-	Increased cardiovascular risk 
Therefore, reducing FN cases is a priority in medical prediction models.

The confusion matrix demonstrated that Logistic Regression was able to correctly classify a substantial proportion of patients with and without heart disease. The model achieved a strong ability to identify disease cases, reflected by its sensitivity/recall performance. However, some false negative predictions remained, indicating that a proportion of patients with underlying disease may not be detected. In a clinical setting, further optimisation may focus on reducing false negatives because missed disease cases carry greater clinical consequences than unnecessary follow-up investigations.


### Random Forest
Random Forest also demonstrated good disease detection capability.

<img width="975" height="589" alt="image" src="https://github.com/user-attachments/assets/8bac1254-270c-4a2a-9110-ef800ade1f71" />




<p align="center">
  <em>Figure.4-Confusion matrix for Random Forest.</em>
</p>

### Confusion Matrix Interpretation (Random Forest)

The confusion matrix provides insight into the model’s ability to correctly identify patients with and without

**True Positive (TP)**
Patients with heart disease correctly identified by Random Forest.
The model successfully detected patients whose clinical profiles matched patterns associated with cardiovascular disease. This demonstrates that Random Forest can recognise complex combinations of risk factors.

**True Negative (TN)**
Patients without heart disease correctly classified. The model successfully identified lower-risk individuals. This reduces unnecessary clinical follow-up.

**False Positive (FP)**
Healthy patients incorrectly classified as having heart disease.
Although these predictions are incorrect, they may be acceptable in a screening context because they lead to further assessment rather than missed diagnosis.

**False Negative (FN)**
Patients with heart disease incorrectly classified as healthy. These are the most concerning predictions.
Possible consequences include:
-	Failure to identify cardiovascular risk 
-	Delayed intervention 
-	Missed opportunity for preventive care 
Therefore, recall remains a critical evaluation metric.

The Random Forest confusion matrix showed that the model successfully identified many patients with heart disease while maintaining reasonable classification of non-disease cases. The model's ability to capture complex interactions between clinical variables contributed to its predictive performance. However, false negative cases remain clinically important because missed cardiovascular disease detection may delay appropriate management. Therefore, model evaluation should prioritise sensitivity and risk stratification rather than accuracy alone



# 12. Cross-Validation Results
    
### Five-Fold Cross-Validation Performance (ROC-AUC)

| Model | Mean ROC-AUC |
|---|---|
| Logistic Regression | 0.894 |
| Random Forest | 0.911 |

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


<img width="1009" height="549" alt="image" src="https://github.com/user-attachments/assets/6765c11c-246c-4cf5-893a-81a0d577053a" />


<p align="center">
  <em>Figure.5-Feature importance captured by Random Forest.</em>
</p>

It is clear from Fig.5 that Chest pain type (cp) was one of the most influential predictors. The model learned that different chest pain patterns are associated with different probabilities of heart disease.


# 15. Clinical Interpretation
The developed machine learning models demonstrate the potential of artificial intelligence for cardiovascular risk prediction.
**Key findings:**
-	Logistic Regression provided strong predictive performance with high interpretability.
-	Random Forest provided strong generalisation capability through cross-validation.
-	Clinical variables related to chest pain, heart rate response, and exercise-related measurements were important predictors.
These models could potentially support clinicians by providing additional risk information during patient assessment.
However, machine learning predictions should be considered decision-support tools rather than replacements for clinical judgement.


# 16. Limitations
Several limitations should be considered:
1.	The dataset size is relatively limited.
2.	The study requires external validation using independent clinical populations.
3.	Dataset characteristics may not represent all demographic groups.
4.	Before clinical deployment, additional evaluation is required, including:
    -	Model calibration
    -	Bias assessment
    -	Prospective validation


# 17. Conclusion
This project demonstrates a complete medical data science workflow for heart disease prediction.
The analysis successfully integrated:
-	Healthcare data cleaning
-	Statistical investigation
-	Machine learning modelling
-	Model validation
-	Explainability analysis
Both Logistic Regression and Random Forest demonstrated the ability to predict heart disease from patient clinical information.
Future work should focus on external clinical validation, explainable AI methods, and integration into clinical decision-support systems.


# 18. Tools and Technologies

### Programming:
Python, Pandas, NumPy, Scikit-learn

### Statistical Analysis:
Hypothesis Testing, Chi-square Test, T-test, Correlation Analysis

### Machine Learning:
Logistic Regression, Random Forest, Cross-validation, Hyperparameter Optimisation

### Visualisation:
Matplotlib, Seaborn

### Documentation:
GitHub, Markdown


