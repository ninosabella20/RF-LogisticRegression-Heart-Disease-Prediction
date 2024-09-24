# Heart Disease Detection using Machine Learning (R)

## Project Overview

This project aims to predict the risk of heart disease using a dataset from Kaggle. We utilize machine learning algorithms, specifically Logistic Regression and Random Forest, to analyze various patient attributes and determine their likelihood of having heart disease.

## Dataset

The dataset consists of the following features:

- **age**: Age of the patient
- **sex**: Gender of the patient (1 = male; 0 = female)
- **chest pain type**: Type of chest pain (4 values)
- **resting blood pressure**: Blood pressure in mm Hg
- **serum cholesterol**: Cholesterol level in mg/dl
- **fasting blood sugar**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- **resting electrocardiographic results**: Results of electrocardiographic test (values 0, 1, 2)
- **maximum heart rate achieved**: Maximum heart rate achieved
- **exercise induced angina**: Exercise induced angina (1 = yes; 0 = no)
- **old peak**: ST depression induced by exercise relative to rest
- **slope of peak exercise**: Slope of the peak exercise ST segment (values 0, 1, 2)
- **number of major vessels**: Number of major vessels (0-3) colored by fluoroscopy
- **thal**: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)

**[Dataset Link](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)**

## Data Handling

- **Renamed Columns**: Improved readability of column names.
- **Checked for Missing Values**: Ensured data quality and completeness.

## Model Development

### Logistic Regression

- Created a Logistic Regression model to predict heart disease risk.
- **Model Evaluation**:
  - **Accuracy**: 85%
  - **Confusion Matrix**:
    - True Negatives (0,0): 123
    - False Negatives (1,0): 18
    - False Positives (0,1): 27
    - True Positives (1,1): 140
  - **ROC AUC**: 0.94

**Analysis**: 
- The model identified 18 patients at risk who were not flagged, and 27 patients not at risk who were incorrectly identified as at risk.

### Model Tuning

- Adjusted the probability threshold for predicting at-risk patients from 0.5 to 0.25.
- Resulted in:
  - **New Accuracy**: 86%
  - **Confusion Matrix**: Improved detection of at-risk patients, reducing false negatives to 7 and increasing false positives to 36.

### Random Forest Model

- Developed a Random Forest model with 100 trees.
- **Model Evaluation**:
  - **Accuracy**: 98%
  - **Confusion Matrix**:
    - True Negatives: 150
    - True Positives: 152
    - False Positives: 0
    - False Negatives: 6
  - **ROC AUC**: 1

**Analysis**: 
- The Random Forest model achieved high accuracy with no false positives, indicating excellent performance.

## Conclusion

The Logistic Regression and Random Forest models demonstrated effective predictions for heart disease risk. The tuning of prediction thresholds improved the model’s reliability, highlighting the importance of evaluating and adjusting model parameters for optimal performance.

## Future Work

- Explore additional machine learning models and ensemble methods to enhance predictive performance.
- Investigate feature importance and selection to understand the impact of each attribute on heart disease risk.
