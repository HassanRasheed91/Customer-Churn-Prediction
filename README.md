
# Bank Customer Churn Prediction

This project aims to predict customer churn for a bank using machine learning models. The dataset contains information about the bank's customers and various features related to their transactions, demographics, and account activity. The main objective is to build and tune machine learning models to accurately predict whether a customer will churn or not.

## Highlights
- Large-scale dataset: 355,190 records × 116 features
- Extensive feature selection using correlation, SHAP, and LIME
- Trained Logistic Regression & SVM with hyperparameter tuning (GridSearchCV)
- Deployed with Flask + Gunicorn + Streamlit UI for real-time predictions

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Model Training and Tuning](#model-training-and-tuning)
6. [Model Evaluation](#model-evaluation)
7. [Interpretability](#interpretability)
8. [Deployment](#deployment)
9. [Usage](#usage)
10. [Results](#results)
11. [Conclusion](#conclusion)

## Introduction
Customer churn is a critical issue for banks, as retaining existing customers is often more cost-effective than acquiring new ones. This project leverages machine learning to predict which customers are likely to churn based on their historical data and behavior patterns.

## Dataset
- Records: 355,190
- Features: 116
- Target variable: `TARGET` → 1 (churned), 0 (retained)
- Data includes: Demographics, product usage, account activity, and more

## Data Preprocessing.
- Handled missing values and duplicates
- One-hot encoded categorical variables
- Normalized numerical columns
- Split into training and test sets

## Feature Engineering
Significant features were identified through various techniques, including correlation analysis, SHAP, and LIME. The top features selected for the model included:
- `REST_AVG_CUR`
- `LDEAL_ACT_DAYS_PCT_AAVG`
- `REST_DYNAMIC_IL_3M`
- `CR_PROD_CNT_IL_5`
- `CR_PROD_CNT_TOVR_4`
- `REST_DYNAMIC_CUR_1M`
- `CR_PROD_CNT_TOVR_5`
- `CR_PROD_CNT_PIL_4`
- `TURNOVER_DYNAMIC_IL_3M`
- `TURNOVER_DYNAMIC_IL_1M`
- `APP_MARITAL_STATUS_Civil Union`
- `CR_PROD_CNT_CC_9`
- `PACK_109`
- `CR_PROD_CNT_VCU_3`
- `CR_PROD_CNT_TOVR_6`

## Model Training and Tuning
Trained the following models:
- Logistic Regression
- Support Vector Machine (SVM)

Used GridSearchCV for hyperparameter tuning  
Evaluated with:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Interpretability
The interpretability of the models was analyzed using LIME. These methods provided insights into the most important features driving the predictions:

- LIME (Local Interpretable Model-agnostic Explanations) was used to explain individual predictions by approximating the model locally.

## Deployment
- Backend: Flask app running with Gunicorn
- Frontend: Streamlit UI for real-time predictions
- Input form for customer details → instant churn prediction in real-time

## Usage
To run the project locally:
1. Clone the repository.
2. Install the required dependencies.
3. Run the Flask app using Gunicorn.
4. Access the Streamlit interface to input customer data and view predictions.

## Results

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| ~78%     | 0.76      | 0.72   | 0.74     | 0.76    |
| SVM                | ~80%     | 0.77      | 0.79   | 0.78     | 0.80    |

> The **Support Vector Machine** model was selected for deployment due to its higher recall and ROC-AUC, making it more effective for minimizing false negatives in churn prediction.


## Conclusion
This project demonstrates the effectiveness of machine learning in predicting customer churn. By understanding the key features contributing to churn, banks can develop targeted strategies to retain customers and reduce churn rates.

