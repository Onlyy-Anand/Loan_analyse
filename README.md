# Loan_analyse
# Loan Approval Prediction

This repository contains a supervised machine learning pipeline designed to predict loan approval statuses based on borrower demographics and financial features. 

## Project Overview
In financial lending, accurately predicting loan defaults is critical. This project focuses on building a robust classification model that handles missing data, scales numerical features, and specifically addresses the class imbalance between approved and rejected loans.

## Repository Contents
* `loan.ipynb`: The main Colab Notebook containing the end-to-end data pipeline, exploratory analysis, and model training.
* `loan_prediction.csv`: The dataset used for training and testing.
## Methodology
1. **Preprocessing:** 
   - Handled missing values using Median (numerical) and Mode (categorical) imputation.
   - Applied One-Hot Encoding for categorical variables.
   - Applied Standard Scaling for numerical features.
2. **Imbalance Handling:** Utilized algorithm-level class weighting (`class_weight='balanced'`) to penalize minority class misclassifications.
3. **Modeling:** Compared Logistic Regression and Random Forest classifiers.

## Results
**Logistic Regression** outperformed Random Forest as the most balanced model for this dataset:
* **F1-Score:** 0.877
* **ROC-AUC:** 0.854
* **Optimal Threshold:** 0.488

The model successfully reduces the approval of bad loans by 31% compared to a baseline approach, providing a tunable probability threshold that can be adjusted based on the institution's risk appetite.

## How to Run
1. Clone this repository.
2. Install the required dependencies: `pip install pandas numpy scikit-learn`
3. Open and run `loan.ipynb` in Google Colab.
