# Assignment-6-DA5401-

# Handling Missing Data: A Comparative Analysis of Imputation Techniques

## Overview

This project provides a comprehensive analysis of various techniques for handling missing data in a real-world dataset. The primary objective is to evaluate the impact of different imputation strategies on the performance of a predictive model. The project uses the "UCI Credit Card Default Clients Dataset" to demonstrate and compare the following methods for handling missing data:

*   **Median Imputation:** A simple and robust method.
*   **Linear Regression Imputation:** A model-based approach assuming linear relationships.
*   **Non-linear Regression Imputation (K-Nearest Neighbors):** A model-based approach for more complex, non-linear relationships.
*   **Listwise Deletion:** A baseline approach where rows with missing data are removed.

A Logistic Regression classifier is trained on each of the resulting datasets to predict credit card default, and the performance is evaluated and compared.

## Table of Contents

*   [Project Objective](#project-objective)
*   [Dataset](#dataset)
*   [Methodology](#methodology)
    *   [Part A: Data Preprocessing and Imputation](#part-a-data-preprocessing-and-imputation)
    *   [Part B: Model Training and Evaluation](#part-b-model-training-and-evaluation)
*   [Results](#results)
*   [Discussion and Recommendations](#discussion-and-recommendations)
*   [How to Run This Project](#how-to-run-this-project)

## Project Objective

The main goals of this project are to:
1.  Introduce missing data into a clean dataset to simulate a real-world scenario.
2.  Implement and compare three different imputation strategies against a baseline of listwise deletion.
3.  Evaluate how each data handling strategy affects the performance of a Logistic Regression classifier.
4.  Provide recommendations on the most effective strategy for this dataset and context.

## Dataset

The project uses the **UCI Credit Card Default Clients Dataset**. This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements for credit card clients in Taiwan from April 2005 to September 2005.

To simulate a missing data scenario, Missing At Random (MAR) values were artificially introduced into the following numeric columns:
*   `AGE`
*   `BILL_AMT1`
*   `PAY_AMT1`

## Methodology

The project is divided into two main parts: Data Preprocessing and Imputation, followed by Model Training and Evaluation.

### Part A: Data Preprocessing and Imputation

Four datasets were created based on different strategies for handling missing data:

1.  **Dataset A (Median Imputation):** Missing values in each feature were replaced with their respective median. The median was chosen for its robustness to outliers and skewed distributions.

2.  **Dataset B (Linear Regression Imputation):** Missing values in the `AGE` column were imputed using a Linear Regression model. This approach leverages the linear relationships between `AGE` and other numeric features to predict the missing values.

3.  **Dataset C (Non-linear Regression Imputation - KNN):** A K-Nearest Neighbors (KNN) Regressor was used to impute the missing `AGE` values. This method was chosen to capture potential non-linear relationships in the data.

4.  **Dataset D (Listwise Deletion):** As a baseline, any row containing one or more missing values was removed from the dataset.

### Part B: Model Training and Evaluation

For each of the four datasets, the following steps were performed:

1.  **Train-Test Split:** The data was split into training (70%) and testing (30%) sets using a stratified split to maintain the class distribution of the target variable (`default.payment.next.month`).

2.  **Feature Standardization:** All numeric features were standardized using `StandardScaler` to ensure that each feature contributes equally to the model's training.

3.  **Classifier Training:** A **Logistic Regression** model was trained on the preprocessed training data.

4.  **Evaluation:** The performance of the trained model was evaluated on the test set using a classification report, which includes metrics such as **Accuracy, Precision, Recall, and F1-score**.

## Results

The performance of the Logistic Regression classifier on each of the four datasets is summarized below:

| Dataset                         | Accuracy | Weighted F1-score | Minority Class Recall |
|---------------------------------|----------|-------------------|-----------------------|
| **A (Median Imputation)**       | 0.8088   | 0.7696            | 0.2366                |
| **B (Linear Regression)**       | 0.8086   | 0.7693            | 0.2356                |
| **C (KNN Regression)**          | 0.8091   | 0.7702            | 0.2381                |
| **D (Listwise Deletion)**       | 0.8059   | 0.7659            | 0.2317                |

## Discussion and Recommendations

**Key Observations:**

*   **Overall Performance:** All imputation methods resulted in very similar model performance, with only marginal differences in accuracy and F1-score. This suggests that for this dataset and the amount of missingness introduced (~7%), the choice of imputation method did not have a dramatic impact on the overall predictive power of the model.
*   **Listwise Deletion:** While being the simplest method, listwise deletion led to a significant reduction in the dataset size. Interestingly, it did not lead to a significant drop in performance, suggesting that the removed rows may not have contained unique information crucial for the model. However, this is not always the case and can be a risky approach.
*   **Linear vs. Non-linear Imputation:** The performance of Linear Regression imputation was very close to that of the more complex KNN imputation. This indicates that the relationships between the features in this dataset are largely linear, and a simple linear model is sufficient for imputation.
*   **Class Imbalance:** A key finding across all models is the low recall for the minority class (default = 1). This is a common issue in credit default prediction and indicates that the model struggles to correctly identify instances of default.

**Recommendations:**

1.  **Prefer Imputation over Listwise Deletion:** Given that imputation preserves the dataset size without a significant trade-off in performance, it is the recommended approach. This is particularly important in scenarios with a higher percentage of missing data.

2.  **Start with Simple Imputation:** For this dataset, simple median or linear regression imputation is sufficient. More complex methods like KNN did not provide a significant benefit and come with higher computational costs.

3.  **Address Class Imbalance:** The low recall for the minority class is a more pressing issue than the choice of imputation method. To improve the model's ability to predict defaults, techniques such as **SMOTE (Synthetic Minority Over-sampling Technique)**, **class weighting** in the logistic regression model, or using different evaluation metrics like the **Area Under the Precision-Recall Curve (AUPRC)** should be explored.

## How to Run This Project

To run this project, you will need to have Python and the following libraries installed:
*   pandas
*   numpy
*   scikit-learn
*   matplotlib
*   seaborn

You can install these dependencies using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd <project-directory>
    ```
3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook DA5401_A6_DA25C011.ipynb
    ```
4.  **Dataset:** Make sure the `UCI_Credit_Card.csv` file is in the correct path as specified in the notebook.
