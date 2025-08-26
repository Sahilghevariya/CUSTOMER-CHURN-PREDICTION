# ☎️ Customer Churn Prediction 📈

A comprehensive machine learning project to predict customer churn for a telecom company. This project involves data preprocessing, exploratory data analysis (EDA), feature scaling, training classification models (Logistic Regression and XGBoost), model evaluation, and feature importance visualization.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Tools & Libraries](#tools--libraries)
- [Installation](#installation)
- [Project Workflow](#project-workflow)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Project Structure](#project-structure)

## Project Overview
The goal of this project is to build a classification model that identifies telecom customers who are likely to churn. Reducing customer churn is critical for the telecom industry as retaining customers is often more cost-effective than acquiring new ones.

## Dataset
The dataset used is the **Telco Customer Churn dataset**, which contains customer details like:
- Demographics (gender, senior citizen, partner, dependents)
- Account information (tenure, contract type, monthly charges, total charges)
- Services subscribed (phone service, internet service, streaming services)
- Payment methods
- Churn label (whether the customer has churned or not)

## Tools & Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not available:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```


## Project Workflow

1. **Data Loading & Exploration**

   * Load the dataset.
   * Perform basic inspection (shape, data types, null values).
   * Drop unnecessary columns like `customerID`.

2. **Encoding & Preprocessing**

   * Encode the target variable (`Churn`) to binary format.
   * Convert categorical variables using one-hot encoding.
   * Fill missing values.

3. **Feature Scaling**

   * Standardize features using `StandardScaler` to improve model convergence.

4. **Model Training**

   * **Logistic Regression:** A simple, interpretable baseline model.
   * **XGBoost Classifier:** An advanced gradient boosting model for better performance.

5. **Evaluation**

   * Accuracy
   * Classification Report (Precision, Recall, F1-score)
   * Confusion Matrix

6. **Feature Importance**

   * Visualize top contributing features from the XGBoost model.


## Model Evaluation

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | \~80%    |
| XGBoost Classifier  | \~82%    |

* XGBoost performed slightly better, handling feature interactions more effectively.
* Feature importance visualization provided insights into key drivers of churn.

## Results

* **Top Features Influencing Churn:**

  * Contract Type
  * Tenure
  * Monthly Charges
  * Payment Method
  * Internet Service Type

* Logistic Regression is interpretable but slightly less accurate.

* XGBoost offers better accuracy with insights via feature importance.


## Conclusion

This project successfully demonstrates a typical machine learning pipeline:

* Data cleaning & encoding
* Scaling & splitting data
* Training multiple models
* Evaluating & comparing models
* Interpreting results through visualization

Such insights can help telecom companies proactively retain customers likely to churn.


## Project Structure

customer-churn-prediction/
│
├── data/
│   └── telecom_customer_churn.csv
├── CUSTOMER_CHURN_PREDICTION.py    # Python script version
├── CUSTOMER_CHURN_PREDICTION.ipynb # Jupyter notebook version
├── requirements.txt
└── README.md

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Meet Limbachiya
