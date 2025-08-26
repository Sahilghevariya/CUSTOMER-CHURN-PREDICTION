# CUSTOMER_CHURN_PREDICTION.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb

# -----------------------------
# Load Dataset
df = pd.read_csv('telecom_customer_churn.csv')  # Make sure file path is correct

print("Data Shape:", df.shape)
print("First 5 rows:\n", df.head())

# -----------------------------
# Data Preprocessing

# Drop customerID
df.drop('customerID', axis=1, inplace=True)

# Convert 'TotalCharges' to numeric (handle errors)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encode categorical features
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print("\nAfter Encoding:\n", df.head())

# -----------------------------
# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('Churn', axis=1))
X = pd.DataFrame(scaled_features, columns=df.columns.drop('Churn'))
y = df['Churn']

# -----------------------------
# Train-test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Logistic Regression Model
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# -----------------------------
# XGBoost Model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("\n--- XGBoost Classifier ---")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# -----------------------------
# Feature Importance Visualization for XGBoost
plt.figure(figsize=(10,6))
xgb.plot_importance(xgb_model, max_num_features=10, importance_type='gain')
plt.title('Top 10 Feature Importances - XGBoost')
plt.show()

# -----------------------------
# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=False, cmap='viridis')
plt.title('Feature Correlation Heatmap')
plt.show()
