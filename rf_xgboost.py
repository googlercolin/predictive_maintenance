import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (assuming it's stored in a CSV file named 'ai4i2020.csv')
df = pd.read_csv('ai4i2020.csv')

# One-hot encode the 'Type' column (assuming it's the categorical column)
df = pd.get_dummies(df, columns=['Type'])

# Rename columns to remove special characters
df.columns = df.columns.str.replace("[", "(").str.replace("]", ")").str.replace("<", "_")

# Split features and target variables
X = df.drop(['UID', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)  # Dropping irrelevant columns
print(X[:5])
y = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']]  # Using the failure type columns as target variables

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reverse mapping dictionary to map encoded labels back to failure types
reverse_mapping = {0: 'TWF', 1: 'HDF', 2: 'PWF', 3: 'OSF', 4: 'RNF'}

# Random Forest Classifier for predicting failure modes
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions on the test set
rf_predictions = rf_classifier.predict(X_test)

# Convert encoded labels back to failure types in the classification report
rf_predictions_failure_types = [reverse_mapping[label] for label in rf_predictions.argmax(axis=1)]

# Evaluate Random Forest Classifier
print("Random Forest Classifier Results:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Classification Report:")
print(classification_report(y_test.idxmax(axis=1), rf_predictions_failure_types, zero_division=0))

# Get feature importance scores for Random Forest
feature_importance_rf = rf_classifier.feature_importances_
print("Random Forest Feature Importance Scores:")
print(feature_importance_rf)

# XGBoost Classifier for predicting failure modes
xgb_classifier = xgb.XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)

# Predictions on the test set
xgb_predictions = xgb_classifier.predict(X_test)

# Convert encoded labels back to failure types in the classification report
xgb_predictions_failure_types = [reverse_mapping[label] for label in xgb_predictions.argmax(axis=1)]

# Evaluate XGBoost Classifier
print("\nXGBoost Classifier Results:")
print("Accuracy:", accuracy_score(y_test, xgb_predictions))
print("Classification Report:")
print(classification_report(y_test.idxmax(axis=1), xgb_predictions_failure_types, zero_division=0))

# Get feature importance scores for XGBoost
feature_importance_xgb = xgb_classifier.feature_importances_
print("\nXGBoost Feature Importance Scores:")
print(feature_importance_xgb)

