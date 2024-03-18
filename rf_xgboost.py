import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (assuming it's stored in a CSV file named 'ai4i2020.csv')
df = pd.read_csv('ai4i2020.csv')

# One-hot encode the 'Type' column
df = pd.get_dummies(df, columns=['Type'])

# Drop the 'UID' and 'Product ID' columns as they're not useful for prediction
df = df.drop(['UID', 'Product ID'], axis=1)

df.columns = df.columns.str.replace("[", "(").str.replace("]", ")").str.replace("<", "_")

# Split features and target variable
X = df.drop(['Machine failure'], axis=1)  # Dropping the target column
y = df['Machine failure']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier for predicting failure modes
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions on the test set for Random Forest
rf_predictions = rf_classifier.predict(X_test)

# XGBoost Classifier
# Define the XGBoost classifier
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

# Train the XGBoost classifier
xgb_classifier.fit(X_train, y_train)

# Predictions on the test set for XGBoost
xgb_predictions = xgb_classifier.predict(X_test)

# Evaluate Random Forest Classifier
print("Random Forest Classifier Results:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Classification Report:")
print(classification_report(y_test, rf_predictions))

# Evaluate XGBoost Classifier
print("\nXGBoost Classifier Results:")
print("Accuracy:", accuracy_score(y_test, xgb_predictions))
print("Classification Report:")
print(classification_report(y_test, xgb_predictions))
