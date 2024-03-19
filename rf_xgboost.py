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
y_failure = df['Machine failure']  # Target variable for predicting machine failure

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_failure, test_size=0.2, random_state=42)

# Random Forest Classifier for predicting machine failure
rf_failure_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_failure_classifier.fit(X_train, y_train)

# Predictions on the test set
rf_failure_predictions = rf_failure_classifier.predict(X_test)

# Filter dataset to include only samples where machine failure is predicted
X_failure = X_test[rf_failure_predictions == 1]
y_failure = df.loc[X_failure.index][['TWF', 'HDF', 'PWF', 'OSF', 'RNF']]  # Target variable for failure types

# Train separate classifiers for each failure type
failure_classifiers = {}
for failure_type in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
    y = y_failure[failure_type]
    X_train, X_test, y_train, y_test = train_test_split(X_failure, y, test_size=0.2, random_state=42)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    failure_classifiers[failure_type] = rf_classifier

# Evaluate the classifiers for each failure type
for failure_type, classifier in failure_classifiers.items():
    y_test_failure = y_failure[failure_type]
    predictions = classifier.predict(X_failure)
    
    print(f"Results for Failure Type: {failure_type}")
    print("Accuracy:", accuracy_score(y_test_failure, predictions))
    print("Classification Report:")
    print(classification_report(y_test_failure, predictions, zero_division=0))
