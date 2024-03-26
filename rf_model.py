import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import os

# Create a directory to store diagrams if it doesn't exist
diagrams_dir = 'diagrams'
if not os.path.exists(diagrams_dir):
    os.makedirs(diagrams_dir)

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

# Accuracy for Machine Failure Classifier
print("Accuracy for Machine Failure Classifier:", accuracy_score(y_test, rf_failure_predictions))
print(classification_report(y_test, rf_failure_predictions, zero_division=0))

# Confusion matrix for Machine Failure Classifier
conf_matrix = confusion_matrix(y_test, rf_failure_predictions)
# print("Confusion Matrix for Machine Failure Classifier:")

# Display confusion matrix using matplotlib
class_labels = ['No Failure', 'Failure']
df_cm = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
plt.figure(figsize=(6, 4))
sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for Machine Failure Classifier')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(diagrams_dir, f'machine_failure_confusion_matrix.png'))

# Feature Importance Analysis
feature_importance = rf_failure_classifier.feature_importances_

# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Visualize Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance for Machine Failure Prediction (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(diagrams_dir, f'machine_failure_feature_importance.png'))

# Filter training data to include only samples where machine failure is predicted
X_train_failure = X_train[y_train == 1]
y_train_failure = df.loc[X_train_failure.index][['TWF', 'HDF', 'PWF', 'OSF', 'RNF']]  # Target variable for failure types

# Train separate classifiers for each failure type
failure_classifiers = {}
for failure_type in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
    y = y_train_failure[failure_type]
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_failure, y)
    failure_classifiers[failure_type] = rf_classifier

    # Extract feature importance scores
    feature_importance = rf_classifier.feature_importances_

    # Create a DataFrame to display feature importance
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Visualize Feature Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title(f'Feature Importance for {failure_type} Prediction (Random Forest)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(diagrams_dir, f'{failure_type}_feature_importance.png'))


# Evaluate the classifiers on the test set
# print(X_test[:5])
X_test_failure = X_test[rf_failure_predictions == 1]
# print(X_test_failure[:5])
print("Number of predicted failures from the test set:", len(X_test_failure))
y_test_failure = df.loc[X_test_failure.index][['TWF', 'HDF', 'PWF', 'OSF', 'RNF']]

for failure_type in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
    classifier = failure_classifiers[failure_type]
    y_test_type = y_test_failure[failure_type]
    predictions = classifier.predict(X_test_failure)
    print(f"\nResults for Failure Type: {failure_type}")
    print("Accuracy:", accuracy_score(y_test_type, predictions))
    print("Classification Report:")
    print(classification_report(y_test_type, predictions, zero_division=0))