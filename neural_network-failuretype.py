import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = './ai4i2020.csv'
df = pd.read_csv(file_path)

# Assuming you have separate columns for each failure type in your dataset
conditions = [
    (df['TWF'] == 1),
    (df['HDF'] == 1),
    (df['PWF'] == 1),
    (df['OSF'] == 1),
    (df['RNF'] == 1)
]

failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
df['FailureType'] = np.select(conditions, failure_types, default='No Failure')
# print(df[df['FailureType'] == 'RNF'].head())

# One-hot encode the 'Type' feature
df = pd.get_dummies(df, columns=['Type'])



# Drop irrelevant columns and prepare features (X) and target (y)
X = df.drop(['UID', 'Machine failure', 'Product ID', 'FailureType', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(df['FailureType'])
y = to_categorical(encoded_Y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network architecture
num_categories = y.shape[1]
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_categories, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Further evaluation with classification report
y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred, target_names=encoder.classes_))

#Confusion Matrix Visualization
cm = confusion_matrix(y_test_labels, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()