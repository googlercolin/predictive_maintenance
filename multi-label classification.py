import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

file_path = './ai4i2020.csv'
df = pd.read_csv(file_path)
df = pd.get_dummies(df, columns=['Type'])

# Preparing the features and target variables
X = df.drop(['UID', 'Machine failure', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
# note: removing machine failure to prevent leaking information to the model
y = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']]  # Multi-label targets


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='sigmoid')  # Sigmoid activation for multi-label classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Binary cross-entropy for multi-label

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# For detailed evaluation, consider using sklearn's classification_report, which you can use for each label
y_pred = model.predict(X_test_scaled) > 0.3  # just used 0.3 as confidence level to convert to binary label
print(classification_report(y_test, y_pred, target_names=['TWF', 'HDF', 'PWF', 'OSF', 'RNF']))
