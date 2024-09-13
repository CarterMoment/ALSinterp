import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load the preprocessed data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Define the neural network model
model = models.Sequential([
    layers.InputLayer(input_shape=(42,)),  # 42 inputs (21 landmarks, each with x and y coordinates)
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(y_train)), activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save the trained model
model.save('gesture_recognition_model.keras')
print("Model training complete. Model saved as 'gesture_recognition_model.keras'.")