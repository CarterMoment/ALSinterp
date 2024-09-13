import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv('gesture_data.csv')

# Separate features (landmarks) and labels (gestures)
X = data.iloc[:, :-1].values  # Hand landmarks
y = data.iloc[:, -1].values   # Gesture labels

# Normalize the landmarks (relative to wrist, i.e., landmark 0)
def normalize_landmarks(landmarks):
    base_x = landmarks[0]
    base_y = landmarks[1]
    normalized_landmarks = [(landmarks[i] - base_x, landmarks[i+1] - base_y) 
                            for i in range(0, len(landmarks), 2)]
    return np.array(normalized_landmarks).flatten()

X_normalized = np.array([normalize_landmarks(x) for x in X])

# Convert string labels to numerical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

joblib.dump(label_encoder, 'gesture_label_encoder.pk1')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.2, random_state=42)

# Save the preprocessed data for future use
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("Preprocessing complete. Data saved as .npy files.")