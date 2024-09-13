import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the trained gesture recognition model
model = tf.keras.models.load_model('gesture_recognition_model.keras')

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Gesture labels (in the same order as during training)
gesture_labels = ['fist', 'letter_a', 'letter_b', 'letter_c', 'middle_finger', 'open_palm', 'thumbs_up']

cap = cv2.VideoCapture(0)
frame_count = 0
while True:
    ret, frame = cap.read()
    frame_count += 1
    if frame_count % 2 == 0:
        continue
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()

            # Normalize the landmarks relative to the wrist
            base_x, base_y = landmarks[0], landmarks[1]
            landmarks_normalized = [(landmarks[i] - base_x, landmarks[i + 1] - base_y)
                                    for i in range(0, len(landmarks), 2)]
            landmarks_normalized = np.array(landmarks_normalized).flatten()

            # Predict the gesture
            gesture_probs = model.predict(np.array([landmarks_normalized]))
            gesture_index = np.argmax(gesture_probs)
            gesture_label = gesture_labels[gesture_index]

            # Display the gesture on the screen
            cv2.putText(frame, f'Gesture: {gesture_label}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Real-Time Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
