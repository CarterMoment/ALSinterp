import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

gesture_label = 'letter_h'  # Example gesture label, change based on the gesture being recorded

# Open a CSV file to save the landmark data
with open('gesture_data.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
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
                # Flatten hand landmarks (21 points) into a 42-length list (x and y coordinates)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)

                # Append the gesture label (e.g., thumbs_up, open_palm, etc.)
                landmarks.append(gesture_label)
                
                # Save the landmarks and label to the CSV
                writer.writerow(landmarks)

                # Draw hand landmarks on the frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Collecting Hand Landmarks', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()