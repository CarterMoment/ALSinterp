import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def is_fist(landmarks):
    # Check if the distance between each finger's tip and the wrist is smaller
    # This means that the hand is in a fist
    fingersdown = 0
    finger_tips = [8, 12, 16, 20]
    for tip in finger_tips:
        if landmarks[tip].y > landmarks[tip -2].y:
            fingersdown+=1
    if fingersdown == 4:
        return True
    else:
        return False

def is_open_palm(landmarks):
    fingersup = 0
    finger_tips = [8, 12, 16, 20]
    for tip in finger_tips:
        if landmarks[tip].y < landmarks[tip -2].y:
            fingersup+=1
    if fingersup == 4:
        return True
    else:
        return False

def is_middle_finger(landmarks):
    fingers_tips = [8, 16, 20]
    fingersdown = 0
    for tip in fingers_tips:
        if landmarks[tip].y > landmarks[tip -2].y:
            fingersdown+=1
    if fingersdown == 3 and landmarks[12].y < landmarks[10].y:
        return True
    else:
        return False

with open('gesture_data.csv', mode='a', newline='') as file:
    writer = csv.writer(file)

while True:
    ret, frame = cap.read()
    
    # Convert the frame to RGB (required by MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect the hands in the frame
    result = hands.process(frame_rgb)
    
    # Draw hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark

            if is_fist(landmarks):
                cv2.putText(frame, "Fist", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_open_palm(landmarks):
                cv2.putText(frame, "Open Palm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_middle_finger(landmarks):
                cv2.putText(frame, "F You!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show the result
    cv2.imshow('Gesture Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

