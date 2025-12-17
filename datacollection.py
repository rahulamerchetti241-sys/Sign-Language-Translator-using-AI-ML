import cv2
import mediapipe as mp
import csv
import os

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Setup CSV File
file_name = 'hand_data.csv'
if not os.path.exists(file_name):
    with open(file_name, mode='w', newline='') as f:
        writer = csv.writer(f)
        # 21 landmarks * 2 coordinates (x,y) + 1 label
        header = ['label'] + [f'coord_{i}' for i in range(42)]
        writer.writerow(header)

cap = cv2.VideoCapture(0)

current_label = input("Enter the label you want to record (e.g., 'Hello', 'A', 'B'): ")
print(f"Press 's' to save a frame for label '{current_label}'. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Flip frame and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])
            
            # Draw instructions
            cv2.putText(frame, f"Label: {current_label} | Press 's' to save", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Save to CSV on keypress
            key = cv2.waitKey(1)
            if key == ord('s'):
                with open(file_name, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([current_label] + landmarks)
                print(f"Saved {current_label}")

    cv2.imshow('Data Collection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()