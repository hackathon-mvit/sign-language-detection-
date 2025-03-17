import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize Mediapipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Create dataset folder if not exists
DATASET_PATH = "dataset/"
GESTURES = ["thumbs up", "thumbs down", "peace", "stop", "fist", "ok"]

# Ensure all gesture folders exist
for gesture in GESTURES:
    os.makedirs(os.path.join(DATASET_PATH, gesture), exist_ok=True)

# Open Camera
cap = cv2.VideoCapture(0)

# Start Data Collection
print("📸 Press 'c' to capture a gesture, 'q' to quit.")

gesture_index = 0  # Index to track current gesture
sample_count = 0   # Number of samples collected

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            # Draw Hand Landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Capture Data when 'c' is pressed
            if cv2.waitKey(1) & 0xFF == ord('c'):
                np.save(os.path.join(DATASET_PATH, GESTURES[gesture_index], f"{sample_count}.npy"), np.array(keypoints))
                print(f"✅ {GESTURES[gesture_index]} sample {sample_count} saved.")
                sample_count += 1

    # Display the frame
    cv2.putText(frame, f"Collecting: {GESTURES[gesture_index]} ({sample_count})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("Gesture Data Collection", frame)

    # Switch gesture with 'n', exit with 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        gesture_index = (gesture_index + 1) % len(GESTURES)
        sample_count = 0  # Reset count for new gesture
        print(f"➡️ Now collecting: {GESTURES[gesture_index]}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Data collection complete.")
