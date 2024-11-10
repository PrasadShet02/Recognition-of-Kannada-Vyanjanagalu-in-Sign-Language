import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe HandLandmark model
mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5)

def detect_hand_landmarks(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Detect hand landmarks
    results = mp_hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),1))

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark])
        return landmarks
    else:
        return np.zeros((21, 3))  # Return a 21x3 array of zeros if no hand is detected

#Storage
def create_store_path(sign_name):
    loc = os.path.join("FEATURES", sign_name)
    if not os.path.exists(loc):
        os.makedirs(loc)
    return loc


# Start point
DATA_PATH = r"Dataset"    #Path where Dataset is located

# Get image path
for sign_name in os.listdir(DATA_PATH):
    sign_data=[]
    store_path = create_store_path(sign_name)
    sign_path = os.path.join(DATA_PATH, sign_name)
    for image_file in os.listdir(sign_path):
        image_path = os.path.join(sign_path, image_file)
        print("Processing", image_path)
        hand_landmarks_np = detect_hand_landmarks(image_path)
        sign_data.append(hand_landmarks_np)
        # print(hand_landmarks_np)
    np_store_path=os.path.join(store_path, sign_name)
    np.save(np_store_path, np.array(sign_data))

