import numpy as np
import mediapipe as mp
import cv2
import os
import joblib
from display import update_window

# Define the class labels
DATA_PATH = r"FEATURES" # Path where Features extracted are stored
CLASSES_LIST = os.listdir(DATA_PATH)
print(CLASSES_LIST)

# Load the SVM or Random Forest model here
model = joblib.load(r'\MODELS\SVM\svm_model.pkl') # Path where SVM model is stored.
# model = joblib.load(r'\MODELS\RandomForest\rf_model.pkl')  # Path where Random forest model is stored.

# Initialize MediaPipe HandLandmark model
mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5)

def detect_hand_landmarks(image):
    # Detect hand landmarks
    results = mp_hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),1))

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark])
        return landmarks
    else:
        return np.zeros((21, 3))  # Return a 21x3 array of zeros if no hand is detected


# Create a function to preprocess a frame and make predictions
def predict_sign(frame):
    # Make a prediction
    predictions = model.predict(frame)
    print(predictions)
    if np.max(predictions) >= 0.8:
        predicted_class_index = int(predictions[0])  # Convert to integer for indexing
        predicted_class = CLASSES_LIST[predicted_class_index]
    else:
        # If no class is predicted with sufficient confidence, set it to "none"
        predicted_class = "None"
    
    return predicted_class

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera (you can change this if you have multiple cameras)

while cap.isOpened():
    success, frame = cap.read()  # Read a frame from the webcam
    
    if not success:
        break
    
    res = detect_hand_landmarks(frame)
    res = res.reshape(1, 21 * 3)  # Reshape the array appropriately for the SVM or Random Forest model
    
    # Make predictions on the frame
    predicted_label = predict_sign(res)
    
    # Display the predicted label on the frame
    # cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Display the frame
    update_window(frame, predicted_label)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
