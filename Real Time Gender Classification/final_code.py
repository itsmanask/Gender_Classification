#FINALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLll

import zipfile
import os
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained gender detection model
model = load_model(r"C:\Users\rajla\OneDrive\Desktop\study material\SY SEM 1 2024\Design Thinking\SAFE-NET\gender_mini_XCEPTION.21-0.95.hdf5", compile=False)

# Gender labels (assuming 0 = female, 1 = male)
labels = {0: 'Female', 1: 'Male'}

# Initialize MediaPipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.8)

def preprocess_face(face):
    """Preprocess the face image to the required format for the model (grayscale + histogram equalization)."""
    gray_face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    equalized_face = cv2.equalizeHist(gray_face)
    resized_face = cv2.resize(equalized_face, (64, 64))
    normalized_face = resized_face / 255.0
    processed_face = np.expand_dims(normalized_face, axis=(0, -1))
    return processed_face

def detect_gender_from_face(face):
    """Detect gender from a face image using the pre-trained model."""
    processed_face = preprocess_face(face)
    prediction = model.predict(processed_face)
    confidence = np.max(prediction)  # Get the maximum confidence score
    return prediction, confidence

def classify_hand_gesture(landmarks):
    """Classify the gesture based on hand landmarks."""
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y

    if (thumb_tip < index_tip and
        thumb_tip < middle_tip and
        thumb_tip < ring_tip and
        thumb_tip < pinky_tip):
        return 'fist'
    else:
        return 'open hand'

def process_image_for_detection(image_path):
    """Process a single image to detect faces, gender, and hand gestures using MediaPipe."""
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    image_resized = cv2.resize(image, (640, 480))
    rgb_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    gender_probabilities = []

    face_results = face_detection.process(rgb_image)

    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = rgb_image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            face_roi = image_resized[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # Predict gender and confidence
            predicted_gender, confidence = detect_gender_from_face(face_rgb)
            gender_probabilities.append(predicted_gender)

            if len(gender_probabilities) > 0:
                averaged_prediction = np.mean(gender_probabilities, axis=0)
                predicted_class = np.argmax(averaged_prediction)
                gender_displayed = labels[predicted_class]

                # Display gender and confidence score on the image
                cv2.putText(image_resized, f"Gender: {gender_displayed} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            cv2.rectangle(image_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)

    else:
        print("No face detected")

    result = hands.process(rgb_image)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image_resized, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = classify_hand_gesture(hand_landmarks.landmark)
            cv2.putText(image_resized, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Image Detection', image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_and_process_images(zip_path):
    """Extract images from a zip file and process them for gender and gesture detection."""
    extract_dir = 'extracted_images'
    
    # Clear the directory before extracting new images
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            file_path = os.path.join(extract_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    
    os.makedirs(extract_dir, exist_ok=True)

    # Extract the zip file
    print(f"Extracting dataset from: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Check for subfolder in the extracted directory
    subfolder_path = os.path.join(extract_dir, 'mydataset')
    if os.path.exists(subfolder_path):
        extract_dir = subfolder_path

    for img_name in os.listdir(extract_dir):
        img_path = os.path.join(extract_dir, img_name)
        if os.path.isfile(img_path):
            print(f"Processing {img_path}")
            process_image_for_detection(img_path)
        else:
            print(f"Skipping {img_name} - not a valid file.")

def process_video_for_detection(video_path=None, use_camera=False, face_frame_skip=1, hand_frame_skip=3):
    """Process video from a file or camera for face and hand detection."""
    if use_camera:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("End of video stream or unable to read video.")
            break

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process face detection
        face_results = face_detection.process(rgb_image)
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = rgb_image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face_roi = image[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

                # Predict gender and confidence
                predicted_gender, confidence = detect_gender_from_face(face_rgb)
                gender_displayed = labels[np.argmax(predicted_gender)]

                # Display gender
                cv2.putText(image, f"Gender: {gender_displayed} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Process hand detection
        result = hands.process(rgb_image)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = classify_hand_gesture(hand_landmarks.landmark)
                cv2.putText(image, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Video Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    # Prompt the user for input on which option to choose
    choice = input("Choose an option:\n1. Process Image from File\n2. Process Video from File\n3. Process Image from Zip File\n4. Use Camera for Detection\nEnter your choice (1/2/3/4): ").strip()

    if choice == '1':
        image_choice = input("Enter the path of the image file for detection: ").strip()
        process_image_for_detection(image_choice)
    elif choice == '2':
        video_choice = input("Enter the path of the video file for detection: ").strip()
        process_video_for_detection(video_path=video_choice)
    elif choice == '3':
        zip_choice = input("Enter the path of the zip file containing images: ").strip()
        extract_and_process_images(zip_choice)
    elif choice == '4':
        process_video_for_detection(use_camera=True)
    else:
        print("Invalid choice! Please select 1, 2, 3, or 4.")

if __name__ == '__main__':
    main()
