# Importing necessary libraries
import cv2 as cv  # OpenCV library for image processing
import numpy as np  # NumPy for numerical computations
import os  # OS module for interacting with the file system
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set TensorFlow logging level to suppress messages
import tensorflow as tf  # TensorFlow for deep learning
from sklearn.preprocessing import LabelEncoder  # LabelEncoder for encoding labels
import pickle  # Pickle module for serializing and deserializing objects
from keras_facenet import FaceNet  # FaceNet for face recognition

# INITIALIZE

# Initialize FaceNet model for face embeddings
facenet = FaceNet()

# Load precomputed face embeddings and their corresponding labels
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']

# Initialize and fit LabelEncoder to encode labels
encoder = LabelEncoder()
encoder.fit(Y)

# Load Haar cascade classifier for face detection
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load pre-trained SVM model for face recognition
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# Set the confidence threshold for face recognition
CONFIDENCE_THRESHOLD = 0.827

# Open video capture device (webcam)
cap = cv.VideoCapture(0)

# WHILE LOOP
while cap.isOpened():
    # Read frame from video capture device
    _, frame = cap.read()
    
    # Convert frame to RGB and grayscale
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    
    # Iterate over detected faces
    for x, y, w, h in faces:
        # Extract face region from the RGB image
        img = rgb_img[y:y+h, x:x+w]
        
        # Resize face image to 160x160 pixels
        img = cv.resize(img, (160, 160))  # 1x160x160x3
        
        # Expand dimensions to match expected input shape for FaceNet
        img = np.expand_dims(img, axis=0)
        
        # Generate embeddings for the face using FaceNet model
        ypred = facenet.embeddings(img)
        
        # Predict the label (name) of the face using the SVM model
        face_name = model.predict(ypred)
        
        # Get confidence score for the prediction
        confidence = model.decision_function(ypred)
        
        # Check if confidence score is above the threshold
        if np.max(confidence) >= CONFIDENCE_THRESHOLD:
            # If confidence is high, get the final name from the label encoder
            final_name = encoder.inverse_transform(face_name)[0]
        else:
            # If confidence is low, label as "Unknown"
            final_name = "Unknown"
        
        # Draw rectangle around the detected face
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
        
        # Write the recognized name or "Unknown" above the rectangle
        cv.putText(frame, str(final_name), (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 3, cv.LINE_AA)

    # Display the frame with face recognition results
    cv.imshow("Face Recognition:", frame)
    
    # Check for 'q' key press to exit the loop
    if cv.waitKey(1) & ord('q') == 27:
        break

# Release video capture device and close all OpenCV windows
cap.release()
cv.destroyAllWindows()

