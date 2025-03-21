import streamlit as st
import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image

# Load pre-trained models
FACE_PROTO = "models/opencv_face_detector.pbtxt"
FACE_MODEL = "models/opencv_face_detector_uint8.pb"
AGE_PROTO = "models/age_deploy.prototxt"
AGE_MODEL = "models/age_net.caffemodel"
GENDER_PROTO = "models/gender_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"

# Load networks
faceNet = cv.dnn.readNet(FACE_MODEL, FACE_PROTO)
ageNet = cv.dnn.readNet(AGE_MODEL, AGE_PROTO)
genderNet = cv.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

# Define age and gender labels
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_BUCKETS = ['Male', 'Female']

# Function to detect faces
def detect_faces(image, confidence_threshold=0.7):
    blob = cv.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    height, width = image.shape[:2]
    faces = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            faces.append((x, y, x1, y1, confidence))
    
    return faces

# Function to predict gender and age
def predict_gender_age(face_img):
    blob = cv.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Predict Gender
    genderNet.setInput(blob)
    gender_preds = genderNet.forward()
    gender = GENDER_BUCKETS[gender_preds[0].argmax()]
    gender_confidence = gender_preds[0].max()

    # Predict Age
    ageNet.setInput(blob)
    age_preds = ageNet.forward()
    age = AGE_BUCKETS[age_preds[0].argmax()]
    age_confidence = age_preds[0].max()

    return gender, gender_confidence, age, age_confidence

# Streamlit UI
st.title("👤 Gender & Age Detection App")
st.write("Upload an image and the model will predict the gender and age of detected faces.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read image
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    
    # Preprocessing: Detect Faces
    faces = detect_faces(image)
    
    if len(faces) == 0:
        st.warning("No faces detected! Try another image.")
    else:
        for (x, y, x1, y1, conf) in faces:
            face_img = image[y:y1, x:x1]
            if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                gender, gender_conf, age, age_conf = predict_gender_age(face_img)

                # Draw bounding box and text
                label = f"{gender} ({gender_conf*100:.1f}%), {age} ({age_conf*100:.1f}%)"
                cv.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 2)
                cv.putText(image, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Display Confidence Levels in Streamlit
                st.write(f"**Gender:** {gender} ({gender_conf*100:.1f}%)")
                st.write(f"**Age:** {age} ({age_conf*100:.1f}%)")

        # Convert image back to RGB for visualization
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        st.image(image, caption="Processed Image", use_container_width=True)

        # Show a confidence bar for accuracy visualization
        fig, ax = plt.subplots()
        ax.bar(["Gender Confidence", "Age Confidence"], [gender_conf * 100, age_conf * 100], color=["blue", "red"])
        ax.set_ylim([0, 100])
        ax.set_ylabel("Confidence (%)")
        ax.set_title("Prediction Confidence Levels")
        st.pyplot(fig)
