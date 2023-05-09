import cv2
import os

# Load the image
img = cv2.imread('C:\\Users\\Tlxna\\Documents\\EmbeddedPyProject\\Screenshot_2023.03.07_09.04.37.225.png');

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Loop over each face and crop it
for i, (x, y, w, h) in enumerate(faces):
    face_img = img[y:y + h, x:x + w]

    # Save the cropped face as a new file
    face_filename = f'face_{i}.jpg'
    cv2.imwrite(face_filename, face_img)

    print(f'Saved {face_filename}')
