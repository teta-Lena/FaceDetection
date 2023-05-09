import cv2
import uuid


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        file_name = f'face_{str(uuid.uuid4())}.jpg'

        face_crop = frame[y:y+h, x:x+w]
      
        cv2.imwrite('C:\\Users\\Tlxna\\Documents\\EmbeddedPyProject\\FaceDetection\\'+file_name, face_crop)
 
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()