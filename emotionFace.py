import cv2
from deepface import DeepFace
import face_recognition

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(frame)

    for face_location in face_locations:
        top, right, bottom, left = face_location

        # Extract the face from the frame
        face_image = frame[top:bottom, left:right]

        # Predict emotion using DeepFace
        predictions = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)

        emotion = predictions['dominant_emotion']

        # Draws a rectangle around the face and display the emotion
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Displays the resulting frame
    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
