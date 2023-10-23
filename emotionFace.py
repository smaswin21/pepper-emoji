# Done cv2 
# Done Deep Face 

!pip install opencv-python
from deepface import DeepFace
import face_recognition
import time
import emoji

# Mapping emotions to emojis
def map_to_emoji(emotion):
    emoji_mapping = {
        "Happiness": emoji.emojize(":smile:"),
        "Sadness": emoji.emojize(":cry:"),
        "Anger": emoji.emojize(":angry:"),
        "Fear": emoji.emojize(":scream:"),
        "Disgust": emoji.emojize(":nauseated_face:"),
        "Surprise": emoji.emojize(":open_mouth:"),
        "Neutral": emoji.emojize(":neutral_face:")
    }
    return emoji_mapping.get(emotion, "Unknown")

# Initialize the webcam
cap = cv2.VideoCapture(0)

start_time = time.time()
DURATION = 15  # 15 seconds

results = []  # To store the results

while True:
    elapsed_time = time.time() - start_time
    if elapsed_time > DURATION:
        break

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(frame)

    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        predictions = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
        raw_emotion = predictions['dominant_emotion']
        basic_emotion = map_to_basic_emotion(raw_emotion)
        
        # Convert emotion to emoji
        emo = map_to_emoji(basic_emotion)
        
        # Store the results
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        results.append((timestamp, basic_emotion, emo))

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, basic_emotion, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

# Save the results in a separate file
with open("emotion_results.txt", "w") as file:
    file.write("Timestamp\tEmotion\tEmoji\n")
    for r in results:
        file.write(f"{r[0]}\t{r[1]}\t{r[2]}\n")
