import cv2
import tensorflow as tf
import numpy as np
from collections import deque

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

emotion_model = tf.keras.models.load_model("emotion_model.hdf5", compile=False)

# Define the emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load corresponding images for each emotion
emotion_images = {
    "Angry": cv2.imread("angry.jpg"),
    "Disgust": cv2.imread("disgust.jpg"),
    "Fear": cv2.imread("fear.jpg"),
    "Happy": cv2.imread("happy.jpg"),
    "Sad": cv2.imread("sad.jpg"),
    "Surprise": cv2.imread("surprise.jpg"),
    "Neutral": cv2.imread("neutral.jpg"),
}

# Resize all images to 640x480
for emotion, img in emotion_images.items():
    if img is not None:
        emotion_images[emotion] = cv2.resize(img, (640, 480))
    else:
        print(f"[ERROR] Could not load image for emotion: {emotion}")
        exit(1)

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit(1)

print("[DEBUG] Starting emotion detection...")

# Emotion stability queue and blending variables
emotion_queue = deque(maxlen=5)
current_emotion = "Neutral"
next_emotion = "Neutral"
blend_factor = 0.0
blend_step = 0.03  # Slower transitions
transitioning = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    detected_emotion = "Neutral"

    for x, y, w, h in faces[:1]:  # Process only the first detected face
        face_roi = gray_frame[y : y + h, x : x + w]
        face_roi = cv2.resize(face_roi, (64, 64))  # Resize to model input size
        face_roi = face_roi / 255.0  # Normalize pixel values
        face_roi = np.expand_dims(face_roi, axis=-1)  # Add channel dimension
        face_roi = np.expand_dims(face_roi, axis=0)  # Add batch dimension

        # Predict emotion
        predictions = emotion_model.predict(face_roi, verbose=0)
        max_index = np.argmax(predictions[0])
        detected_emotion = emotion_labels[max_index]

        # Draw rectangle and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            detected_emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )

    # Update emotion queue and determine the stable emotion
    emotion_queue.append(detected_emotion)
    stable_emotion = max(set(emotion_queue), key=emotion_queue.count)

    # Debugging the emotion queue
    print(f"[DEBUG] Detected Emotion: {detected_emotion}, Stable Emotion: {stable_emotion}")

    # Start a new transition if a stable emotion is detected
    if stable_emotion != current_emotion and not transitioning:
        next_emotion = stable_emotion
        transitioning = True
        blend_factor = 0.0
        print(f"[DEBUG] Starting transition: {current_emotion} -> {next_emotion}")

    # Perform blending transition
    current_image = emotion_images.get(current_emotion, emotion_images["Neutral"])
    next_image = emotion_images.get(next_emotion, emotion_images["Neutral"])

    if transitioning:
        blended_image = cv2.addWeighted(
            current_image, 1 - blend_factor, next_image, blend_factor, 0
        )
        blend_factor += blend_step
        if blend_factor >= 1.0:
            # Transition complete
            current_emotion = next_emotion
            transitioning = False
            blend_factor = 0.0
            print(f"[DEBUG] Transition complete: {current_emotion}")
    else:
        blended_image = current_image

    # Debug transition state
    print(
        f"[DEBUG] Transitioning: {transitioning}, Blend Factor: {blend_factor:.2f}, Current Emotion: {current_emotion}"
    )

    # Display the blended emotion image
    cv2.imshow("Emotion-Based Image", blended_image)

    # Display the frame with face detection
    cv2.imshow("Emotion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
