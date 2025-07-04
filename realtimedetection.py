import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import custom_object_scope
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  

@tf.keras.utils.register_keras_serializable()
class CustomSequential(tf.keras.Sequential):
    pass

try:
    with custom_object_scope({'Sequential': CustomSequential}):
        with open("facialemotionmodel.json", "r") as json_file:
            model = model_from_json(json_file.read())
        model.load_weights("facialemotionmodel.h5")
        print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    exit()


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
if face_cascade.empty():
    print("❌ Failed to load Haar cascade")
    exit()

EMOTIONS = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 
    3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'
}

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype('float32') / 255.0
    return np.expand_dims(np.expand_dims(face_img, -1), 0)

def run_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            try:
                face = gray[y:y+h, x:x+w]
                processed = preprocess_face(face)
                
                preds = model.predict(processed, verbose=0)[0]
                emotion = EMOTIONS[np.argmax(preds)]
                confidence = np.max(preds)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion} ({confidence:.2f})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                           (0, 255, 255), 2)
                
            except Exception as e:
                print(f"⚠️ Face processing error: {e}")
                continue

        cv2.imshow('Real-time Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    run_detection()