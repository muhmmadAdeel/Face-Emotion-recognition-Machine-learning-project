# Face Emotion Recognition System

A real-time emotion detection system using deep learning and OpenCV. Detects 7 basic emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) from webcam feed.
## 📂 File Setup
1. Download these files and place them in the same folder as the script:
   - [facialemotionmodel.h5](https://drive.google.com/file/d/1thSlLooc6AOTiWCflj3k4zK0V72FjOie/view?usp=sharing)) [(model weights)
   - ].(https://drive.google.com/file/d/1thSlLooc6AOTiWCflj3k4zK0V72FjOie/view?usp=sharing)
## 📁 File Structure
```
face-emotion-recognition/
├── facialemotionmodel.h5       # Pretrained model weights
├── facialemotionmodel.json     # Model architecture (JSON)
├── haarcascade_frontalface_default.xml  # Haar cascade for face detection
├── emotion_detection.py        # Main Python script (your provided code)
└── README.md
```

## 🛠️ Dependencies
```bash
pip install opencv-python tensorflow numpy
```

## 🚀 How to Run
1. Ensure all files are in the **same directory**
2. Run the script:
```bash
python emotion_detection.py
```
3. Press `Q` to quit the webcam feed

## 🔍 Code Explanation

### Key Components:
1. **Model Loading**  
   - Loads pre-trained CNN model (`facialemotionmodel.json` + `.h5` weights)
   - Uses custom class wrapper for TensorFlow compatibility

2. **Face Detection**  
   - OpenCV's Haar Cascade (`haarcascade_frontalface_default.xml`) detects faces
   - Converts frames to grayscale for processing

3. **Emotion Prediction**  
   - Preprocesses faces (resize to 48x48, normalize pixels)
   - Model predicts among 7 emotions with confidence score

4. **Real-Time Display**  
   - Draws bounding boxes and emotion labels on detected faces
   - Webcam feed displays live results

## ⚠️ Common Issues
- Ensure all files are in the **same directory**
- Webcam access permissions required
- TensorFlow/OpenCV version conflicts may occur (tested with TF 2.x)

## 📝 Notes
- Model trained on FER-2013 dataset (48x48 grayscale images)
- For better accuracy, consider using MTCNN or Dlib for face detection

---

### Suggested Improvements:
1. Add a `requirements.txt` file for dependencies
2. Include sample input/output images in repo
3. Add a section for training custom models
