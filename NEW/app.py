import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for,jsonify
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import cv2
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'saved_models'
MODEL_PATH = os.path.join(MODEL_FOLDER, 'final_model.h5')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
model = load_model(MODEL_PATH)

# Preprocess frame for model prediction
def preprocess_frame(frame, input_shape=(128, 128)):
    detector = MTCNN()
    faces = detector.detect_faces(frame)
    if faces:
        x, y, width, height = faces[0]['box']
        face = frame[y:y+height, x:x+width]
        face = cv2.resize(face, input_shape[:2])
        face = face / 255.0  # Normalize
        return face
    return None

# Predict deepfake on the video
def predict_deepfake(model, video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fake_scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % int(cap.get(cv2.CAP_PROP_FPS) / frame_rate) == 0:
            face = preprocess_frame(frame)
            if face is not None:
                face = np.expand_dims(face, axis=0)  # Add batch dimension
                score = model.predict(face)[0][0]
                fake_scores.append(score)

        frame_count += 1

    cap.release()

    # Reporting
    if fake_scores:  # Ensure there are scores recorded
        average_score = np.mean(fake_scores)
        print(f"Average Deepfake Score: {average_score:.4f}")

        # Adjust the threshold for classification, you can set it as per your evaluation.
        threshold = 0.7  # Example adjusted threshold, change as needed.

        if average_score < threshold:
            return 'Deepfake', average_score
        else:
            return 'Real', average_score
    else:
        print("No faces detected in the video.")
        return 'No faces detected', None

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        try:
            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                prediction,score = predict_deepfake(model, file_path)
                return render_template('result.html', prediction=prediction)
            return render_template('index.html')
        except Exception as e:
            print(f"Error during video processing: {e}")
            return jsonify({'error': 'Internal server error, please check server logs.'}), 500







# Run the app
if __name__ == '__main__':
    app.run(debug=True)
