from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'saved_models'
MODEL_PATH = os.path.join(MODEL_FOLDER, 'model.h5')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
model = load_model(MODEL_PATH)

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (112, 112))
        frames.append(frame)
        if len(frames) == 16:
            break
    cap.release()
    frames = np.array(frames)
    if frames.shape == (16, 112, 112, 3):
        frames = np.expand_dims(frames, axis=0)
        prediction = model.predict(frames)
        return np.argmax(prediction)
    return None


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser may submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = predict_video(file_path)
            result = "Fake" if prediction == 1 else "Real" if prediction == 0 else "Error"
            return render_template('result.html', result=result)
    return render_template('index.html')

@app.route('/result')
def result_page():
    return render_template('result.html')

if __name__ == '__main__':
    # Before running the server, we ensure the model is trained and saved.
    # train_and_save_model()
    app.run(debug=True)