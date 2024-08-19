# Video Deepfake Detection

This project is a Flask web application that predicts whether a video is real or fake using a pre-trained Convolutional Neural Network (CNN) model. Users can upload a video, and the model will analyze the video frames to make a prediction.

## Prerequisites

Before running the project, ensure you have the following installed:

- **Python 3.11 or higher**
- Pip (Python package manager)
- Git (optional, for cloning the repository)

## Installation

1. **Clone the Repository (if applicable):**

   ```bash
   git clone https://github.com/SAARKS-BH/DeepFakeDetection_-DFDC-.git
   cd DeepFakeDetection_-DFDC-
   ```

2. **Create a Virtual Environment (optional but recommended):**

   ```bash
   python -m venv v<your-env-name>
   source <your-env-name>\Scripts\activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages:**

   ```bash
   pip install Flask numpy opencv-python tensorflow scikit-learn
   ```

## Model Setup

1. **Download the Pre-trained Model:**

   Download the `model.h5` file from [this link](https://drive.google.com/drive/folders/1sWSWn692h9AgGma3hgFuNNQ7fKOVo6J1?hl=en) and place it in the `saved_models` folder.

   Ensure the folder structure looks like this:

   ```
   video-deepfake-detection/
   ├── saved_models/
   │   └── model.h5
   ├── uploads/
   ├── separated_dataset/  # Optional, if needed for further training or evaluation
   ├── templates/
   ├── static/
   └── app.py
   ```

2. **Download the Separated Dataset (Optional):**

   If the separated dataset is needed for further training or evaluation, download it from [this link](https://drive.google.com/drive/folders/1sWSWn692h9AgGma3hgFuNNQ7fKOVo6J1?hl=en).

   Place the `separated_dataset` folder in the root directory of the project, ensuring it follows the structure above.

## Running the Application

1. **Run the Flask Application:**

   Start the Flask development server by running:

   ```bash
   python app.py
   ```

   This will start the server on `http://127.0.0.1:5000/`.

2. **Access the Application:**

   Open your web browser and go to `http://127.0.0.1:5000/`. You will see the upload interface where you can upload a video file.

3. **Upload a Video:**

   Upload a video file (in a supported format) to predict whether the video is real or fake. The result will be displayed on the result page.

## Project Structure

- `app.py`: Main Flask application file that handles video uploads and prediction logic.
- `saved_models/`: Directory where the pre-trained model (`model.h5`) is stored.
- `uploads/`: Directory where uploaded video files are temporarily stored.
- `separated_dataset/`: (Optional) Directory where the separated dataset is stored, if needed.
- `templates/`: Contains HTML templates (`index.html` for the upload page and `result.html` for the result page).
- `static/`: Contains static files (CSS, JavaScript, images).

## Notes

- Ensure that the uploaded videos are in a format supported by OpenCV (e.g., MP4).
- The model expects videos to be resized to 112x112 and to contain exactly 16 frames for proper prediction.

## Troubleshooting

- If the Flask server fails to start, check the terminal for any error messages and ensure all dependencies are installed.
- If you encounter issues with video format or prediction, ensure that the input video meets the model's requirements (112x112 resolution and 16 frames).

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
