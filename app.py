from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

app.config["MONGO_URI"] = "mongodb://localhost:27017/archery"
app.config['UPLOAD_FOLDER'] = 'uploads/'
mongo = PyMongo(app)

# Load pre-trained models
pose_estimation_model = load_model('original.mp4')
feedback_model = load_model('path/to/feedback_model.h5')

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (368, 368))
        frame = np.expand_dims(frame, axis=0)
        frames.append(frame)

    frames = np.vstack(frames)
    predictions = pose_estimation_model.predict(frames)
    feedback = feedback_model.predict(predictions)
    feedback_text = "Correct Technique" if feedback.mean() > 0.5 else "Incorrect Technique"

    return feedback_text

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return "No video file provided", 400

    video = request.files['video']
    filename = secure_filename(video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)

    feedback = analyze_video(video_path)

    feedback_entry = {
        "feedback": feedback,
        "date": request.form.get("date", "Unknown date")
    }

    mongo.db.feedbacks.insert_one(feedback_entry)

    return jsonify({"feedback": feedback})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
