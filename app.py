from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import base64
from PIL import Image
import io
import os

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect_mood', methods=['POST'])
def detect_mood():
    try:
        image_data = None

        # If webcam sends base64 image
        if 'image' in request.form:
            image_data = request.form['image']
            image_data = image_data.split(',')[1]
            img = Image.open(io.BytesIO(base64.b64decode(image_data)))
            img_path = "temp_image.jpg"
            img.save(img_path)

        # If user uploads a file
        elif 'file' in request.files:
            file = request.files['file']
            img_path = os.path.join("uploads", file.filename)
            os.makedirs("uploads", exist_ok=True)
            file.save(img_path)

        else:
            return jsonify({'error': 'No image found'}), 400

        # Analyze emotion
        result = DeepFace.analyze(
            img_path=img_path,
            actions=['emotion'],
            enforce_detection=False
        )

        mood = result[0]['dominant_emotion'].capitalize()
        return jsonify({'mood': mood})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'Error detecting mood'}), 500


if __name__ == "__main__":
    app.run(debug=True)
