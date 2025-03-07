from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications import MobileNetV2, imagenet_utils

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = MobileNetV2(weights="imagenet")

def detect_objects(filepath):
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)
    results = imagenet_utils.decode_predictions(predictions)

    detections = [{"object": res[1]} for res in results[0]]
    
    return detections

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detections", methods=["POST"])
def upload_file():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        detections = detect_objects(filepath)

        return jsonify({
            "status": "success",
            "image_path": filepath,
            "detections": detections
        }), 200

if __name__ == "__main__":
    app.run(debug=True)
