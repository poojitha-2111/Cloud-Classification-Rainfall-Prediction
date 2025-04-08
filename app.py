from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json
from werkzeug.utils import secure_filename
import os
import base64
import logging

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load class names and rainfall mapping
try:
    with open("names.json", "r") as f:
        class_names = json.load(f)
        logger.info(f"Loaded class names: {class_names}")

    with open("rainfall_mapping.json", "r") as f:
        rainfall_mapping = json.load(f)
        logger.info(f"Loaded rainfall mapping: {rainfall_mapping}")
except Exception as e:
    logger.error(f"Error loading class names or rainfall mapping: {e}")
    exit(1)

# Load models
models = {}
try:
    models['mobilenetv2'] = tf.keras.models.load_model("best_mobilenetv2_dual.keras")
    models['xception'] = tf.keras.models.load_model("cloud_rainfall_model.h5")
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    exit(1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def rainfall_to_category(rainfall_value):
    try:
        value = int(rainfall_value)
        if value == 0:
            return "No to Low Rain"
        elif value == 1:
            return "Low to Medium Rain"
        elif value == 2:
            return "Medium to High Rain"
        else:
            return "Unknown"
    except (ValueError, TypeError):
        return "Unknown"

@app.route('/')
def serve_html():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            logger.error("No file part in request")
            return jsonify({"error": "No file part"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            logger.error("No selected file")
            return jsonify({"error": "No selected file"}), 400
        
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({"error": "Invalid file type. Please upload a PNG, JPG, or JPEG image"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"File saved: {filepath}")

        try:
            target_size = (224, 224)
            img = image.load_img(filepath, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Predict with both models
            mobilenet_preds = models['mobilenetv2'].predict(img_array)
            xception_preds = models['xception'].predict(img_array)

            # Process MobileNetV2 predictions
            if isinstance(mobilenet_preds, list):
                mobilenet_cloud_preds = mobilenet_preds[0]
                if not np.allclose(np.sum(mobilenet_cloud_preds[0]), 1.0, atol=1e-5):
                    mobilenet_cloud_preds = tf.nn.softmax(mobilenet_preds[0]).numpy()
            else:
                mobilenet_cloud_preds = mobilenet_preds
                if not np.allclose(np.sum(mobilenet_cloud_preds[0]), 1.0, atol=1e-5):
                    mobilenet_cloud_preds = tf.nn.softmax(mobilenet_preds).numpy()
            mobilenet_max_confidence = np.max(mobilenet_cloud_preds[0])
            mobilenet_class_idx = np.argmax(mobilenet_cloud_preds[0])
            mobilenet_class = class_names[mobilenet_class_idx]
            mobilenet_rainfall = rainfall_to_category(rainfall_mapping.get(mobilenet_class, "Unknown"))

            # Process Xception predictions
            if isinstance(xception_preds, list):
                xception_cloud_preds = xception_preds[0]
                if not np.allclose(np.sum(xception_cloud_preds[0]), 1.0, atol=1e-5):
                    xception_cloud_preds = tf.nn.softmax(xception_preds[0]).numpy()
            else:
                xception_cloud_preds = xception_preds
                if not np.allclose(np.sum(xception_cloud_preds[0]), 1.0, atol=1e-5):
                    xception_cloud_preds = tf.nn.softmax(xception_preds).numpy()
            xception_max_confidence = np.max(xception_cloud_preds[0])
            xception_class_idx = np.argmax(xception_cloud_preds[0])
            xception_class = class_names[xception_class_idx]
            xception_rainfall = rainfall_to_category(rainfall_mapping.get(xception_class, "Unknown"))

            # Log predictions for debugging
            logger.info(f"MobileNetV2 max confidence: {mobilenet_max_confidence}, class: {mobilenet_class}")
            logger.info(f"Xception max confidence: {xception_max_confidence}, class: {xception_class}")

            # Validation logic: both models must agree on the same class with high confidence
            CONFIDENCE_THRESHOLD = 0.8  # Stricter threshold
            if (mobilenet_class != xception_class) or (mobilenet_max_confidence < CONFIDENCE_THRESHOLD) or (xception_max_confidence < CONFIDENCE_THRESHOLD):
                with open(filepath, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                logger.info("Image classified as non-cloud: models disagree or low confidence")
                return jsonify({
                    "error": "This doesn't appear to be a cloud image. Please upload an image of clouds.",
                    "image_base64": encoded_image
                })

            with open(filepath, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            logger.info(f"MobileNetV2 prediction: {mobilenet_class}, Rainfall: {mobilenet_rainfall}")
            logger.info(f"Xception prediction: {xception_class}, Rainfall: {xception_rainfall}")

            return jsonify({
                "mobilenetv2": {"class": mobilenet_class, "rainfall": mobilenet_rainfall},
                "xception": {"class": xception_class, "rainfall": xception_rainfall},
                "image_base64": encoded_image
            })
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"File removed: {filepath}")
    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)