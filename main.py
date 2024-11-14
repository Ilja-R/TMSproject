from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from preprocess import get_scale_area
import base64
from google_connect import detect_text


app = Flask(__name__)
CORS(app)

def image_to_base64(image):
    # Encode the image as PNG
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/process-image', methods=['POST'])
def process_image_endpoint():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    response = {}

    image_file = request.files['image']
    # Read the image file into a NumPy array
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Get scale area
    scale_area = get_scale_area(image_array)

    if scale_area is None:
        return jsonify({"error": "No scale area detected"}), 400

    # Convert processed image to Base64
    image_base64 = image_to_base64(scale_area)

    # Image text
    image_text = detect_text(scale_area)

    response["image"] = image_base64

    if image_text:
        response["text"] = image_text
    else:
        response["text"] = "Could not detect text in the image."

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
