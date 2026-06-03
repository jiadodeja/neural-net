import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
from model.network import NeuralNetwork
import numpy as np
from PIL import Image, ImageOps
import io
import base64

app = Flask(__name__)
CORS(app)

# Load the trained model once when the server starts
nn = NeuralNetwork([784, 128, 64, 10])
nn.load("model/saved/mnist_model.npz")
print("Model loaded successfully")


def preprocess(image_data):
    # Decode base64 image from browser
    image_bytes = base64.b64decode(image_data.split(",")[1])
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Crop to just the drawn area with a small padding
    bbox = image.getbbox()
    if bbox is None:
        return None
    padding = 20
    bbox = (
        max(0, bbox[0] - padding),
        max(0, bbox[1] - padding),
        min(image.width, bbox[2] + padding),
        min(image.height, bbox[3] + padding)
    )
    image = image.crop(bbox)

    # Resize to fit inside 20x20 (like MNIST) then pad to 28x28
    image.thumbnail((20, 20), Image.LANCZOS)
    padded = Image.new("L", (28, 28), 0)
    x_offset = (28 - image.width) // 2
    y_offset = (28 - image.height) // 2
    padded.paste(image, (x_offset, y_offset))

    # Normalize to 0-1 and flatten
    pixels = np.array(padded).astype(np.float32) / 255.0
    return pixels.reshape(1, 784)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    pixels = preprocess(data["image"])

    if pixels is None:
        return jsonify({"error": "empty canvas"}), 400

    # Run through the network
    output = nn.forward(pixels)
    confidences = output[0].tolist()
    prediction = int(np.argmax(output))
    activations = [a[0].tolist() for a in nn.activations]

    return jsonify({
        "prediction": prediction,
        "confidences": confidences,
        "activations": activations
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
