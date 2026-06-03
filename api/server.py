import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
from model.network import NeuralNetwork
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

# Load the trained model once when the server starts
nn = NeuralNetwork([784, 128, 64, 10])
nn.load("model/saved/mnist_model.npz")
print("Model loaded successfully")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # The image arrives as a base64 string from the browser canvas
    image_data = data["image"].split(",")[1]
    image_bytes = base64.b64decode(image_data)

    # Open the image, convert to grayscale, resize to 28x28
    image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))

    # Convert to numpy array, normalize to 0-1, flatten to 784 numbers
    pixels = np.array(image).astype(np.float32) / 255.0
    pixels = pixels.reshape(1, 784)

    # Run through the network
    output = nn.forward(pixels)
    confidences = output[0].tolist()
    prediction = int(np.argmax(output))

    # Also return the activations of each layer for the visualization
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
