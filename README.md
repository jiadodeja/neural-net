# Neural Network from Scratch

I built a neural network without using PyTorch, TensorFlow, or any ML library using just NumPy and math. Trained it on MNIST handwritten digits and wrapped it in a small web demo where you can draw a digit and watch it predict in real time.

## Why

I wanted to actually understand what's happening inside a neural network. Building it from scratch forced me to understand forward passes, backpropagation, and gradient descent properly.

## How it works

The network is a simple feedforward network with two hidden layers:

```
Input (784) -> Hidden (128) -> Hidden (64) -> Output (10)
```

784 inputs because each MNIST image is 28x28 pixels. 10 outputs, one per digit.

Training took about a minute on CPU and hit 97.8% accuracy on the test set.

## Project structure

```
model/network.py   - the neural network class (forward pass, backprop, train, save/load)
model/train.py     - downloads MNIST, trains the network, saves weights
data/loader.py     - handles downloading and preprocessing the dataset
api/server.py      - Flask server that loads the trained model and serves predictions
demo/index.html    - drawing canvas with confidence bars and live network visualization
```

## Running it

Install dependencies:
```bash
pip install numpy flask flask-cors pillow
```

Train the model:
```bash
python model/train.py
```

Start the server:
```bash
python api/server.py
```

Open `demo/index.html` in your browser, draw a digit, hit Predict.

## Notes

It works well on digits that look like MNIST (small, centered, thin strokes). If you draw thick or off-center it struggles; that's a real ML problem called distribution shift. The training data doesn't match the real-world input style perfectly. To fix it properly you'd augment the training data with different stroke widths, rotations, and positions.

The demo also shows the network's internal activations layer by layer as it processes your drawing, which I thought was a cool way to see what's actually happening under the hood.
