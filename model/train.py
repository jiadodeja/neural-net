import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.network import NeuralNetwork
from data.loader import load_mnist
import numpy as np

def main():
    # Load data
    print("Loading MNIST data...")
    X_train, y_train, X_test, y_test = load_mnist()

    # Build the network
    # 784 inputs (pixels) -> 128 neurons -> 64 neurons -> 10 outputs (digits)
    print("Building network...")
    nn = NeuralNetwork([784, 128, 64, 10])

    # Train
    print("Training...")
    nn.train(
        X_train,
        y_train,
        epochs=50,
        learning_rate=0.1,
        batch_size=64
    )

    # Test accuracy
    print("\nEvaluating on test set...")
    y_pred = nn.forward(X_test)
    predictions = np.argmax(y_pred, axis=1)
    accuracy = np.mean(predictions == y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # Save the trained weights
    os.makedirs("model/saved", exist_ok=True)
    nn.save("model/saved/mnist_model")
    print("Model saved to model/saved/mnist_model.npz")

if __name__ == "__main__":
    main()
