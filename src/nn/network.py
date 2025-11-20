import numpy as np
from .layers import DenseLayer
from .activations import get_activation
from .losses import MSE

class NeuralNet:
    def __init__(self, layers, activation="relu", lr=0.01, init="xavier"):
        self.lr = lr
        self.layers_config = layers
        self.activation = get_activation(activation)
        self.init = init
        self.layers = []

        for i in range(len(layers) - 1):
            self.layers.append(
                DenseLayer(
                    layers[i],
                    layers[i + 1],
                    activation=self.activation,
                    init=self.init
                )
            )

    def forward(self, X):
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, loss_grad):
        da = loss_grad
        for layer in reversed(self.layers):
            da = layer.backward(da, self.lr)

    def train(self, X, y, epochs=2000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = MSE.forward(y, y_pred)
            loss_grad = MSE.derivative(y, y_pred)
            self.backward(loss_grad)

            if epoch % 200 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)
