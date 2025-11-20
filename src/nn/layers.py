import numpy as np
from .initialisers import xavier_init, he_init

class DenseLayer:
    def __init__(self, input_dim, output_dim, activation, init="xavier"):
        if init == "xavier":
            self.W = xavier_init(input_dim, output_dim)
        else:
            self.W = he_init(input_dim, output_dim)

        self.b = np.zeros((1, output_dim))
        self.activation = activation

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        self.a = self.activation.forward(self.z)
        return self.a

    def backward(self, da, lr):
        dz = da * self.activation.derivative(self.a)
        dW = self.x.T @ dz
        db = np.sum(dz, axis=0, keepdims=True)
        dx = dz @ self.W.T

        self.W -= lr * dW
        self.b -= lr * db

        return dx
