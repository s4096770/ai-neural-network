import numpy as np

class Sigmoid:
    @staticmethod
    def forward(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def derivative(a):
        return a * (1 - a)

class ReLU:
    @staticmethod
    def forward(z):
        return np.maximum(0, z)

    @staticmethod
    def derivative(a):
        return (a > 0).astype(float)

class Tanh:
    @staticmethod
    def forward(z):
        return np.tanh(z)

    @staticmethod
    def derivative(a):
        return 1 - np.power(a, 2)

def get_activation(name):
    name = name.lower()
    if name == "sigmoid":
        return Sigmoid
    elif name == "relu":
        return ReLU
    elif name == "tanh":
        return Tanh
    else:
        raise ValueError(f"Unknown activation: {name}")
