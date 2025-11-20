import numpy as np

class MSE:
    @staticmethod
    def forward(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def derivative(y_true, y_pred):
        return (y_pred - y_true)
