import numpy as np
from src.nn.network import NeuralNet

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = NeuralNet(layers=[2,4,1], activation="sigmoid", lr=0.5)
model.train(X, y, epochs=2000)

print("Predictions:")
print(model.predict(X))
