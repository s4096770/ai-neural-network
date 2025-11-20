import pandas as pd
import numpy as np
from src.nn.network import NeuralNet

print("🔐 Loading security dataset...")

data = pd.read_csv("data/security_dataset.csv")

X = data[["packet_size", "connection_rate", "login_attempts"]].values
y = data["attack"].values.reshape(-1, 1)

print("🧠 Training Intrusion Detection Neural Network...")

model = NeuralNet(layers=[3, 6, 1], activation="sigmoid", lr=0.1)
model.train(X, y, epochs=2000)

predictions = model.predict(X)
accuracy = np.mean(predictions == y) * 100

print(f"✅ Training Complete — Accuracy: {accuracy:.2f}%")
