import pandas as pd
from src.nn.network import NeuralNet

print("🔐 Loading security dataset...")

data = pd.read_csv("data/security_dataset.csv")

X = data[["packet_size", "connection_rate", "login_attempts"]].values
y = data[["attack"]].values

print("🧠 Training Intrusion Detection Neural Network...")

model = NeuralNet(layers=[3, 6, 1], activation="sigmoid", lr=0.1)
model.train(X, y, epochs=1500)

predictions = model.predict(X)
accuracy = (predictions == y).mean() * 100

print(f"✅ Training Complete — Accuracy: {accuracy:.2f}%")
