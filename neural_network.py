import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Define the architecture ---
inputs = 2
hidden = 3
outputs = 1

# --- Step 2: Initialise weights and biases ---
np.random.seed(42)
W1 = np.random.randn(inputs, hidden)
b1 = np.zeros((1, hidden))
W2 = np.random.randn(hidden, outputs)
b2 = np.zeros((1, outputs))

# --- Step 3: Activation function ---
def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    return x * (1 - x)

# --- Step 4: Training data (XOR pattern) ---
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# --- Step 5: Training loop ---
learning_rate = 0.1
epochs = 10_000
losses = []

for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_hat = sigmoid(z2)

    # Compute loss (Mean Squared Error)
    loss = np.mean((y - y_hat) ** 2)
    losses.append(loss)

    # Backpropagation (mathematical optimisation)
    d_loss = 2 * (y_hat - y)
    d_z2 = d_loss * sigmoid_derivative(y_hat)
    d_W2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)
    
    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * sigmoid_derivative(a1)
    d_W1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    # Update weights using gradient descent
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2

# --- Step 6: Plot the loss curve ---
plt.figure(figsize=(8, 5))
plt.plot(losses, colour="royalblue", linewidth=2)
plt.title("Neural Network Training Loss Curve", fontsize=14)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss (Mean Squared Error)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Save the figure before displaying it
plt.savefig("loss_curve.png", dpi=300)
plt.show()

# --- Step 7: Display final predictions ---
print("\nFinal predictions:")
print(y_hat.round(3))
