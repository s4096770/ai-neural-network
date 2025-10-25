import numpy as np

# --- Step 1: Define the architecture ---
inputs = 2      # input features
hidden = 3      # hidden neurons
outputs = 1     # output neuron

# --- Step 2: Initialize weights and biases ---
np.random.seed(42)
W1 = np.random.randn(inputs, hidden)
b1 = np.zeros((1, hidden))
W2 = np.random.randn(hidden, outputs)
b2 = np.zeros((1, outputs))

# --- Step 3: Activation function ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# --- Step 4: Training data (XOR pattern) ---
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# --- Step 5: Training loop ---
lr = 0.1
for epoch in range(10000):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_hat = sigmoid(z2)

    # Compute loss
    loss = np.mean((y - y_hat) ** 2)

    # Backpropagation
    d_loss = 2 * (y_hat - y)
    d_z2 = d_loss * sigmoid_derivative(y_hat)
    d_W2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)
    
    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * sigmoid_derivative(a1)
    d_W1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    # Update weights
    W1 -= lr * d_W1
    b1 -= lr * d_b1
    W2 -= lr * d_W2
    b2 -= lr * d_b2

    # Print progress occasionally
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nFinal predictions:")
print(y_hat.round(3))
