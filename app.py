import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Title and Description ---
st.title("🧠 Neural Network Visualiser — XOR Problem")
st.write("""
This interactive app demonstrates a simple **neural network** trained on the XOR problem.  
You can explore how the model predicts outputs and visualise its training loss curve.
""")

# --- Define Sigmoid Function ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --- Network Parameters (trained weights) ---
W1 = np.array([[5.63, 5.40, 5.14],
               [5.55, 5.49, 5.22]])
b1 = np.array([[-2.74, -2.64, -2.56]])
W2 = np.array([[7.37], [7.35], [-11.3]])
b2 = np.array([[-3.6]])

# --- User Input ---
st.subheader("🔢 Try Your Own Input")
x1 = st.slider("Input 1", 0.0, 1.0, 0.0, 0.01)
x2 = st.slider("Input 2", 0.0, 1.0, 0.0, 0.01)
X = np.array([[x1, x2]])

# --- Forward Pass ---
z1 = np.dot(X, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
y_hat = sigmoid(z2)

st.write(f"### ✅ Predicted Output: {float(y_hat):.3f}")

# --- Loss Curve Example ---
losses = [0.3182, 0.1420, 0.0202, 0.0063, 0.0034, 0.0023, 0.0017, 0.0014, 0.0011, 0.0010]
epochs = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]

st.subheader("📉 Training Loss Curve")
fig, ax = plt.subplots()
ax.plot(epochs, losses, marker='o', linestyle='-', color='royalblue')
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Training Progress")
ax.grid(True)
st.pyplot(fig)
