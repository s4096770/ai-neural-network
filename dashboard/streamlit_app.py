import streamlit as st
import numpy as np
import pandas as pd
from src.nn.network import NeuralNet

st.set_page_config(
    page_title="AI Cyber Intrusion Detection",
    layout="wide",
    page_icon="🛡"
)

st.title("🛡 AI Cyber Intrusion Detection Dashboard")
st.caption("Neural Network powered real-time threat analysis interface")

# ================================
# LOAD DATA + TRAIN MODEL ONCE
# ================================
@st.cache_resource
def load_model_and_data():
    data = pd.read_csv("data/security_dataset.csv")

    X = data[["packet_size", "connection_rate", "login_attempts"]].values
    y = data[["attack"]].values

    model = NeuralNet(layers=[3, 6, 1], activation="sigmoid", lr=0.1)
    model.train(X, y, epochs=1500)

    return model, data


model, dataset = load_model_and_data()

# ================================
# LIVE TRAFFIC SIMULATION PANEL
# ================================
st.sidebar.header("📡 Simulate Network Traffic")

packet_size = st.sidebar.slider("Packet Size (bytes)", 40, 2000, 600)
connection_rate = st.sidebar.slider("Connection Rate (per min)", 1, 200, 20)
login_attempts = st.sidebar.slider("Login Attempts", 0, 20, 2)

if st.sidebar.button("🔍 Analyse Traffic"):
    input_data = np.array([[packet_size, connection_rate, login_attempts]])
    result = model.predict(input_data)[0][0]

    st.subheader("⚠ Threat Evaluation Result")

    if result == 1:
        st.error("🚨 INTRUSION DETECTED — MALICIOUS ACTIVITY")
    else:
        st.success("✅ NORMAL TRAFFIC — SAFE CONNECTION")

    st.metric("Threat Confidence", f"{result * 100:.2f}%")

# ================================
# DATA MONITORING FEED
# ================================
st.subheader("📊 Recent Network Activity")
st.dataframe(dataset.head(30))

# ================================
# MODEL PERFORMANCE GRAPH
# ================================
st.subheader("📉 Training Loss Curve")

if hasattr(model, "loss_history") and len(model.loss_history) > 0:
    st.line_chart(model.loss_history)
else:
    st.info("Loss tracking not enabled yet.")
