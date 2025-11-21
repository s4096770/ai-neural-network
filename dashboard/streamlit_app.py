import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

# ==============================
# PATH FIX FOR STREAMLIT CLOUD
# ==============================

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from nn.network import NeuralNet


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="AI Cyber Intrusion Detection",
    layout="wide",
    page_icon="🛡️"
)

st.title("🛡️ AI Cyber Intrusion Detection Dashboard")
st.caption("Neural Network powered real-time threat analysis interface")


# ==============================
# LOAD DATA + TRAIN MODEL
# ==============================

@st.cache_resource
def load_model_and_data():
    data = pd.read_csv("data/security_dataset.csv")

    X = data[["packet_size", "connection_rate", "login_attempts"]].values
    y = data[["attack"]].values

    model = NeuralNet(
        layers=[3, 8, 1],
        activation="sigmoid",
        lr=0.3
    )
    model.train(X, y, epochs=1000)

    return model, data


model, dataset = load_model_and_data()


# ==============================
# SIDEBAR - SIMULATION CONTROLS
# ==============================

st.sidebar.header("🖧 Simulate Network Traffic")

packet_size = st.sidebar.slider("Packet Size (bytes)", 100, 2000, 600)
connection_rate = st.sidebar.slider("Connection Rate (per min)", 1, 200, 20)
login_attempts = st.sidebar.slider("Login Attempts", 0, 20, 2)

analyse_btn = st.sidebar.button("🔍 Analyse Traffic")


# ==============================
# MAIN DASHBOARD
# ==============================

st.subheader("📊 Recent Network Activity")
st.dataframe(dataset.head(20))


if analyse_btn:
    input_data = np.array([[packet_size, connection_rate, login_attempts]])
    prediction = model.predict(input_data)[0][0]

    st.subheader("🚨 Threat Analysis Result")

    if prediction >= 0.5:
        st.error(f"⚠️ Intrusion Detected! Risk Score: {prediction:.2f}")
    else:
        st.success(f"✅ Normal Traffic. Risk Score: {prediction:.2f}")


# ==============================
# TRAINING VISUAL (LOSS PLACEHOLDER)
# ==============================

st.subheader("📉 Training Loss Curve")
st.line_chart(model.loss_history)
