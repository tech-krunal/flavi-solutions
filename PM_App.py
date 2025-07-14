import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
from streamlit_echarts import st_echarts
from collections import deque
import matplotlib.pyplot as plt
import pickle

model = joblib.load('Predict_model.pkl')  #load model 

st.set_page_config(page_title="Predictive Maintenance", layout="wide") #page setting
st.title("🔧 Predictive Maintenance App")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        color: black; 
        background-color: #9BE1FA;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("Get ready to fine-tune your predictions! Tweak the parameters below to uncover the likelihood that the machine might need some maintenance.")

# Sidebar 
st.sidebar.markdown(
    "<h1 style='color: #00172b;'>Machine Parameters</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] {
        background-color: pink;
    }
    .stSlider > div[data-baseweb="slider"] > div > div > div {
        background: linear-gradient(to right, #108ffd 0%, #083a71 50%, rgba(172, 177, 195, 0.25) 50%, rgba(172, 177, 195, 0.25) 100%);
        background-color: yellow;
    }

    .stSlider label {
        color: darkblue;
    }
    </style>
    """,
    unsafe_allow_html=True
)


load_percentage = st.sidebar.slider('Load Percentage (%)', min_value=0, max_value=100, value=50, step=5)
motor_current = st.sidebar.slider('Motor Current (A)', min_value=0.5, max_value=40.0, value=10.0, step=0.5)

st.markdown(
    """
    <style>
    .stNumberInput label {
        color: DarkBlue;
    }
    </style>
    """,
    unsafe_allow_html=True
)

rpm = st.sidebar.number_input('RPM', min_value=500, max_value=2500, value=1500, step=1)
bearing_temp = st.sidebar.number_input('Bearing Temperature (°C)', min_value=30.0, max_value=150.0, value=49.0, step=1.0)
vibration_magnitude = st.sidebar.number_input('Vibration Magnitude', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
temp_diff = st.sidebar.number_input('Temperature Difference (°C)', min_value=0.0, max_value=100.0, value=25.0, step=1.0)

st.subheader("📊 Input Gauges")

def gauge_options(value, min_val, max_val, name):
    return {
        "series": [{
            "type": 'gauge',
            "startAngle": 180,
            "endAngle": 0,
            "min": min_val,
            "max": max_val,
            "splitNumber": 5,
            "axisLine": {"lineStyle": {"width": 10}},
            "pointer": {"icon": 'rect', "width": 6, "length": '70%'},
            "title": {"offsetCenter": [0, "60%"], "fontSize": 16},
            "lineStyle": {
                    "width": 10,
                    "color": [
                        [1, "#9BE1FA"]
                    ]},
            "detail": {
                "valueAnimation": True,
                "formatter": f"{value}",
                "offsetCenter": [0, "20%"],
                "color": "white"
            },
            "data": [{"value": value, "name": name}]
        }]
    }

col1, col2, col3 = st.columns(3)
with col1:
    st_echarts(gauge_options(rpm, 500, 3000, "RPM"), height="250px")
with col2:
    st_echarts(gauge_options(load_percentage, 0, 100, "Load %"), height="250px")
with col3:
    st_echarts(gauge_options(motor_current, 0.5, 40, "Current A"), height="250px")

# ---------------------------
# Prediction
# ---------------------------

input_features = np.array([[rpm, load_percentage, bearing_temp, motor_current, vibration_magnitude, temp_diff]])

try:
    prediction = model.predict(input_features)
    probs = model.predict_proba(input_features)

    if prediction[0] == 0:
        st.markdown(f"<h3 style='color: green;'>✅ Normal state </h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color: red;'>⚠️ Predicted Fault Code: {prediction[0]} (Error). Maintenance required</h3>", unsafe_allow_html=True)

    st.write("Class Probabilities:")
    st.write({f"Class {i}": f"{prob:.2f}" for i, prob in enumerate(probs[0])})

except Exception as e:
    st.error(f"Error predicting: {e}")

# Prediction trend plot

if 'history' not in st.session_state:
    st.session_state.history = deque(maxlen=50)
    st.session_state.time_steps = deque(maxlen=50)

# Save new prediction
st.session_state.history.append(prediction[0])
st.session_state.time_steps.append(len(st.session_state.history))

st.subheader("📈 Fault Code Trend (last 50 predictions)")

fig, ax = plt.subplots(figsize=(10, 4), facecolor="#00172b")
ax.set_facecolor("#00172b") 
ax.plot(
    st.session_state.time_steps,
    st.session_state.history,
    marker='o',
    color='green' if prediction[0] == 0 else 'red'
)
ax.set_xlabel("Time step")
ax.set_ylabel("Predicted Fault Code")
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.set_ylim(-0.5, max(3, max(st.session_state.history) + 0.5))
ax.grid(True, color='white', linestyle='--', alpha=0.5)
st.pyplot(fig)

# Show input values

with st.expander("Show Input Values"):
    st.write({
        'RPM': rpm,
        'Load Percentage': load_percentage,
        'Bearing Temp': bearing_temp,
        'Motor Current': motor_current,
        'Vibration Magnitude': vibration_magnitude,
        'Temp Diff': temp_diff
    })
