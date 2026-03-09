import streamlit as st
import pickle
import pandas as pd
import numpy as np
st.set_page_config(page_title="IPL Predictor", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 4rem;
    padding-right: 4rem;
}
</style>
""", unsafe_allow_html=True)
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load saved files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("🏏 IPL Win Probability Predictor")

teams = [
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Mumbai Indians",
    "Chennai Super Kings",
    "Delhi Capitals",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Punjab Kings"
]

batting_team = st.selectbox("Batting Team", teams)
bowling_team = st.selectbox("Bowling Team", teams)

runs_left = st.number_input("Runs Left", min_value=0)
balls_left = st.number_input("Balls Left", min_value=0, max_value=120)
wickets_left = st.number_input("Wickets Left", min_value=0, max_value=10)
target = st.number_input("Target", min_value=0)
crr = st.number_input("Current Run Rate")
rrr = st.number_input("Required Run Rate")

if st.button("Predict"):

    input_data = pd.DataFrame({
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'target': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Add team columns
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0

    if f'batting_team_{batting_team}' in input_data.columns:
        input_data[f'batting_team_{batting_team}'] = 1

    if f'bowling_team_{bowling_team}' in input_data.columns:
        input_data[f'bowling_team_{bowling_team}'] = 1

    input_data = input_data[columns]

    input_scaled = scaler.transform(input_data)

    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Winning Probability")

    st.success(f"{batting_team}: {round(probability[1]*100,2)}%")

    st.error(f"{bowling_team}: {round(probability[0]*100,2)}%")

