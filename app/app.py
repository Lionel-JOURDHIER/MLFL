# app/app.py
import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_PORT = int(os.getenv('FASTAPI_PORT'))
API_URL = os.getenv('API_ROOT_URL')
API_ROOT_URL = f"http://{API_URL}:{API_PORT}"



st.title("Bienvenue dans votre application pour générer des Model de Machine Learning")
st.header("Selection du Dataset")
dataset = st.selectbox("Choisissez le dataset", ["iris", "penguins", "titanic", "churn"])
if st.button("ping l'API (Route /)"):
    response = requests.get(API_ROOT_URL)