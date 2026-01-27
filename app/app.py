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
DATASET_NAME = st.selectbox("Choisissez le dataset", ["iris", "penguins", "titanic", "churn"])
json_dataset = {"name": DATASET_NAME}
CURRENT_DATASET = {}
if st.button("Telecharger le Dataset"):
    route_dataset = API_ROOT_URL + "/dataset"
    response = requests.post(route_dataset, json=json_dataset)
    if response.status_code == 200:
            response_json = response.json()
            CURRENT_DATASET = response_json["dataset"]
            st.dataframe(CURRENT_DATASET, width="stretch")
    else : 
        st.error(f"L'API a répondu avec une erreur : {response.status_code}")

st.session_state["CURRENT_DATASET"] = CURRENT_DATASET
st.session_state["DATASET_NAME"] = DATASET_NAME