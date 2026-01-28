# app/pages/0-graph.py
import streamlit as st
import requests
import seaborn as sns
import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_PORT = int(os.getenv('FASTAPI_PORT'))
API_URL = os.getenv('API_ROOT_URL')
API_ROOT_URL = f"http://{API_URL}:{API_PORT}"

st.title("Afficher un graph")
st.header("Selection des colonnes")

if "DATASET_NAME" in st.session_state:
    DATASET_NAME = st.session_state["DATASET_NAME"]
else:
    st.error("Le dataset est introuvable. Revenez à l'accueil.")

if "CURRENT_DATASET" in st.session_state:
    CURRENT_DATASET = st.session_state["CURRENT_DATASET"]
else:
    st.error("Le dataset est introuvable. Revenez à l'accueil.")


st.write(f"Vous avez Selectionné le dataset {DATASET_NAME}")
route_columns = API_ROOT_URL + "/columns"
json_dataset = {"name": DATASET_NAME}
response_columns = requests.post(route_columns, json=json_dataset)
if response_columns.status_code == 200:
    json_columns = response_columns.json()
    list_columns = json_columns["columns"]
    list_columns_None = [None] + list_columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        x_axis  = st.selectbox("Choisissez l'axe X", list_columns)
    with col2:
        y_axis  = st.selectbox("Choisissez l'axe Y", list_columns)
    with col3:
        color  = st.selectbox("Choisissez la couleur", list_columns_None)
    with col4:
        size  = st.selectbox("Choisissez la taille", list_columns_None)
else:
    st.error("Impossible de récupérer les colonnes")

if st.button("Générer le graph"):
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=x_axis, 
        y=y_axis, 
        hue=color, 
        size=size, 
        data=CURRENT_DATASET,
        alpha=0.5,
        ax=ax)
    if color or size:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

