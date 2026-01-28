# app/pages/0-graph.py
import streamlit as st
import requests
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_PORT = int(os.getenv('FASTAPI_PORT'))
API_URL = os.getenv('API_ROOT_URL')
API_ROOT_URL = f"http://{API_URL}:{API_PORT}"

st.title("Entrainer un model")
st.header("Selectionner les données pour l'entrainement")

if "DATASET_NAME" in st.session_state:
    DATASET_NAME = st.session_state["DATASET_NAME"]
else:
    st.error("Le dataset est introuvable. Revenez à l'accueil.")

if "CURRENT_DATASET" in st.session_state:
    CURRENT_DATASET = st.session_state["CURRENT_DATASET"]
else:
    st.error("Le dataset est introuvable. Revenez à l'accueil.")


st.write(f"Vous allez entraîner un modèle sur le dataset {DATASET_NAME}")
route_train = API_ROOT_URL + "/train"
col1, col2, col3 = st.columns(3)
with col1:
    list_solvers = ["lbfgs", "newton-cg", "sag", "saga", "newton-cholesky"]
    solver  = st.selectbox("Choisissez le solver", list_solvers)

with col2:
    max_iter  = st.slider("Choisissez le max_iter", min_value=100, max_value=3000, step=100, value = 1000)

with col3:
    exposant = st.slider(
    "Choisissez le C",
    min_value=-3.0,
    max_value=3.0,
    value=0.0,
    step=0.1)

    C = 10**exposant
    # 4. Affichage du résultat avec un formatage propre
    st.write(f"C : `{C:.4f}`")

json_train = {
    "dataset_name" : DATASET_NAME,
    "dataset" : CURRENT_DATASET,
    "max_iter" : max_iter,
    "C" : C,
    "solver" : solver
}

if st.button("Entrainer le modèle"):
    route_train = API_ROOT_URL + "/train"
    response = requests.post(route_train, json=json_train)
    if response.status_code == 200:
        response_json = response.json()
        st.markdown("**Le modèle a été entrainé avec succès !**")
        
        st.markdown("Confusion matrix : ")
        matrix_data = response_json["confusion_matrix"]["matrix"]
        
        labels = response_json["confusion_matrix"].get("labels", range(len(matrix_data)))
    
        conf_df = pd.DataFrame(matrix_data, index=labels, columns=labels)

        fig, ax = plt.subplots()
        sns.heatmap(
            conf_df, 
            annot=True,           # Affiche les nombres
            fmt='d',              # Entiers
            cmap='GnBu',          # Dégradé plus moderne (Green-Blue)
            cbar=False,           # Pas de légende à droite
            linewidths=.5,        # Lignes de séparation entre les cases
            linecolor='w',# Couleur des lignes
            annot_kws={"size": 14, "weight": "bold"}, # Chiffres plus gros et gras
            ax=ax,
            xticklabels=False,
            yticklabels=False
        )
        # 3. Cosmétique des axes
        ax.set_xlabel("Valeurs Prédites", fontsize=10, labelpad=10)
        ax.set_ylabel("Valeurs Réelles", fontsize=10, labelpad=10)
        
        # Rotation des labels pour plus de lisibilité
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        st.pyplot(fig)
        st.markdown("Classification report : ")
        st.markdown(f"```\n{response_json['classification_report_test']}\n```")
        # CURRENT_DATASET = response_json["dataset"]
        # st.dataframe(CURRENT_DATASET, width="stretch")
    else : 
        st.error(f"L'API a répondu avec une erreur : {response.status_code}")
