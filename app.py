import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import io

# --- Initialisation de la mémoire (Session State) ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=[
        "Condition / Protéine", 
        "X Absolu", 
        "Y Absolu Début Piste", 
        "Y Absolu Fin Piste",
        "Y Absolu Début Bande",
        "Y Absolu Fin Bande",
        "Intensité brute (Aire)", 
        "Bruit de fond Local"
    ])

# --- Configuration de la page ---
st.set_page_config(page_title="Quantificateur WB V4", layout="wide")
st.title("🔬 Quantificateur de Western Blot (V4)")
st.markdown("Utilisez les curseurs de gauche pour définir la zone de lecture, puis affinez le pic sur le **graphique vertical** à droite.")

# --- Étape 1 : Chargement de l'image ---
uploaded_file = st.file_uploader("Choisissez une image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    inverted_img = 255 - gray
    height, width = inverted_img.shape

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Définir la Piste Active")
        
        x_center = st.slider("Position X", 0, width, int(width/2))
        lane_y_range = st.slider(
            "Plage Verticale de la Piste (Exclure les légendes)", 
            0, height, (int(height*0.1), int(height*0.9))
        )
        
        y_start, y_end = lane_y_range
        
        if y_end <= y_start + 1:
            st.error("Veuillez sélectionner une plage verticale valide (début < fin).")
            st.stop()
            
        img_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.line(img_display, (x_center, y_start), (x_center, y_end), (255, 0, 0), 4)
        cv2.line(img_display, (x_center-5, y_start), (x_center+5, y_start), (255, 0, 0), 4)
        cv2.line(img_display, (x_center-5, y_end), (x_center+5, y_end), (255, 0, 0), 4)
        
        st.image(img_display, caption=f"Piste sélectionnée (X:{x_center}, Y:{y_start}-{y_end})", use_container_width=True)

    # --- Étape 2 : Extraction et Affinement ---
    full_col_profile = inverted_img[:, x_center]
    active_profile = full_col_profile[y_start:y_end]
    profile_len = len(active_profile)

    with col2:
        st.subheader("2. Isoler la Bande (Profil Vertical)")
        peak_y_range = st.slider(
            "Isoler le pic de protéine (Haut / Bas)", 
            0, profile_len, (int(
