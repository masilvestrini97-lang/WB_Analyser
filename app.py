import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# --- Configuration de la page ---
st.set_page_config(page_title="Quantificateur Western Blot", layout="wide")
st.title("🔬 Quantificateur de Western Blot (Semi-Automatisé)")
st.markdown("Chargez votre image, sélectionnez une piste, et isolez un pic pour obtenir l'intensité (Aire sous la courbe).")

# --- Étape 1 : Chargement de l'image ---
uploaded_file = st.file_uploader("Choisissez une image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lecture de l'image avec Pillow puis conversion pour OpenCV
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Conversion en niveaux de gris si l'image est en couleur
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Inversion des couleurs : le fond blanc (255) devient 0, les bandes noires deviennent des pics positifs
    inverted_img = 255 - gray
    height, width = inverted_img.shape

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Définir la piste (Lane)")
        # Curseurs pour définir la zone d'analyse (colonne)
        x_center = st.slider("Position X (Centre de la piste)", 0, width, int(width/2))
        lane_width = st.slider("Largeur de la piste", 5, 100, 30)
        
        # Calcul des bords de la piste
        x_start = max(0, x_center - lane_width // 2)
        x_end = min(width, x_center + lane_width // 2)

        # Dessiner un rectangle rouge sur l'image pour visualiser la piste
        img_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(img_display, (x_start, 0), (x_end, height), (255, 0, 0), 2)
        
        st.image(img_display, caption="Votre Blot avec la piste sélectionnée", use_column_width=True)

    # --- Étape 2 : Profil de la piste ---
    # On extrait la colonne et on fait la moyenne horizontale pour avoir un profil 1D
    lane_roi = inverted_img[:, x_start:x_end]
    profile = np.mean(lane_roi, axis=1)

    with col2:
        st.subheader("2. Profil d'intensité et Quantification")
        # Curseur pour isoler la bande sur l'axe Y (de haut en bas)
        y_min, y_max = st.slider(
            "Sélectionnez le pic de votre protéine (Haut / Bas)", 
            0, height, (int(height*0.2), int(height*0.4))
        )

        # Tracé du graphique avec Matplotlib
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(height), profile, color='black', label="Profil d'intensité")
        
        # Colorier la zone sélectionnée (le pic)
        ax.fill_between(range(y_min, y_max), profile[y_min:y_max], color='blue', alpha=0.3, label="Bande ciblée")
        
        ax.set_xlim(0, height)
        ax.set_xlabel("Position Y (Pixels de haut en bas)")
        ax.set_ylabel("Intensité (Unités arbitraires)")
        ax.invert_xaxis() # Pour que le haut de l'image soit à gauche du graphique (plus intuitif)
        ax.legend()
        st.pyplot(fig)

        # --- Étape 3 : Calcul ---
        # Méthode simple : Somme des intensités dans la zone sélectionnée (Aire sous la courbe)
        # On soustrait un bruit de fond basique (la valeur la plus basse de la zone sélectionnée)
        band_profile = profile[y_min:y_max]
        if len(band_profile) > 0:
            local_background = np.min(band_profile)
            net_profile = band_profile - local_background
            
            # L'intégration (Aire sous la courbe) en utilisant la règle des trapèzes
            area = np.trapz(net_profile)
            
            st.success(f"**Intensité brute calculée (Aire) : {area:.2f}**")
            st.info(f"Bruit de fond local soustrait : {local_background:.2f}")
        else:
            st.warning("Veuillez sélectionner une zone valide.")
