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
            0, profile_len, (int(profile_len*0.2), int(profile_len*0.4))
        )
        
        peak_start, peak_end = peak_y_range

        # --- NOUVEAU GRAPHIQUE VERTICAL ---
        fig, ax = plt.subplots(figsize=(4, 6)) # Plus haut que large
        
        # On inverse les axes : X devient l'intensité, Y devient la position
        ax.plot(active_profile, range(profile_len), color='black', label="Profil d'intensité")
        
        # fill_betweenx permet de colorier de gauche à droite au lieu de bas en haut
        ax.fill_betweenx(range(peak_start, peak_end), 0, active_profile[peak_start:peak_end], color='blue', alpha=0.3, label="Bande ciblée")
        
        # On inverse l'axe Y pour que le point 0 (haut de la piste) soit en haut du graphique
        ax.set_ylim(profile_len, 0) 
        
        # Limite de l'axe X (Intensité) pour garder une échelle propre
        max_intensity = np.max(active_profile) if np.max(active_profile) > 0 else 255
        ax.set_xlim(0, max_intensity * 1.1)
        
        ax.set_ylabel("Position (Pixels de haut en bas)")
        ax.set_xlabel("Intensité")
        ax.legend()
        st.pyplot(fig)

        # Calcul
        band_profile = active_profile[peak_start:peak_end]
        if len(band_profile) > 0:
            local_background = np.min(band_profile)
            net_profile = band_profile - local_background
            area = np.trapezoid(net_profile)
            
            st.success(f"**Intensité calculée : {area:.2f}**")
            
            # --- Étape 3 : Sauvegarde ---
            st.markdown("---")
            st.subheader("3. Sauvegarder dans le tableau")
            band_name = st.text_input("Nom de la condition (ex: WT 7min p-PLCG1)")
            
            if st.button("➕ Ajouter au tableau"):
                if band_name:
                    abs_peak_start = y_start + peak_start
                    abs_peak_end = y_start + peak_end
                    
                    new_data = pd.DataFrame([{
                        "Condition / Protéine": band_name, 
                        "X Absolu": x_center, 
                        "Y Absolu Début Piste": y_start,
                        "Y Absolu Fin Piste": y_end,
                        "Y Absolu Début Bande": abs_peak_start,
                        "Y Absolu Fin Bande": abs_peak_end,
                        "Intensité brute (Aire)": round(area, 2),
                        "Bruit de fond Local": round(local_background, 2)
                    }])
                    st.session_state.results_df = pd.concat([st.session_state.results_df, new_data], ignore_index=True)
                else:
                    st.warning("Veuillez entrer un nom pour cette bande.")

# --- Affichage et Export ---
if not st.session_state.results_df.empty:
    st.markdown("---")
    st.subheader("📊 Tableau de vos résultats")
    st.dataframe(st.session_state.results_df, use_container_width=True)
    
    csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les résultats (CSV)",
        data=csv,
        file_name="quantification_western_blot.csv",
        mime="text/csv",
    )
    
    if st.button("🗑️ Vider le tableau"):
        st.session_state.results_df = st.session_state.results_df.iloc[0:0]
        st.rerun()
