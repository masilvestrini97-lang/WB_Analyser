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
st.set_page_config(page_title="Quantificateur WB V3", layout="wide")
st.title("🔬 Quantificateur de Western Blot (V3)")
st.markdown("Utilisez les curseurs de gauche pour définir la zone de lecture exacte (sans légendes), puis affinez à droite.")

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
        st.subheader("1. Définir la Piste Active (segment rouge)")
        
        # Curseurs pour définir le segment de lecture
        x_center = st.slider("Position X", 0, width, int(width/2))
        lane_y_range = st.slider(
            "Plage Verticale de la Piste (Exclure les légendes)", 
            0, height, (int(height*0.1), int(height*0.9)),
            help="Ce segment rouge définit la zone d'où proviendra le profil d'intensité."
        )
        
        # Variables de plage Y
        y_start, y_end = lane_y_range
        
        # Sécurité pour éviter les erreurs d'extraction
        if y_end <= y_start + 1:
            st.error("Veuillez sélectionner une plage verticale valide (début < fin).")
            st.stop()
            
        # Dessin du segment rouge sur l'image
        img_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # On dessine un segment, pas une ligne complète
        cv2.line(img_display, (x_center, y_start), (x_center, y_end), (255, 0, 0), 4)
        # Petites marques horizontales pour bien voir les bornes
        cv2.line(img_display, (x_center-5, y_start), (x_center+5, y_start), (255, 0, 0), 4)
        cv2.line(img_display, (x_center-5, y_end), (x_center+5, y_end), (255, 0, 0), 4)
        
        st.image(img_display, caption=f"Piste sélectionnée (X:{x_center}, Y:{y_start}-{y_end})", use_container_width=True)

    # --- Étape 2 : Extraction et Affinement ---
    # On n'extrait la colonne que dans la plage Y définie à gauche
    full_col_profile = inverted_img[:, x_center]
    # 'active_profile' est le profil PROPRE correspondant AU SEGMENT ROUGE
    active_profile = full_col_profile[y_start:y_end]
    profile_len = len(active_profile)

    with col2:
        st.subheader("2. Sélectionner la Bande d'Intérêt (dans le segment)")
        # Curseurs d'ajustement du pic, relatifs au début du segment propre
        peak_y_range = st.slider(
            "Isoler le pic de protéine (Début/Fin dans le segment)", 
            0, profile_len, (int(profile_len*0.2), int(profile_len*0.4)),
            help="Sélectionnez le début et la fin de votre pic à l'intérieur du graphique ci-dessous."
        )
        
        peak_start, peak_end = peak_y_range

        # Graphique du profil PROPRE (active_profile)
        fig, ax = plt.subplots(figsize=(6, 4))
        # Axe des X = Position relative dans le segment propre (0 = Y Absolu Début Piste)
        ax.plot(range(profile_len), active_profile, color='black', label="Profil d'intensité propre")
        ax.fill_between(range(peak_start, peak_end), active_profile[peak_start:peak_end], color='blue', alpha=0.3, label="Bande ciblée")
        ax.set_xlim(0, profile_len)
        ax.set_xlabel("Position Relative dans la Piste (Pixels)")
        ax.set_ylabel("Intensité")
        ax.invert_xaxis() # Conserver l'orientation haut-bas intuitive
        ax.legend()
        st.pyplot(fig)

        # Calcul
        band_profile = active_profile[peak_start:peak_end]
        if len(band_profile) > 0:
            local_background = np.min(band_profile)
            net_profile = band_profile - local_background
            area = np.trapezoid(net_profile)
            
            st.success(f"**Intensité calculée : {area:.2f}**")
            
            # --- Étape 3 : Sauvegarde avec coordonnées absolues ---
            st.markdown("---")
            st.subheader("3. Sauvegarder dans le tableau")
            band_name = st.text_input("Nom de la condition (ex: WT 7min p-PLCG1)")
            
            if st.button("➕ Ajouter au tableau"):
                if band_name:
                    # Conversion en coordonnées absolues de l'image originale pour le tableau
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

# --- Affichage et Export (inchangé, mais tableau mis à jour) ---
if not st.session_state.results_df.empty:
    st.markdown("---")
    st.subheader("📊 Tableau de vos résultats")
    st.dataframe(st.session_state.results_df, use_container_width=True)
    
    csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les résultats (CSV)",
        data=csv,
        file_name="quantification_ western_blot.csv",
        mime="text/csv",
    )
    
    if st.button("🗑️ Vider le tableau"):
        st.session_state.results_df = st.session_state.results_df.iloc[0:0] # Méthode plus propre pour vider
        st.rerun()
