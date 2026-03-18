import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from scipy.signal import find_peaks
import io

# --- Initialisation de la mémoire (Session State) pour le tableau ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=[
        "Condition / Protéine", 
        "X Absolu", 
        "Y Absolu Début Piste", 
        "Y Absolu Fin Piste",
        "Position Pic Relative (Y)",
        "Position Pic Absolue (Y)",
        "Intensité brute (Aire)", 
        "Bruit de fond Local"
    ])

# --- Configuration de la page ---
st.set_page_config(page_title="Quantificateur WB V5 Semi-Auto", layout="wide")
st.title("🔬 Quantificateur de Western Blot (V5 Semi-Automatique)")
st.markdown("Utilisez les curseurs de gauche pour définir la zone de lecture (en rouge), puis réglez la **sensibilité de détection** pour que l'algorithme trouve vos pics à droite.")

# --- Étape 1 : Chargement de l'image ---
uploaded_file = st.file_uploader("Choisissez une image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lecture de l'image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Conversion en niveaux de gris si l'image est en couleur
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Inversion des couleurs pour l'analyse (noir = intensité)
    inverted_img = 255 - gray
    height, width = inverted_img.shape

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Définir la Piste Active")
        
        # Curseurs pour définir le segment de lecture
        x_center = st.slider("Position X", 0, width, int(width/2))
        lane_y_range = st.slider(
            "Plage Verticale de la Piste (Exclure les légendes)", 
            0, height, (int(height*0.1), int(height*0.9)),
            help="Ce segment rouge définit la zone d'où proviendra le profil d'intensité propre."
        )
        
        # Variables de plage Y
        y_start, y_end = lane_y_range
        
        # Sécurité pour éviter les erreurs d'extraction
        if y_end <= y_start + 1:
            st.error("Veuillez sélectionner une plage verticale valide (début < fin).")
            st.stop()
            
        # Dessin du segment rouge sur l'image pour visualisation
        img_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.line(img_display, (x_center, y_start), (x_center, y_end), (255, 0, 0), 4)
        # Marques horizontales
        cv2.line(img_display, (x_center-10, y_start), (x_center+10, y_start), (255, 0, 0), 4)
        cv2.line(img_display, (x_center-10, y_end), (x_center+10, y_end), (255, 0, 0), 4)
        
        st.image(img_display, caption=f"Piste sélectionnée (X:{x_center}, Y:{y_start}-{y_end})", use_container_width=True)

    # --- Étape 2 : Extraction et Détection Semi-Automatique ---
    # On n'extrait la colonne que dans la plage Y définie à gauche
    full_col_profile = inverted_img[:, x_center]
    active_profile = full_col_profile[y_start:y_end]
    profile_len = len(active_profile)

    with col2:
        st.subheader("2. Détection de Pics (Profil Vertical)")
        
        # --- NOUVEAUX CURSEURS DE DÉTECTION ---
        threshold = st.slider(
            "Sensibilité de détection (Intensité minimale)", 
            0, 255, 30, 
            help="Un pic doit dépasser ce niveau d'intensité pour être détecté. Augmentez pour filtrer le bruit."
        )
        min_width = st.slider(
            "Largeur minimale du pic (en pixels Y)", 
            1, 100, 10,
            help="Un pic doit avoir au moins cette largeur pour être retenu. Utile pour filtrer les micro-bulles de 1 pixel."
        )

        # --- NOUVELLE FONCTION DE DÉTECTION (Scipy) ---
        # `find_peaks` est l'outil parfait. Nous l'inversons car on cherche des bandes SOMBRES sur fond blanc.
        peaks, properties = find_peaks(active_profile, height=threshold, width=min_width)
        num_peaks = len(peaks)
        
        # Calcul des intensités brutes (Aire sous la courbe) pour chaque pic détecté
        peaks_data = []
        for i in range(num_peaks):
            peak_relative_y = peaks[i]
            # Récupération des bases du pic (gauche/droite) pour l'intégration, relatives au début du segment
            peak_left_base = int(properties["left_bases"][i])
            peak_right_base = int(properties["right_bases"][i])
            
            # Extraction du profil de ce pic unique
            band_profile = active_profile[peak_left_base:peak_right_base]
            
            # Calcul de l'intensité
            local_background = np.min(band_profile) if len(band_profile) > 0 else 0
            net_profile = band_profile - local_background
            
            # Règle des trapèzes pour une aire plus robuste (Numpy 2.0+ compatible)
            if len(net_profile) > 0:
                area = np.trapezoid(net_profile)
                
                # Conversion en coordonnées absolues de l'image originale
                abs_peak_y = y_start + peak_relative_y
                
                # Stockage des données
                peaks_data.append({
                    "X Absolu": x_center, 
                    "Y Absolu Début Piste": y_start,
                    "Y Absolu Fin Piste": y_end,
                    "Position Pic Relative (Y)": peak_relative_y,
                    "Position Pic Absolue (Y)": abs_peak_y,
                    "Intensité brute (Aire)": round(area, 2),
                    "Bruit de fond Local": round(local_background, 2)
                })

        # --- NOUVEAU GRAPHIQUE VERTICAL AMÉLIORÉ ---
        fig, ax = plt.subplots(figsize=(4, 6)) # Plus haut que large
        
        # On inverse les axes : X devient l'intensité, Y devient la position relative
        ax.plot(active_profile, range(profile_len), color='black', label="Profil d'intensité propre")
        
        # Colorier CHAQUE pic individuellement
        for i in range(num_peaks):
            peak_relative_y = peaks[i]
            peak_left_base = int(properties["left_bases"][i])
            peak_right_base = int(properties["right_bases"][i])
            
            # fill_betweenx permet de colorier horizontalement
            ax.fill_betweenx(range(peak_left_base, peak_right_base), 0, active_profile[peak_left_base:peak_right_base], color='blue', alpha=0.3)
            # Marqueur au sommet
            ax.scatter(active_profile[peak_relative_y], peak_relative_y, color='blue', marker='o', s=50, edgecolors='white', linewidths=1)
        
        # On inverse l'axe Y pour que le point 0 (haut de la piste rouge) soit en haut
        ax.set_ylim(profile_len, 0) 
        # Limite de l'axe X (Intensité) propre
        max_intensity = np.max(active_profile) if np.max(active_profile) > 0 else 255
        ax.set_xlim(0, max_intensity * 1.1)
        
        ax.set_ylabel("Position Relative dans la Piste (Pixels)")
        ax.set_xlabel("Intensité (noir = fort)")
        
        # Mise à jour de la légende
        handles, labels = ax.get_legend_handles_labels()
        if num_peaks > 0:
            # On crée un faux marqueur pour la légende
            h, = ax.plot([], [], color='blue', marker='o', linestyle='None')
            handles.append(h)
            labels.append(f"Pics détectés ({num_peaks})")
        
        ax.legend(handles, labels)
        st.pyplot(fig)
        
        if num_peaks == 0:
            st.warning("Aucun pic détecté avec la sensibilité actuelle. Essayez d'augmenter la sensibilité (Intensité minimale).")
        else:
            st.success(f"**{num_peaks} pics détectés.** Les résultats sont prêts à être enregistrés.")

        # --- Étape 3 : Sauvegarde instantanée de TOUS les pics ---
        st.markdown("---")
        st.subheader("3. Enregistrer les résultats de cette piste")
        band_base_name = st.text_input("Nom de base pour cette piste (ex: WT 7min)")
        
        if st.button("➕ Enregistrer TOUS les pics détectés"):
            if band_base_name and len(peaks_data) > 0:
                # Création d'un DataFrame temporaire avec les noms de pics numérotés
                new_data_list = []
                for idx, peak_data in enumerate(peaks_data):
                    # Exemple de nom : "WT 7min (Pic 1 / 155 kDa)"
                    peak_data_named = peak_data.copy()
                    peak_data_named["Condition / Protéine"] = f"{band_base_name} (Pic {idx + 1})"
                    new_data_list.append(peak_data_named)
                
                new_data_df = pd.DataFrame(new_data_list)
                
                # Ajout au tableau global
                st.session_state.results_df = pd.concat([st.session_state.results_df, new_data_df], ignore_index=True)
                st.toast("Pics enregistrés avec succès !")
            else:
                if len(peaks_data) == 0:
                    st.warning("Aucun pic à enregistrer. Réglez d'abord la sensibilité.")
                else:
                    st.warning("Veuillez entrer un nom de base pour cette piste.")

# --- Affichage et Export (inchangé) ---
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
        st.session_state.results_df = st.session_state.results_df.iloc[0:0]
        st.rerun()
