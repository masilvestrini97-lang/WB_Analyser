import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from scipy.signal import find_peaks

# --- Liste standardisée de vos conditions ---
CONDITIONS = [
    "EV -", "EV 2", "EV 7", "EV 15",
    "WT -", "WT 2", "WT 7", "WT 15",
    "D776N -", "D776N 2", "D776N 7", "D776N 15"
]

# --- Initialisation de la mémoire ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=[
        "Puits N°", "Condition", "Nom du Pic", "X Absolu", "Y Début Bande", "Y Fin Bande", "Intensité (AUC)", "Bruit de fond"
    ])

st.set_page_config(page_title="Quantificateur WB V8 (Contrôle Visuel)", layout="wide")
st.title("🔬 Quantificateur à Grille avec Contrôle des Pics (V8)")
st.markdown("Placez la grille globale à gauche, puis inspectez et validez chaque puits à droite.")

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
        st.subheader("1. Placer la Grille Globale")
        x_first = st.slider("Axe X : Première piste (EV -)", 0, width, int(width*0.1))
        x_last = st.slider("Axe X : Dernière piste (D776N 15)", 0, width, int(width*0.9))
        
        y_range = st.slider(
            "Plage Verticale (Exclure les textes)", 
            0, height, (int(height*0.1), int(height*0.9))
        )
        y_start, y_end = y_range
        
        if y_end <= y_start + 1:
            st.error("Plage verticale invalide.")
            st.stop()

        # Calcul des 12 positions
        x_positions = np.linspace(x_first, x_last, 12, dtype=int)

        # Dessin de la grille
        img_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Rectangle de la zone de recherche
        cv2.rectangle(img_display, (x_first-15, y_start), (x_last+15, y_end), (100, 100, 100), 2, cv2.LINE_AA)
        
        for i, x in enumerate(x_positions):
            cv2.line(img_display, (x, y_start), (x, y_end), (255, 0, 0), 2)
            cv2.putText(img_display, str(i+1), (x-10, y_start-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        st.image(img_display, caption="Grille de lecture (12 puits)", use_container_width=True)

    with col2:
        # --- SÉLECTION DU PUITS À INSPECTER ---
        selected_well = st.selectbox(
            "2. Sélectionner le puits à inspecter et valider", 
            options=range(12), 
            format_func=lambda x: f"Puits {x+1} : {CONDITIONS[x]}"
        )
        
        current_x = x_positions[selected_well]
        active_profile = inverted_img[y_start:y_end, current_x]
        profile_len = len(active_profile)

        st.markdown(f"**Analyse du profil (X = {current_x}) :**")
        
        # --- CURSEURS DE SENSIBILITÉ ---
        col_thresh, col_width = st.columns(2)
        with col_thresh:
            threshold = st.slider("Sensibilité (Intensité min)", 0, 255, 30)
        with col_width:
            min_width = st.slider("Largeur min du pic", 1, 100, 10)

        # --- DÉTECTION ET GRAPHIQUE ---
        peaks, properties = find_peaks(active_profile, height=threshold, width=min_width)
        num_peaks = len(peaks)

        fig, ax = plt.subplots(figsize=(5, 6))
        ax.plot(active_profile, range(profile_len), color='black', label="Profil d'intensité")
        
        peaks_data_for_this_well = []
        
        for i in range(num_peaks):
            peak_relative_y = peaks[i]
            peak_left = int(properties["left_bases"][i])
            peak_right = int(properties["right_bases"][i])
            
            # Colorier le pic
            ax.fill_betweenx(range(peak_left, peak_right), 0, active_profile[peak_left:peak_right], color='blue', alpha=0.3)
            ax.scatter(active_profile[peak_relative_y], peak_relative_y, color='blue', marker='o', s=50, edgecolors='white', linewidths=1)
            
            # Calcul de l'AUC pour ce pic
            band_profile = active_profile[peak_left:peak_right]
            local_background = np.min(band_profile) if len(band_profile) > 0 else 0
            net_profile = band_profile - local_background
            area = np.trapezoid(net_profile) if len(net_profile) > 0 else 0
            
            peaks_data_for_this_well.append({
                "Puits N°": selected_well + 1,
                "Condition": CONDITIONS[selected_well],
                "X Absolu": current_x,
                "Y Début Bande": y_start + peak_left,
                "Y Fin Bande": y_start + peak_right,
                "Intensité (AUC)": round(area, 2),
                "Bruit de fond": round(local_background, 2)
            })

        ax.set_ylim(profile_len, 0) 
        max_intensity = np.max(active_profile) if np.max(active_profile) > 0 else 255
        ax.set_xlim(0, max_intensity * 1.1)
        ax.set_ylabel("Position Relative (Pixels)")
        ax.set_xlabel("Intensité")
        
        handles, labels = ax.get_legend_handles_labels()
        if num_peaks > 0:
            h, = ax.plot([], [], color='blue', marker='o', linestyle='None')
            handles.append(h)
            labels.append(f"Pics détectés ({num_peaks})")
        ax.legend(handles, labels)
        
        st.pyplot(fig)

        # --- ENREGISTREMENT ---
        st.markdown("---")
        protein_name = st.text_input("Nom de la protéine ciblée (ex: p-PLCG1)", key=f"prot_{selected_well}")
        
        if st.button(f"➕ Valider et ajouter le puits {selected_well + 1} au tableau", type="primary"):
            if num_peaks > 0:
                new_data_list = []
                for idx, peak_data in enumerate(peaks_data_for_this_well):
                    peak_data_named = peak_data.copy()
                    # On nomme le pic si plusieurs sont détectés
                    peak_suffix = f" (Pic {idx+1})" if num_peaks > 1 else ""
                    peak_data_named["Nom du Pic"] = f"{protein_name}{peak_suffix}" if protein_name else f"Inconnu{peak_suffix}"
                    new_data_list.append(peak_data_named)
                
                new_data_df = pd.DataFrame(new_data_list)
                st.session_state.results_df = pd.concat([st.session_state.results_df, new_data_df], ignore_index=True)
                st.success(f"Puits {selected_well + 1} enregistré ! Passez au suivant.")
            else:
                st.warning("Aucun pic détecté à enregistrer.")

# --- AFFICHAGE ET EXPORT DU TABLEAU GLOBAL ---
if not st.session_state.results_df.empty:
    st.markdown("---")
    st.subheader("📊 Tableau de Synthèse")
    st.dataframe(st.session_state.results_df, use_container_width=True)
    
    csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger la synthèse (CSV)",
        data=csv,
        file_name="Quantification_Complete.csv",
        mime="text/csv",
    )
    
    if st.button("🗑️ Vider le tableau"):
        st.session_state.results_df = st.session_state.results_df.iloc[0:0]
        st.rerun()
