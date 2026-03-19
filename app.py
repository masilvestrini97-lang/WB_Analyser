import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# --- Initialisation de la mémoire ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=[
        "Condition", "Bande", "Surface (Pixels²)", "Intensité brute (IntDen)", "Bruit de fond estimé", "Intensité Nette"
    ])

st.set_page_config(page_title="Quantificateur WB V9 (Détourage 2D)", layout="wide")
st.title("🔬 Quantificateur 2D par Détourage Automatique (ROIs)")
st.markdown("Encadrez la piste, puis réglez la détection pour que le logiciel **dessine des boîtes autour de vos bandes** et calcule leur volume total.")

uploaded_file = st.file_uploader("Choisissez une image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Inversion : le noir devient du signal positif (0-255)
    inverted_img = 255 - gray
    height, width = inverted_img.shape

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Définir la Piste (Colonne)")
        
        # On définit maintenant un rectangle (et plus une ligne)
        x_center = st.slider("Position X (Centre de la piste)", 0, width, int(width/2))
        lane_width = st.slider("Largeur de la piste", 10, 200, 60, help="Doit englober toute la largeur de vos bandes.")
        
        y_range = st.slider("Plage Verticale", 0, height, (int(height*0.1), int(height*0.9)))
        y_start, y_end = y_range
        
        x_start = max(0, x_center - lane_width // 2)
        x_end = min(width, x_center + lane_width // 2)

        # Extraction de la zone d'intérêt (La Piste)
        lane_roi = inverted_img[y_start:y_end, x_start:x_end]
        
        # Affichage du cadre global
        img_display_global = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(img_display_global, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        st.image(img_display_global, caption="Piste sélectionnée", use_container_width=True)

    with col2:
        st.subheader("2. Détourage Semi-Automatique des Bandes")
        
        # --- RÉGLAGES DE DÉTECTION 2D ---
        st.markdown("Ajustez ces curseurs pour encadrer parfaitement les bandes.")
        threshold_val = st.slider("Seuil de détection (Sensibilité)", 0, 255, 50, help="Plus c'est haut, plus seules les bandes très noires seront détourées.")
        min_area = st.slider("Surface minimale (exclure les poussières)", 10, 1000, 100)

        # 1. Binarisation de l'image (création d'un masque)
        _, binary_mask = cv2.threshold(lane_roi, threshold_val, 255, cv2.THRESH_BINARY)
        
        # 2. Recherche des contours sur le masque
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Image pour visualiser les boîtes (uniquement la piste découpée)
        roi_display = cv2.cvtColor(255 - lane_roi, cv2.COLOR_GRAY2RGB) # Sur fond blanc original pour y voir clair
        
        detected_bands_data = []
        
        # 3. Filtrage et dessin des boîtes
        band_counter = 1
        # Trier les contours de haut en bas (selon Y)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if area >= min_area:
                # Dessin de la boîte 2D englobante (Bounding Box)
                cv2.rectangle(roi_display, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(roi_display, f"B{band_counter}", (x, max(0, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # --- CALCUL 2D (Integrated Density) ---
                # Extraction des pixels *exactement* dans cette boîte
                band_pixels = lane_roi[y:y+h, x:x+w]
                
                # Somme totale du signal (Volume Brut)
                raw_intden = np.sum(band_pixels)
                
                # Soustraction basique du bruit de fond local (valeur min de la boîte * surface)
                local_bg = np.min(band_pixels)
                total_bg = local_bg * area
                net_intden = raw_intden - total_bg
                
                detected_bands_data.append({
                    "Bande": f"Bande {band_counter}",
                    "Surface (Pixels²)": area,
                    "Intensité brute (IntDen)": raw_intden,
                    "Bruit de fond estimé": total_bg,
                    "Intensité Nette": net_intden
                })
                band_counter += 1

        # Affichage du résultat de la détection (Masque + Boîtes)
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            st.image(binary_mask, caption="Masque de détection (Vue Machine)", use_container_width=True)
        with subcol2:
            st.image(roi_display, caption=f"{len(detected_bands_data)} Bandes détourées", use_container_width=True)

        # --- ENREGISTREMENT ---
        st.markdown("---")
        condition_name = st.text_input("Nom de la piste (ex: WT 7min)")
        
        if st.button("➕ Enregistrer ces bandes au tableau", type="primary"):
            if len(detected_bands_data) > 0 and condition_name:
                # Ajout du nom de la condition
                for b_data in detected_bands_data:
                    b_data["Condition"] = condition_name
                
                new_data_df = pd.DataFrame(detected_bands_data)
                st.session_state.results_df = pd.concat([st.session_state.results_df, new_data_df], ignore_index=True)
                st.success("Bandes enregistrées !")
            else:
                st.warning("Veuillez entrer un nom de condition ou ajuster le seuil pour détecter des bandes.")

# --- AFFICHAGE ET EXPORT DU TABLEAU ---
if not st.session_state.results_df.empty:
    st.markdown("---")
    st.subheader("📊 Tableau de Synthèse (Densitométrie 2D)")
    st.dataframe(st.session_state.results_df, use_container_width=True)
    
    csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger la quantification 2D (CSV)",
        data=csv,
        file_name="Quantification_2D_ROIs.csv",
        mime="text/csv",
    )
    
    if st.button("🗑️ Vider le tableau"):
        st.session_state.results_df = st.session_state.results_df.iloc[0:0]
        st.rerun()
