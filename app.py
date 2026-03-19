import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# --- Initialisation de la mémoire ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=[
        "Condition", "Bande", "Surface (Pixels²)", "Intensité brute", "Bruit de fond", "Intensité Nette"
    ])
    
# Nouvelle mémoire pour stocker nos boîtes 2D manuellement
if 'saved_boxes' not in st.session_state:
    st.session_state.saved_boxes = [] # Liste de dicts: {'y': int, 'h': int}

st.set_page_config(page_title="Quantificateur WB V12 (Éditeur Natif)", layout="wide")
st.title("🔬 Quantificateur 2D : Éditeur de Boîtes Natif")
st.markdown("Fini les bugs de plugin ! Utilisez l'**Auto-détection**, puis ajustez ou ajoutez des boîtes avec le **Viseur Vert**.")

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

    # --- ÉTAPE 1 : DÉFINITION DE LA PISTE ---
    st.markdown("---")
    col_setup1, col_setup2 = st.columns(2)
    with col_setup1:
        x_center = st.slider("Position X (Centre de la piste)", 0, width, int(width/2))
        lane_width = st.slider("Largeur de la piste", 10, 200, 60)
    with col_setup2:
        y_range = st.slider("Plage Verticale", 0, height, (int(height*0.1), int(height*0.9)))
        y_start, y_end = y_range

    if lane_width < 2:
        st.stop()

    x_start = max(0, x_center - lane_width // 2)
    x_end = min(width, x_center + lane_width // 2)

    # Extraction
    lane_gray = gray[y_start:y_end, x_start:x_end]
    lane_inverted = inverted_img[y_start:y_end, x_start:x_end]
    lane_height, lane_width_actual = lane_gray.shape

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1.5, 1])

    with col1:
        st.subheader("1. Auto-Détection")
        threshold_val = st.slider("Sensibilité (Seuil)", 0, 255, 50)
        min_area = st.slider("Surface minimale", 10, 1000, 50)
        
        if st.button("🔄 Lancer l'Auto-Détection", type="primary"):
            _, binary_mask = cv2.threshold(lane_inverted, threshold_val, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            new_boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h >= min_area:
                    # On stocke uniquement Y et H, le X et W sont définis par la largeur de la piste
                    new_boxes.append({'y': y, 'h': h})
            
            st.session_state.saved_boxes = sorted(new_boxes, key=lambda b: b['y'])
            st.rerun()
            
        if st.button("🗑️ Effacer toutes les boîtes"):
            st.session_state.saved_boxes = []
            st.rerun()

    with col3:
        st.subheader("2. Le Viseur (Ajout manuel)")
        st.markdown("Ajustez la boîte **verte** pour encadrer une bande manquante.")
        
        # Le Viseur Vert
        target_y = st.slider("Position Y (Haut-Bas)", 0, lane_height, int(lane_height/2))
        target_h = st.slider("Épaisseur de la bande (Hauteur)", 2, 100, 20)
        
        if st.button("✅ Ajouter cette boîte verte"):
            st.session_state.saved_boxes.append({'y': target_y, 'h': target_h})
            # Trier les boîtes de haut en bas
            st.session_state.saved_boxes = sorted(st.session_state.saved_boxes, key=lambda b: b['y'])
            st.rerun()
            
        st.markdown("---")
        st.markdown("**Correction / Suppression :**")
        if len(st.session_state.saved_boxes) > 0:
            box_to_delete = st.selectbox("Sélectionner une boîte à supprimer", range(len(st.session_state.saved_boxes)), format_func=lambda i: f"Boîte {i+1} (Y: {st.session_state.saved_boxes[i]['y']})")
            if st.button("❌ Supprimer cette boîte"):
                st.session_state.saved_boxes.pop(box_to_delete)
                st.rerun()

    with col2:
        st.subheader("Aperçu en Direct")
        
        # Création de l'image d'affichage
        display_img = cv2.cvtColor(lane_gray, cv2.COLOR_GRAY2RGB)
        
        # 1. Dessiner les boîtes sauvegardées (ROUGES)
        for i, box in enumerate(st.session_state.saved_boxes):
            cv2.rectangle(display_img, (0, box['y']), (lane_width_actual, box['y'] + box['h']), (0, 0, 255), 2)
            cv2.putText(display_img, f"B{i+1}", (2, max(0, box['y']-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        # 2. Dessiner le Viseur (VERT)
        cv2.rectangle(display_img, (0, target_y), (lane_width_actual, target_y + target_h), (0, 255, 0), 2)
        cv2.putText(display_img, "VISEUR", (2, min(lane_height-5, target_y + target_h + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Zoom pour un meilleur confort visuel (purement cosmétique pour l'écran)
        zoom_factor = 3
        display_img_zoomed = cv2.resize(display_img, (lane_width_actual * zoom_factor, lane_height * zoom_factor), interpolation=cv2.INTER_NEAREST)
        
        st.image(display_img_zoomed, caption="Rouge = Sauvegardé | Vert = Viseur actif", use_container_width=True)

    # --- CALCUL ET ENREGISTREMENT ---
    st.markdown("---")
    st.subheader("3. Enregistrer les résultats")
    condition_name = st.text_input("Nom de la piste (ex: WT 7min)")
    
    if st.button("💾 Calculer et Enregistrer toutes les boîtes ROUGES", type="primary"):
        if len(st.session_state.saved_boxes) > 0:
            detected_bands_data = []
            
            for i, box in enumerate(st.session_state.saved_boxes):
                y = box['y']
                h = box['h']
                w = lane_width_actual
                area = float(w * h)
                
                # Extraction sur l'image inversée pour le calcul
                band_pixels = lane_inverted[y:y+h, 0:w]
                
                # Calcul 2D sécurisé
                raw_intden = np.sum(band_pixels, dtype=np.float64)
                local_bg = float(np.min(band_pixels)) if band_pixels.size > 0 else 0.0
                total_bg = local_bg * area
                net_intden = raw_intden - total_bg
                
                detected_bands_data.append({
                    "Condition": condition_name if condition_name else "Inconnue",
                    "Bande": f"Bande {i+1}",
                    "Surface (Pixels²)": round(area, 2),
                    "Intensité brute": round(raw_intden, 2),
                    "Bruit de fond": round(total_bg, 2),
                    "Intensité Nette": round(net_intden, 2)
                })
            
            new_data_df = pd.DataFrame(detected_bands_data)
            st.session_state.results_df = pd.concat([st.session_state.results_df, new_data_df], ignore_index=True)
            st.success(f"{len(detected_bands_data)} bandes enregistrées !")
        else:
            st.warning("Aucune boîte rouge à calculer.")

# --- AFFICHAGE ET EXPORT ---
if not st.session_state.results_df.empty:
    st.markdown("---")
    st.subheader("📊 Tableau de Synthèse")
    st.dataframe(st.session_state.results_df, use_container_width=True)
    
    csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger (CSV)", data=csv, file_name="Quantification_2D_Native.csv", mime="text/csv")
    
    if st.button("🗑️ Vider le tableau"):
        st.session_state.results_df = st.session_state.results_df.iloc[0:0]
        st.rerun()
