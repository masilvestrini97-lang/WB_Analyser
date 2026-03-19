import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from streamlit_drawable_canvas import st_canvas

# --- Initialisation de la mémoire ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=[
        "Condition", "Bande", "Surface (Pixels²)", "Intensité brute", "Bruit de fond", "Intensité Nette"
    ])

st.set_page_config(page_title="Quantificateur WB V11 (Zoom & Toile)", layout="wide")
st.title("🔬 Quantificateur 2D : Détection + Édition Manuelle")
st.markdown("Laissez l'algorithme proposer des boîtes (à gauche), puis **ajustez-les confortablement sur l'image agrandie** (à droite).")

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
    lane_rgb = cv2.cvtColor(lane_gray, cv2.COLOR_GRAY2RGB)
    
    canvas_height = int(lane_rgb.shape[0])
    canvas_width = int(lane_rgb.shape[1])

    st.markdown("---")
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("1. Auto-Détection")
        threshold_val = st.slider("Sensibilité de pré-détection", 0, 255, 50)
        min_area = st.slider("Surface minimale", 10, 1000, 50)
        
        # --- NOUVEAU : LE ZOOM ---
        st.markdown("---")
        zoom_factor = st.slider("🔍 Zoom d'affichage (pour dessiner plus facilement)", 1, 5, 3)
        
        if st.button("🔄 Lancer l'Auto-Détection", type="primary"):
            _, binary_mask = cv2.threshold(lane_inverted, threshold_val, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            canvas_objects = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h >= min_area:
                    # On multiplie les coordonnées par le Zoom pour l'affichage
                    canvas_objects.append({
                        "type": "rect",
                        "left": float(x * zoom_factor), 
                        "top": float(y * zoom_factor), 
                        "width": float(w * zoom_factor), 
                        "height": float(h * zoom_factor),
                        "fill": "rgba(0, 0, 0, 0)",
                        "stroke": "red",
                        "strokeWidth": 2,
                        "transparentCorners": False
                    })
            
            st.session_state["initial_drawing"] = {
                "version": "4.4.0",
                "objects": canvas_objects
            }
            st.rerun()

    with col2:
        st.subheader("2. La Toile Interactive")
        drawing_mode = st.radio("Outil actif :", ("transform", "rect"), horizontal=True, 
                                format_func=lambda x: "🖱️ Modifier/Sélectionner" if x == "transform" else "✏️ Dessiner Nouvelle Boîte")

        initial_state = st.session_state.get("initial_drawing", {"version": "4.4.0", "objects": []})

        # Application du Zoom sur l'image de fond (INTER_NEAREST pour ne pas flouter les bandes)
        lane_rgb_zoomed = cv2.resize(lane_rgb, (canvas_width * zoom_factor, canvas_height * zoom_factor), interpolation=cv2.INTER_NEAREST)
        lane_pil_zoomed = Image.fromarray(lane_rgb_zoomed)

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=2,
            stroke_color="red",
            background_image=lane_pil_zoomed,
            update_streamlit=True,
            height=canvas_height * zoom_factor, 
            width=canvas_width * zoom_factor,
            drawing_mode=drawing_mode,
            initial_drawing=initial_state,
            key="canvas",
        )

        st.markdown("---")
        condition_name = st.text_input("Nom de la piste (ex: WT 7min)")
        
        if st.button("✅ Calculer et Enregistrer ces Boîtes"):
            if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                detected_bands_data = []
                band_counter = 1
                
                objects = canvas_result.json_data["objects"]
                objects = sorted(objects, key=lambda obj: obj["top"])
                
                for obj in objects:
                    if obj["type"] == "rect":
                        # On DIVISE par le Zoom pour revenir aux vraies dimensions de l'image
                        x = int(obj["left"] / zoom_factor)
                        y = int(obj["top"] / zoom_factor)
                        w = int((obj["width"] * obj.get("scaleX", 1)) / zoom_factor)
                        h = int((obj["height"] * obj.get("scaleY", 1)) / zoom_factor)
                        
                        area = float(w * h)
                        
                        # Sécurité pour éviter les bugs si on dessine à l'envers ou hors cadre
                        if area > 0 and w > 0 and h > 0:
                            band_pixels = lane_inverted[max(0, y):max(0, y)+h, max(0, x):max(0, x)+w]
                            raw_intden = np.sum(band_pixels, dtype=np.float64)
                            
                            local_bg = float(np.min(band_pixels)) if band_pixels.size > 0 else 0.0
                            total_bg = local_bg * area
                            net_intden = raw_intden - total_bg
                            
                            detected_bands_data.append({
                                "Condition": condition_name if condition_name else "Inconnue",
                                "Bande": f"Bande {band_counter}",
                                "Surface (Pixels²)": round(area, 2),
                                "Intensité brute": round(raw_intden, 2),
                                "Bruit de fond": round(total_bg, 2),
                                "Intensité Nette": round(net_intden, 2)
                            })
                            band_counter += 1
                
                new_data_df = pd.DataFrame(detected_bands_data)
                st.session_state.results_df = pd.concat([st.session_state.results_df, new_data_df], ignore_index=True)
                st.success(f"{band_counter-1} bandes calculées et enregistrées !")
            else:
                st.warning("Aucune boîte n'est présente sur l'image.")

# --- AFFICHAGE ET EXPORT ---
if not st.session_state.results_df.empty:
    st.markdown("---")
    st.subheader("📊 Tableau de Synthèse")
    st.dataframe(st.session_state.results_df, use_container_width=True)
    
    csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger (CSV)", data=csv, file_name="Quantification_2D_Zoom.csv", mime="text/csv")
    
    if st.button("🗑️ Vider le tableau"):
        st.session_state.results_df = st.session_state.results_df.iloc[0:0]
        st.rerun()
