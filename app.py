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

st.set_page_config(page_title="Quantificateur WB V10 (Toile Interactive)", layout="wide")
st.title("🔬 Quantificateur 2D : Détection + Édition Manuelle")
st.markdown("Laissez l'algorithme proposer des boîtes (à gauche), puis **ajustez-les, supprimez-les ou dessinez-en de nouvelles** directement avec la souris (à droite).")

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

    # Sécurité pour éviter les plantages si la largeur est à 0
    if lane_width < 2:
        st.warning("Veuillez augmenter la largeur de la piste.")
        st.stop()

    x_start = max(0, x_center - lane_width // 2)
    x_end = min(width, x_center + lane_width // 2)

    # Extraction de l'image de la piste
    lane_gray = gray[y_start:y_end, x_start:x_end]
    lane_inverted = inverted_img[y_start:y_end, x_start:x_end]
    
    # On passe la piste en RGB pour le Canvas
    lane_rgb = cv2.cvtColor(lane_gray, cv2.COLOR_GRAY2RGB)
    lane_pil = Image.fromarray(lane_rgb)

    # CORRECTION CRUCIALE : Forcer les dimensions en entiers purs de Python
    canvas_height = int(lane_pil.height)
    canvas_width = int(lane_pil.width)

    st.markdown("---")
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("1. Auto-Détection (Le Brouillon)")
        threshold_val = st.slider("Sensibilité de pré-détection", 0, 255, 50)
        min_area = st.slider("Surface minimale", 10, 1000, 50)
        
        if st.button("🔄 Lancer l'Auto-Détection", type="primary"):
            _, binary_mask = cv2.threshold(lane_inverted, threshold_val, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            canvas_objects = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h >= min_area:
                    canvas_objects.append({
                        "type": "rect",
                        "left": float(x), "top": float(y), "width": float(w), "height": float(h),
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
        st.subheader("2. La Toile Interactive (Correction)")
        st.markdown("Utilisez **Modifier/Sélectionner** pour ajuster les boîtes rouges. Utilisez **Dessiner** pour en créer de nouvelles.")
        
        drawing_mode = st.radio("Outil actif :", ("transform", "rect"), horizontal=True, 
                                format_func=lambda x: "🖱️ Modifier/Sélectionner" if x == "transform" else "✏️ Dessiner Nouvelle Boîte")

        initial_state = st.session_state.get("initial_drawing", {"version": "4.4.0", "objects": []})

        # --- LE CANVAS INTERACTIF ---
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=2,
            stroke_color="red",
            background_image=lane_pil,
            update_streamlit=True,
            height=canvas_height, # Utilisation des dimensions sécurisées
            width=canvas_width,   # Utilisation des dimensions sécurisées
            drawing_mode=drawing_mode,
            initial_drawing=initial_state,
            key="canvas",
        )

        # --- CALCUL ET ENREGISTREMENT ---
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
                        x = int(obj["left"])
                        y = int(obj["top"])
                        w = int(obj["width"] * obj.get("scaleX", 1))
                        h = int(obj["height"] * obj.get("scaleY", 1))
                        
                        area = float(w * h)
                        
                        if area > 0:
                            band_pixels = lane_inverted[y:y+h, x:x+w]
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
    st.subheader("📊 Tableau de Synthèse (Densitométrie 2D)")
    st.dataframe(st.session_state.results_df, use_container_width=True)
    
    csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger (CSV)", data=csv, file_name="Quantification_2D_Interactive.csv", mime="text/csv")
    
    if st.button("🗑️ Vider le tableau"):
        st.session_state.results_df = st.session_state.results_df.iloc[0:0]
        st.rerun()
