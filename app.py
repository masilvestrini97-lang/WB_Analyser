import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import io

# --- Liste standardisée de vos conditions ---
CONDITIONS = [
    "EV -", "EV 2", "EV 7", "EV 15",
    "WT -", "WT 2", "WT 7", "WT 15",
    "D776N -", "D776N 2", "D776N 7", "D776N 15"
]

st.set_page_config(page_title="Quantificateur WB V7 Flexible", layout="wide")
st.title("🔬 Quantificateur Haut Débit Flexible (12 Puits)")
st.markdown("Ciblez la grille, puis ajustez **chaque piste individuellement** si nécessaire pour corriger la migration.")

# --- Initialisation de la mémoire (Session State) pour les ajustements individuels ---
if 'lane_adjustments' not in st.session_state:
    # On stocke les décalages (offsets) X, Y-top et Y-bottom pour chaque puits
    st.session_state.lane_adjustments = {i: {"dx": 0, "dy_top": 0, "dy_bottom": 0} for i in range(12)}

# Fonction pour réinitialiser les ajustements
def reset_adjustments():
    st.session_state.lane_adjustments = {i: {"dx": 0, "dy_top": 0, "dy_bottom": 0} for i in range(12)}

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
        st.subheader("1. Placer et Ajuster la Grille")
        
        tab1, tab2 = st.tabs(["🏗️ Placement Global", "🔧 Ajustement Individuel"])
        
        with tab1:
            st.markdown("**Étape A :** Définir le contour de la grille et la zone verticale par défaut.")
            x_first = st.slider("Axe X : Première piste (EV -)", 0, width, int(width*0.1))
            x_last = st.slider("Axe X : Dernière piste (D776N 15)", 0, width, int(width*0.9))
            
            global_y_range = st.slider(
                "Axe Y : Encadrer la protéine (Zone par défaut)", 
                0, height, (int(height*0.4), int(height*0.5)),
                help="Ajustez pour le 'gros' de la rangée de bandes."
            )
            g_y_start, g_y_end = global_y_range

            if st.button("🗑️ Réinitialiser tous les ajustements individuels"):
                reset_adjustments()
                st.rerun()

        with tab2:
            st.markdown("**Étape B :** Si une piste a bougé (X) ou a migré plus haut/bas (Y), corrigez-la ici.")
            
            # Sélection du puits à corriger
            well_to_adjust = st.selectbox(
                "Sélectionner la piste à corriger", 
                options=range(1, 13), 
                format_func=lambda i: f"Puits {i} ({CONDITIONS[i-1]})"
            )
            idx = well_to_adjust - 1
            
            # Curseurs pour ajuster cette piste spécifique (relativement à la grille par défaut)
            st.markdown(f"**Ajustement pour {CONDITIONS[idx]} :**")
            current_adj = st.session_state.lane_adjustments[idx]
            
            new_dx = st.slider("Ajustement X précis (Pix)", -50, 50, current_adj["dx"], key=f"dx_{idx}")
            new_dy_top = st.slider("Ajustement Y-Haut ( Pix)", -100, 100, current_adj["dy_top"], key=f"dyt_{idx}", help="Monter (négatif) ou descendre (positif) le haut de la ligne.")
            new_dy_bottom = st.slider("Ajustement Y-Bas (Pix)", -100, 100, current_adj["dy_bottom"], key=f"dyb_{idx}", help="Monter (négatif) ou descendre (positif) le bas de la ligne.")
            
            # Mise à jour de la mémoire
            st.session_state.lane_adjustments[idx] = {"dx": new_dx, "dy_top": new_dy_top, "dy_bottom": new_dy_bottom}

        # --- CALCUL ET DESSIN DE LA GRILLE FLEXIBLE ---
        # 1. Grille théorique de base (linspace)
        base_x_positions = np.linspace(x_first, x_last, 12, dtype=int)
        
        # 2. Application des ajustements pour créer la grille RÉELLE
        actual_lanes = []
        for i, x_base in enumerate(base_x_positions):
            adj = st.session_state.lane_adjustments[i]
            actual_x = x_base + adj["dx"]
            actual_y_start = g_y_start + adj["dy_top"]
            actual_y_end = g_y_end + adj["dy_bottom"]
            
            # Sécurité pour ne pas sortir de l'image
            actual_x = max(0, min(width, actual_x))
            actual_y_start = max(0, min(height, actual_y_start))
            actual_y_end = max(0, min(height, actual_y_end))
            
            actual_lanes.append({"x": actual_x, "y_start": actual_y_start, "y_end": actual_y_end})

        # Dessin de la grille flexible
        img_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Rectangle global par défaut (pointillé)
        cv2.rectangle(img_display, (x_first, g_y_start), (x_last, g_y_end), (100, 100, 100), 2, cv2.LINE_AA)
        
        # Dessiner les 12 segments FLEXIBLES
        for i, lane in enumerate(actual_lanes):
            # Couleur spéciale pour la piste en cours d'ajustement (Rouge) vs les autres (Bleu)
            color = (0, 0, 255) if i == idx else (255, 0, 0)
            thickness = 4 if i == idx else 2
            
            # Dessiner le segment de lecture
            cv2.line(img_display, (lane["x"], lane["y_start"]), (lane["x"], lane["y_end"]), color, thickness)
            
            # Marqueurs de début/fin
            cv2.circle(img_display, (lane["x"], lane["y_start"]), 5, color, -1)
            cv2.circle(img_display, (lane["x"], lane["y_end"]), 5, color, thickness)
            
            # Petit numéro du puits
            cv2.putText(img_display, str(i+1), (lane["x"]-15, lane["y_start"]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

        st.image(img_display, caption="Grille de lecture flexible ajustée", use_container_width=True)

    with col2:
        st.subheader("2. Résultats Instantanés")
        
        # Calcul en boucle sur la grille FLEXIBLE
        results_list = []
        for i, lane in enumerate(actual_lanes):
            # Extraction du profil de la bande pour cette piste SPÉCIFIQUE
            band_profile = inverted_img[lane["y_start"]:lane["y_end"], lane["x"]]
            
            if len(band_profile) > 0:
                local_background = np.min(band_profile)
                net_profile = band_profile - local_background
                area = np.trapezoid(net_profile)
            else:
                area, local_background = 0, 0
                
            results_list.append({
                "Puits N°": i + 1,
                "Condition": CONDITIONS[i],
                "X Absolu": lane["x"],
                "Y-Haut Absolu": lane["y_start"],
                "Y-Bas Absolu": lane["y_end"],
                "Intensité (AUC)": round(area, 2),
                "Bruit de fond": round(local_background, 2)
            })

        # Création du DataFrame
        df_results = pd.DataFrame(results_list)
        
        # Nom de la protéine pour l'export
        protein_name = st.text_input("Protéine quantifiée (ex: p-PLCG1)", value="Protéine_Inconnue")
        
        st.dataframe(df_results, use_container_width=True)
        
        # Bouton d'export CSV
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Télécharger les 12 intensités ({protein_name})",
            data=csv,
            file_name=f"Quantification_{protein_name}.csv",
            mime="text/csv",
        )
