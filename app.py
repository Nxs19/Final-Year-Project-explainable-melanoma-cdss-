import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# Page configuration for a professional medical interface
st.set_page_config(
    page_title="DermaExplain CDSS",
    page_icon="🔬",
    layout="wide"
)

# Diagnostic mapping for all 7 HAM10000 classes
DIAGNOSTIC_MAP = {
    'akiec': {'name': 'Actinic Keratosis', 'status': 'Pre-malignant', 'color': 'orange'},
    'bcc':   {'name': 'Basal Cell Carcinoma', 'status': 'Malignant', 'color': 'red'},
    'bkl':   {'name': 'Benign Keratosis', 'status': 'Benign', 'color': 'green'},
    'df':    {'name': 'Dermatofibroma', 'status': 'Benign', 'color': 'green'},
    'mel':   {'name': 'Melanoma', 'status': 'Malignant', 'color': 'red'},
    'nv':    {'name': 'Melanocytic Nevus', 'status': 'Benign', 'color': 'green'},
    'vasc':  {'name': 'Vascular Lesion', 'status': 'Benign', 'color': 'green'}
}
CLASS_KEYS = list(DIAGNOSTIC_MAP.keys())

def apply_dullrazor(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(img_bgr, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted

# --- MODEL INGESTION (CACHED) ---
@st.cache_resource
def load_model_pipeline():
    focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0, alpha=0.25)
    model = tf.keras.models.load_model('best_melanoma_model.keras', custom_objects={'focal_loss': focal_loss})
    base_model = model.get_layer('mobilenetv2_1.00_224')
    return model, base_model

model, base_model = load_model_pipeline()

# --- EXPLAINABLE AI ENGINE ---
def compute_gradcam(image_tensor, model, base_model, last_conv_layer='out_relu'):
    conv_output = base_model.get_layer(last_conv_layer).output
    grad_base_model = tf.keras.Model(inputs=base_model.inputs, outputs=conv_output)
    
    with tf.GradientTape() as tape:
        feature_maps = grad_base_model(image_tensor)
        tape.watch(feature_maps)
        
        # Forward pass through classification head
        x = model.get_layer('global_avg_pool')(feature_maps)
        x = model.get_layer('batch_normalization')(x)
        x = model.get_layer('dropout')(x)
        x = model.get_layer('dense')(x)
        x = model.get_layer('dropout_1')(x)
        preds = model.get_layer('predictions')(x)
        
        pred_idx = tf.argmax(preds[0])
        target_channel = preds[:, pred_idx]
        
    grads = tape.gradient(target_channel, feature_maps)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    heatmap = feature_maps[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_idx.numpy(), preds[0].numpy()

# --- UI HEADER & SIDEBAR ---
st.title("🔬 Explainable AI Clinical Decision Support System (CDSS)")
st.caption("Dermatological Pathology Classification with Gradient-weighted Class Activation Mapping (Grad-CAM)")

st.sidebar.header("Inference Controls")
alpha_val = st.sidebar.slider("Heatmap Transparency (Alpha)", min_value=0.1, max_value=0.9, value=0.4, step=0.05)
confidence_threshold = st.sidebar.slider("Clinical Certainty Flag (%)", min_value=50, max_value=90, value=70, step=5)

uploaded_file = st.sidebar.file_uploader("Upload Dermoscopy Image (.jpg, .jpeg, .png)", type=['jpg', 'jpeg', 'png'])
enable_hair_removal = st.sidebar.checkbox("Enable DullRazor Preprocessing", value=True)

# --- MAIN INFERENCE WORKFLOW ---
if uploaded_file is not None:
    # Read and standardize the uploaded image
    pil_img = Image.open(uploaded_file).convert('RGB')
    raw_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    if enable_hair_removal:
        processed_cv2 = apply_dullrazor(raw_cv2)
    else:
        processed_cv2 = raw_cv2
        
    resized_img = cv2.resize(processed_cv2, (224, 224))
    
    # Preprocess tensor for MobileNetV2
    tensor_input = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.expand_dims(resized_img.copy(), axis=0)
    )
    
    # Run prediction and XAI extraction
    with st.spinner("Executing neural inference and computing gradient tape activations..."):
        heatmap, pred_idx, all_probs = compute_gradcam(tensor_input, model, base_model)
    
    # Construct visual heatmap overlay
    heatmap_scaled = cv2.resize(heatmap, (224, 224))
    jet_map = cv2.applyColorMap(np.uint8(255 * heatmap_scaled), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(jet_map, alpha_val, resized_img, 1.0 - alpha_val, 0)
    
    rgb_original = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    rgb_superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    
    predicted_key = CLASS_KEYS[pred_idx]
    diagnostic_info = DIAGNOSTIC_MAP[predicted_key]
    confidence_pct = all_probs[pred_idx] * 100
    
    # Clinical triage warning banner
    if diagnostic_info['status'] == 'Malignant':
        st.error(f"⚠️ **URGENT PATHOLOGY DETECTED: {diagnostic_info['name'].upper()} ({confidence_pct:.1f}% Confidence)**\n\nImmediate specialist evaluation and histological biopsy recommended.")
    elif diagnostic_info['status'] == 'Pre-malignant':
        st.warning(f"⚡ **PRE-CANCEROUS LESION: {diagnostic_info['name'].upper()} ({confidence_pct:.1f}% Confidence)**\n\nMonitor closely for progression.")
    else:
        st.success(f"✅ **BENIGN MORPHOLOGY: {diagnostic_info['name'].upper()} ({confidence_pct:.1f}% Confidence)**")
        
    if confidence_pct < confidence_threshold:
        st.info(f"ℹ️ **Low Diagnostic Certainty (< {confidence_threshold}%):** Verify Grad-CAM localization for potential artifact bias (e.g., hair, ruler markings).")

    # Image panels
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Patient Lesion (Standardized 224x224)")
        st.image(rgb_original, use_container_width=True)
        
    with col2:
        st.subheader(f"Grad-CAM Visual Audit (Alpha = {alpha_val:.2f})")
        st.image(rgb_superimposed, use_container_width=True)
        st.caption("Hotspots (Red/Orange) indicate pixel regions driving the diagnostic probability.")

    # Probability breakdown
    st.subheader("Diagnostic Probability Distribution Across Pathologies")
    prob_df = pd.DataFrame({
        'Pathology': [DIAGNOSTIC_MAP[k]['name'] for k in CLASS_KEYS],
        'Classification': [DIAGNOSTIC_MAP[k]['status'] for k in CLASS_KEYS],
        'Probability (%)': [p * 100 for p in all_probs]
    }).sort_values(by='Probability (%)', ascending=False)
    
    st.dataframe(
        prob_df.style.format({'Probability (%)': '{:.2f}%'}),
        use_container_width=True,
        hide_index=True
    )
    st.bar_chart(data=prob_df.set_index('Pathology')['Probability (%)'])

else:
    st.info("👈 Upload a dermoscopy image in the sidebar to run clinical inference and generate the Explainable AI activation map.")