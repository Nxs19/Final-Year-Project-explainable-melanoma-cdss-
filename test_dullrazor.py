import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & MODEL LOADING ---
MODEL_PATH = 'best_melanoma_model.keras'
IMAGE_PATH = 'data/val/mel/' + os.listdir('data/val/mel')[0]
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0, alpha=0.25)
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'focal_loss': focal_loss})
base_model = model.get_layer('mobilenetv2_1.00_224')

# --- 2. DULLRAZOR MORPHOLOGICAL FILTER ---
def apply_dullrazor(img_bgr):
    """
    Digitally removes hair artifacts using morphological black-hat filtering
    and Telea inpainting.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 7x7 structural element to capture elongated hair contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    
    # Black-hat transform isolates dark elements against lighter backgrounds
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Threshold to create binary mask of hair shafts
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpaint original image over masked pixels (radius=3px)
    inpainted = cv2.inpaint(img_bgr, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted, hair_mask

# --- 3. GRAD-CAM INFERENCE FUNCTION ---
def run_inference_and_cam(raw_bgr):
    raw_resized = cv2.resize(raw_bgr, (224, 224))
    tensor_input = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.expand_dims(raw_resized.copy(), axis=0)
    )
    
    conv_output = base_model.get_layer('out_relu').output
    grad_base_model = tf.keras.Model(inputs=base_model.inputs, outputs=conv_output)
    
    with tf.GradientTape() as tape:
        feature_maps = grad_base_model(tensor_input)
        tape.watch(feature_maps)
        
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
    
    heatmap_resized = cv2.resize(heatmap.numpy(), (224, 224))
    jet_map = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(jet_map, 0.4, raw_resized, 0.6, 0)
    
    pred_label = CLASS_NAMES[pred_idx.numpy()]
    confidence = preds[0][pred_idx].numpy() * 100
    
    return raw_resized, superimposed, pred_label, confidence

# --- 4. EXECUTE COMPARATIVE PIPELINE ---
print(f"[INFO] Ingesting test sample: {IMAGE_PATH}")
orig_bgr = cv2.imread(IMAGE_PATH)
cleaned_bgr, hair_mask = apply_dullrazor(orig_bgr)

# Run inference on both variants
orig_img, orig_cam, orig_label, orig_conf = run_inference_and_cam(orig_bgr)
clean_img, clean_cam, clean_label, clean_conf = run_inference_and_cam(cleaned_bgr)

# --- 5. RENDER 4-PANEL VERIFICATION FIGURE ---
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.title(f"Original Input ({IMAGE_PATH.split('/')[-2]})")
plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title(f"Raw Prediction: {orig_label.upper()} ({orig_conf:.1f}%)")
plt.imshow(cv2.cvtColor(orig_cam, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("After DullRazor (Inpainted)")
plt.imshow(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title(f"Post-Filter Prediction: {clean_label.upper()} ({clean_conf:.1f}%)")
plt.imshow(cv2.cvtColor(clean_cam, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.savefig('dullrazor_impact_comparison.png', dpi=300)
plt.show()

print(f"[RESULT] Before Filter: {orig_label} ({orig_conf:.2f}%)")
print(f"[RESULT] After Filter:  {clean_label} ({clean_conf:.2f}%)")
print("[SUCCESS] Output figure exported to 'dullrazor_impact_comparison.png'.")