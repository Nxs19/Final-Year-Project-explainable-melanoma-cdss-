import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Define custom objects needed to deserialize the model
focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0, alpha=0.25)
MODEL_PATH = 'best_melanoma_model.keras'
IMAGE_PATH = 'data/val/mel/' + os.listdir('data/val/mel')[0] # Grabs the first validation melanoma
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

print(f"[INFO] Loading fine-tuned model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'focal_loss': focal_loss})

# Access the nested MobileNetV2 base model and its last conv layer
base_model = model.get_layer('mobilenetv2_1.00_224')
LAST_CONV_LAYER_NAME = 'out_relu'

def get_gradcam_heatmap(img_array, model, base_model, last_conv_layer_name):
    # Construct a sub-model that outputs the feature maps from the base model
    conv_output = base_model.get_layer(last_conv_layer_name).output
    grad_base_model = tf.keras.Model(inputs=base_model.inputs, outputs=conv_output)
    
    with tf.GradientTape() as tape:
        # Pass input through the base model feature extractor
        feature_maps = grad_base_model(img_array)
        tape.watch(feature_maps)
        
        # Pass feature maps through the remaining top layers of the full model
        x = model.get_layer('global_avg_pool')(feature_maps)
        x = model.get_layer('batch_normalization')(x)
        x = model.get_layer('dropout')(x)
        x = model.get_layer('dense')(x)
        x = model.get_layer('dropout_1')(x)
        preds = model.get_layer('predictions')(x)
        
        pred_index = tf.argmax(preds[0])
        target_class_channel = preds[:, pred_index]

    grads = tape.gradient(target_class_channel, feature_maps)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    heatmap = feature_maps[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy(), pred_index.numpy(), preds[0].numpy()

# Preprocess image
raw_img = cv2.imread(IMAGE_PATH)
raw_img = cv2.resize(raw_img, (224, 224))
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(raw_img.copy(), axis=0))

heatmap, pred_idx, all_preds = get_gradcam_heatmap(img_array, model, base_model, LAST_CONV_LAYER_NAME)

# Superimpose heatmap
heatmap_resized = cv2.resize(heatmap, (224, 224))
jet_heatmap = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
superimposed = cv2.addWeighted(jet_heatmap, 0.4, raw_img, 0.6, 0)

# Display result
pred_label = CLASS_NAMES[pred_idx]
pred_conf = all_preds[pred_idx] * 100

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title(f"Original Input ({IMAGE_PATH.split('/')[-2]})")
plt.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Prediction: {pred_label.upper()} ({pred_conf:.1f}%)")
plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.tight_layout()
plt.savefig('production_gradcam_result.png', dpi=300)
plt.show()
print(f"[SUCCESS] Prediction: {pred_label} ({pred_conf:.2f}%). Output saved as 'production_gradcam_result.png'.")