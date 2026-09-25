import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. CONFIGURATION ---
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 7
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 15

# --- 2. ADVANCED DATA PIPELINE ---
print("[INFO] Building data generators with deep augmentation...")
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_names = list(val_gen.class_indices.keys())

# --- 3. MODEL ARCHITECTURE ---
print("[INFO] Instantiating MobileNetV2 architecture...")
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze base during Phase 1

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name="predictions")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Multi-class Focal Loss implementation via Keras native loss
focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(
    gamma=2.0,
    alpha=0.25,
    name='focal_loss'
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=focal_loss,
    metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
)

# --- 4. CALLBACKS SETUP ---
callbacks = [
    ModelCheckpoint(
        filepath='best_melanoma_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# --- 5. PHASE 1: WARMUP TRAINING ---
print(f"\n[PHASE 1] Training classification head for {INITIAL_EPOCHS} epochs...")
history_phase1 = model.fit(
    train_gen,
    epochs=INITIAL_EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)

# --- 6. PHASE 2: FINE-TUNING ---
print("\n[PHASE 2] Unfreezing top blocks of base model for fine-tuning...")
base_model.trainable = True

# Freeze all layers except the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with a significantly lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=focal_loss,
    metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
)

total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
history_phase2 = model.fit(
    train_gen,
    epochs=total_epochs,
    initial_epoch=INITIAL_EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)

# --- 7. EVALUATION AND METRIC EXPORT ---
print("\n[INFO] Evaluating best serialized model on validation set...")
best_model = tf.keras.models.load_model(
    'best_melanoma_model.keras',
    custom_objects={'focal_loss': focal_loss}
)

predictions = best_model.predict(val_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = val_gen.classes

print("\n--- Final Model Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix Export
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Final Model Confusion Matrix (Focal Loss & Fine-Tuned)')
plt.ylabel('True Pathology')
plt.xlabel('Predicted Pathology')
plt.tight_layout()
plt.savefig('final_confusion_matrix.png', dpi=300)
plt.close()

# Combined Learning Curves Export
def combine_histories(h1, h2, metric):
    return h1.history[metric] + h2.history[metric]

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(combine_histories(history_phase1, history_phase2, 'loss'), label='Train Loss')
plt.plot(combine_histories(history_phase1, history_phase2, 'val_loss'), label='Val Loss')
plt.axvline(x=INITIAL_EPOCHS - 1, color='red', linestyle='--', label='Fine-Tuning Start')
plt.title('Final Model Loss Progression')
plt.xlabel('Epoch')
plt.ylabel('Focal Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(combine_histories(history_phase1, history_phase2, 'recall'), label='Train Recall')
plt.plot(combine_histories(history_phase1, history_phase2, 'val_recall'), label='Val Recall')
plt.axvline(x=INITIAL_EPOCHS - 1, color='red', linestyle='--', label='Fine-Tuning Start')
plt.title('Final Model Diagnostic Recall Progression')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

plt.tight_layout()
plt.savefig('final_learning_curves.png', dpi=300)
plt.close()
print("[SUCCESS] Production model saved as 'best_melanoma_model.keras' alongside evaluation plots.")