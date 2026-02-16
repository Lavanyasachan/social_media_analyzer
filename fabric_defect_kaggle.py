"""
Fabric Defect Detection using ResNet50 CNN - Kaggle Optimized Script

This standalone script is optimized to run in Kaggle notebooks/scripts.
It implements a complete fabric defect detection pipeline using ResNet50 transfer learning.

KAGGLE USAGE:
1. In Kaggle, add the dataset: nexuswho/fabric-defects-dataset
2. Upload this script or copy-paste into a Kaggle notebook cell
3. Run: python fabric_defect_kaggle.py
   OR in notebook: %run fabric_defect_kaggle.py

CONFIGURATION:
- Modify the DATASET_PATH variable if using Kaggle's "Add Data" feature
- Adjust hyperparameters in the CONFIGURATION section below
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("="*80)
print("FABRIC DEFECT DETECTION USING RESNET50 CNN - KAGGLE VERSION")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print("="*80)

# ============================================================================
#                           CONFIGURATION
# ============================================================================

# Dataset Configuration
# Option 1: If you added the dataset in Kaggle, use this path:
DATASET_PATH = "/kaggle/input/fabric-defects-dataset"

# Option 2: If dataset path is different, modify here:
# DATASET_PATH = "/kaggle/input/your-dataset-path"

# Option 3: Download using kagglehub (if not already added in Kaggle)
USE_KAGGLEHUB = False  # Set to True to download via kagglehub

# Model Hyperparameters
IMG_SIZE = (224, 224)  # ResNet50 standard input size
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

# Output paths (Kaggle saves to /kaggle/working/)
OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle") else "."
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "fabric_defect_resnet_model.h5")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_fabric_defect_model.h5")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"\nConfiguration:")
print(f"  Dataset Path: {DATASET_PATH}")
print(f"  Image Size: {IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Max Epochs: {EPOCHS}")
print(f"  Output Directory: {OUTPUT_DIR}")
print("="*80)

# ============================================================================
#                         DATASET PREPARATION
# ============================================================================

print("\n[1/7] DATASET PREPARATION")
print("-"*80)

# If using kagglehub to download
if USE_KAGGLEHUB:
    try:
        import kagglehub
        print("Downloading dataset using kagglehub...")
        DATASET_PATH = kagglehub.dataset_download("nexuswho/fabric-defects-dataset")
        print(f"Dataset downloaded to: {DATASET_PATH}")
    except Exception as e:
        print(f"Error downloading with kagglehub: {e}")
        print("Please add the dataset manually in Kaggle or set correct DATASET_PATH")
        raise

# Verify dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Dataset not found at {DATASET_PATH}")
    print("\nKaggle Instructions:")
    print("1. Click 'Add Data' in the right panel")
    print("2. Search for 'fabric-defects-dataset'")
    print("3. Add the dataset by nexuswho")
    print("4. The dataset will be available at /kaggle/input/fabric-defects-dataset")
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

print(f"✓ Dataset found at: {DATASET_PATH}")

# Explore dataset structure
print("\nDataset structure:")
for root, dirs, files in os.walk(DATASET_PATH):
    level = root.replace(DATASET_PATH, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    if len(dirs) > 0:
        for d in dirs[:10]:
            print(f'{subindent}{d}/')
        if len(dirs) > 10:
            print(f'{subindent}... and {len(dirs) - 10} more directories')
    break  # Only show first level

# ============================================================================
#                         DATA GENERATORS
# ============================================================================

print("\n[2/7] DATA GENERATORS SETUP")
print("-"*80)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT
)

# Only rescaling for validation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT
)

# Create training generator
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Create validation generator
validation_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Get class information
class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)

print(f"✓ Data generators created successfully")
print(f"\nDataset Statistics:")
print(f"  Number of classes: {num_classes}")
print(f"  Class names: {class_names}")
print(f"  Training samples: {train_generator.samples}")
print(f"  Validation samples: {validation_generator.samples}")
print(f"  Steps per epoch (training): {len(train_generator)}")
print(f"  Validation steps: {len(validation_generator)}")

# ============================================================================
#                         MODEL BUILDING
# ============================================================================

print("\n[3/7] BUILDING RESNET50 MODEL")
print("-"*80)

# Load pre-trained ResNet50
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Freeze base model
base_model.trainable = False

# Build complete model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("✓ ResNet50 model built successfully!")
print(f"\nModel Architecture Summary:")
print(f"  Total layers: {len(model.layers)}")
print(f"  Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
print(f"  Non-trainable parameters: {sum([tf.size(w).numpy() for w in model.non_trainable_weights]):,}")
print(f"  Input shape: {model.input_shape}")
print(f"  Output shape: {model.output_shape}")

# ============================================================================
#                         CALLBACKS SETUP
# ============================================================================

print("\n[4/7] CONFIGURING TRAINING CALLBACKS")
print("-"*80)

callbacks = [
    ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

print("✓ Callbacks configured:")
print("  - ModelCheckpoint: Saves best model based on validation accuracy")
print("  - EarlyStopping: Stops if no improvement for 10 epochs")
print("  - ReduceLROnPlateau: Reduces learning rate when stuck")

# ============================================================================
#                         MODEL TRAINING
# ============================================================================

print("\n[5/7] TRAINING MODEL")
print("-"*80)
print(f"Training for up to {EPOCHS} epochs...")
print("(Training may stop early if model stops improving)\n")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Training completed!")

# ============================================================================
#                         MODEL EVALUATION
# ============================================================================

print("\n[6/7] EVALUATING MODEL")
print("-"*80)

# Get predictions
validation_generator.reset()
y_pred_probs = model.predict(validation_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = validation_generator.classes

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print(f"\n{'='*80}")
print(f"MODEL EVALUATION RESULTS")
print(f"{'='*80}")
print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ============================================================================
#                         VISUALIZATIONS
# ============================================================================

print("\n[7/7] GENERATING VISUALIZATIONS")
print("-"*80)

# Plot 1: Training History
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(history.history['precision'], label='Train Precision', linewidth=2)
axes[1, 0].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Recall
axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
training_plot_path = os.path.join(OUTPUT_DIR, 'training_history.png')
plt.savefig(training_plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Training history saved to: {training_plot_path}")
plt.show()

# Plot 2: Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Count'},
    square=True,
    linewidths=0.5
)
plt.title('Confusion Matrix - Fabric Defect Detection', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.tight_layout()
cm_plot_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix saved to: {cm_plot_path}")
plt.show()

# Plot 3: Per-Class Accuracy
class_accuracies = cm.diagonal() / cm.sum(axis=1)
accuracy_df = pd.DataFrame({
    'Class': class_names,
    'Accuracy': class_accuracies,
    'Percentage': class_accuracies * 100
})
accuracy_df = accuracy_df.sort_values('Accuracy', ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(accuracy_df['Class'], accuracy_df['Percentage'], color='steelblue', alpha=0.8)
plt.xlabel('Class', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Per-Class Accuracy - Fabric Defect Detection', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
per_class_plot_path = os.path.join(OUTPUT_DIR, 'per_class_accuracy.png')
plt.savefig(per_class_plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Per-class accuracy saved to: {per_class_plot_path}")
plt.show()

# ============================================================================
#                         SAVE MODEL
# ============================================================================

print("\nSaving final model...")
model.save(MODEL_SAVE_PATH)
print(f"✓ Final model saved to: {MODEL_SAVE_PATH}")

# ============================================================================
#                         FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TRAINING AND EVALUATION COMPLETE!")
print("="*80)

print(f"\n📊 RESULTS SUMMARY:")
print(f"  • Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  • Number of Classes: {num_classes}")
print(f"  • Classes: {', '.join(class_names)}")
print(f"  • Training Samples: {train_generator.samples}")
print(f"  • Validation Samples: {validation_generator.samples}")

print(f"\n💾 SAVED FILES:")
print(f"  • Final Model: {MODEL_SAVE_PATH}")
print(f"  • Best Model Checkpoint: {BEST_MODEL_PATH}")
print(f"  • Training History Plot: {training_plot_path}")
print(f"  • Confusion Matrix: {cm_plot_path}")
print(f"  • Per-Class Accuracy: {per_class_plot_path}")

print(f"\n📁 Output Location:")
if os.path.exists("/kaggle"):
    print(f"  All outputs saved to /kaggle/working/")
    print(f"  You can download them from the Kaggle output panel")
else:
    print(f"  All outputs saved to current directory")

print("\n" + "="*80)
print("✓ Script completed successfully!")
print("="*80)

# Optional: Display sample predictions
print("\n🔍 SAMPLE PREDICTIONS (first 10 validation images):")
print("-"*80)
for i in range(min(10, len(y_true))):
    true_label = class_names[y_true[i]]
    pred_label = class_names[y_pred[i]]
    confidence = y_pred_probs[i][y_pred[i]] * 100
    status = "✓" if y_true[i] == y_pred[i] else "✗"
    print(f"{status} Sample {i+1}: True={true_label:15s} | Predicted={pred_label:15s} | Confidence={confidence:5.2f}%")

print("\n" + "="*80)
print("Thank you for using Fabric Defect Detection!")
print("="*80)
