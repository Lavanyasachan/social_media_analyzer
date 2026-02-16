# How to Run Fabric Defect Detection in Kaggle

This guide explains how to use `fabric_defect_kaggle.py` in your Kaggle account.

## Quick Start (3 Steps)

### Step 1: Create a New Kaggle Notebook/Script

1. Go to [Kaggle.com](https://www.kaggle.com)
2. Click **"Code"** in the top menu
3. Click **"New Notebook"** or **"New Script"**
4. Choose **"Script"** for running the .py file directly

### Step 2: Add the Dataset

1. In the right panel, click **"+ Add Data"**
2. Search for **"fabric-defects-dataset"**
3. Find the dataset by **nexuswho**
4. Click **"Add"** to add it to your notebook

The dataset will be available at: `/kaggle/input/fabric-defects-dataset`

### Step 3: Upload and Run the Script

#### Option A: Upload the Python File
1. Click **"File"** → **"Upload"** (or the upload icon)
2. Upload `fabric_defect_kaggle.py`
3. In a code cell, run:
   ```python
   %run fabric_defect_kaggle.py
   ```

#### Option B: Copy-Paste the Code
1. Open `fabric_defect_kaggle.py` in a text editor
2. Copy all the code
3. Paste it into a Kaggle code cell
4. Run the cell

#### Option C: Use as a Python Script
1. Upload `fabric_defect_kaggle.py` to Kaggle
2. In Settings, change notebook type to **"Script"**
3. The script will run automatically when you click **"Run All"**

## What the Script Does

The script performs a complete fabric defect detection pipeline:

1. **Loads Dataset**: Automatically finds the dataset in Kaggle's environment
2. **Prepares Data**: Creates training/validation splits with data augmentation
3. **Builds Model**: Constructs a ResNet50 CNN with transfer learning
4. **Trains Model**: Trains for up to 50 epochs with early stopping
5. **Evaluates**: Computes accuracy, confusion matrix, and per-class metrics
6. **Visualizes**: Creates plots of training history and results
7. **Saves Output**: Saves models and plots to `/kaggle/working/`

## Expected Output

When the script completes, you'll get:

### Console Output
- Training progress with accuracy/loss per epoch
- Final accuracy score and classification report
- Summary of results

### Saved Files (in `/kaggle/working/`)
- `fabric_defect_resnet_model.h5` - Final trained model
- `best_fabric_defect_model.h5` - Best model checkpoint
- `training_history.png` - Training curves (accuracy, loss, precision, recall)
- `confusion_matrix.png` - Confusion matrix heatmap
- `per_class_accuracy.png` - Per-class accuracy bar chart

### Visualizations
The script displays plots directly in the output:
- Training history (4 subplots)
- Confusion matrix
- Per-class accuracy

## Configuration Options

You can modify these variables at the top of the script:

```python
# Dataset path (auto-detected in Kaggle)
DATASET_PATH = "/kaggle/input/fabric-defects-dataset"

# Model hyperparameters
IMG_SIZE = (224, 224)  # Image size
BATCH_SIZE = 32        # Batch size
EPOCHS = 50            # Maximum epochs
VALIDATION_SPLIT = 0.2 # 20% for validation
LEARNING_RATE = 0.001  # Initial learning rate
```

## Typical Runtime

- **With GPU**: 5-15 minutes per epoch → Total: 30-90 minutes
- **Without GPU**: 30-60 minutes per epoch → Total: 3-5 hours

**Tip**: Enable GPU in Kaggle for faster training!
- Go to **Settings** (right panel)
- Under **Accelerator**, select **"GPU"**

## Troubleshooting

### Problem: "Dataset not found"
**Solution**: Make sure you added the dataset using "Add Data" in Kaggle

### Problem: "Out of memory"
**Solutions**:
- Reduce `BATCH_SIZE` from 32 to 16 or 8
- Enable GPU accelerator
- Use a smaller `IMG_SIZE` (not recommended as ResNet expects 224×224)

### Problem: Script runs too long
**Solutions**:
- Reduce `EPOCHS` from 50 to 20 or 30
- Early stopping will automatically stop if model stops improving

### Problem: Low accuracy
**Solutions**:
- Train for more epochs
- Check if dataset is loaded correctly
- Verify class balance in the dataset

## Example Output

```
================================================================================
FABRIC DEFECT DETECTION USING RESNET50 CNN - KAGGLE VERSION
================================================================================
TensorFlow version: 2.x.x
GPU Available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
================================================================================

Configuration:
  Dataset Path: /kaggle/input/fabric-defects-dataset
  Image Size: (224, 224)
  Batch Size: 32
  Max Epochs: 50
  Output Directory: /kaggle/working
================================================================================

[1/7] DATASET PREPARATION
--------------------------------------------------------------------------------
✓ Dataset found at: /kaggle/input/fabric-defects-dataset

[2/7] DATA GENERATORS SETUP
--------------------------------------------------------------------------------
Found 1200 images belonging to 4 classes.
Found 300 images belonging to 4 classes.
✓ Data generators created successfully

Dataset Statistics:
  Number of classes: 4
  Class names: ['defect_type1', 'defect_type2', 'defect_type3', 'no_defect']
  Training samples: 1200
  Validation samples: 300

[3/7] BUILDING RESNET50 MODEL
--------------------------------------------------------------------------------
✓ ResNet50 model built successfully!

[4/7] CONFIGURING TRAINING CALLBACKS
--------------------------------------------------------------------------------
✓ Callbacks configured

[5/7] TRAINING MODEL
--------------------------------------------------------------------------------
Epoch 1/50
38/38 [==============================] - 45s - loss: 1.2345 - accuracy: 0.6543
...
Epoch 25/50
38/38 [==============================] - 42s - loss: 0.1234 - accuracy: 0.9543
✓ Training completed!

[6/7] EVALUATING MODEL
--------------------------------------------------------------------------------
Overall Accuracy: 0.9533 (95.33%)

[7/7] GENERATING VISUALIZATIONS
--------------------------------------------------------------------------------
✓ Training history saved to: /kaggle/working/training_history.png
✓ Confusion matrix saved to: /kaggle/working/confusion_matrix.png
✓ Per-class accuracy saved to: /kaggle/working/per_class_accuracy.png

================================================================================
TRAINING AND EVALUATION COMPLETE!
================================================================================
```

## Downloading Results from Kaggle

After the script finishes:

1. Click on **"Output"** tab in the right panel
2. You'll see all saved files (models and plots)
3. Click the **download** icon next to each file
4. Or click **"Download All"** to get everything

## Advanced Usage

### Using the Trained Model

After training, you can load and use the model:

```python
from tensorflow import keras

# Load the model
model = keras.models.load_model('/kaggle/working/best_fabric_defect_model.h5')

# Make predictions on new images
# predictions = model.predict(your_images)
```

### Fine-tuning

To fine-tune the model with unfrozen layers, add this after initial training:

```python
# Unfreeze ResNet50 layers
base_model.trainable = True

# Freeze all but last 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20
)
```

## Support

For issues or questions:
- Check the main README: `README_FABRIC_DEFECT.md`
- Review implementation details: `IMPLEMENTATION_SUMMARY.md`
- Check architecture: `ARCHITECTURE.txt`

## Summary

This script provides a complete, production-ready fabric defect detection system optimized for Kaggle. Simply add the dataset, run the script, and get your trained model with evaluation metrics!

**Happy Training! 🚀**
