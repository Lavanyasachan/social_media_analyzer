# Fabric Defect Detection using ResNet CNN

A deep learning-based fabric defect detection system using ResNet50 architecture with transfer learning.

## Overview

This project implements a Convolutional Neural Network (CNN) model for detecting defects in fabric images. The model uses ResNet50 as the base architecture with transfer learning from ImageNet weights, providing high accuracy in classifying different types of fabric defects.

## Features

- **ResNet50 Architecture**: Leverages pre-trained ResNet50 model for transfer learning
- **Data Augmentation**: Implements various augmentation techniques to improve model generalization
- **Comprehensive Evaluation**: Provides accuracy metrics and confusion matrix visualization
- **Model Checkpointing**: Automatically saves the best model during training
- **Early Stopping**: Prevents overfitting with early stopping callbacks
- **Training Visualization**: Plots training history including accuracy, loss, precision, and recall

## Dataset

The project uses the [Fabric Defects Dataset](https://www.kaggle.com/datasets/nexuswho/fabric-defects-dataset) from Kaggle.

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("nexuswho/fabric-defects-dataset")
print("Path to dataset files:", path)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

- TensorFlow >= 2.10.0
- Keras >= 2.10.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0
- scikit-learn >= 1.0.0
- kagglehub >= 0.1.0
- Pillow >= 8.0.0

## Usage

### Option 1: Using Jupyter Notebook (Recommended)

1. Open the Jupyter notebook:
```bash
jupyter notebook fabric_defect_detection.ipynb
```

2. Run all cells sequentially to:
   - Download the dataset
   - Build the ResNet model
   - Train the model
   - Evaluate and visualize results

### Option 2: Using Python Script

Run the complete pipeline with a single command:

```bash
python fabric_defect_detection.py
```

### Option 3: Using the FabricDefectDetector Class

```python
from fabric_defect_detection import FabricDefectDetector

# Initialize detector
detector = FabricDefectDetector(img_size=(224, 224), batch_size=32)

# Download dataset
dataset_path = detector.download_dataset()

# Prepare data
train_gen, val_gen = detector.prepare_data_generators(dataset_path)

# Build and train model
num_classes = len(detector.class_names)
detector.build_resnet_model(num_classes)
detector.train_model(train_gen, val_gen, epochs=50)

# Evaluate
results = detector.evaluate_model(val_gen)

# Visualize
detector.plot_training_history()
detector.plot_confusion_matrix(results['confusion_matrix'])

# Save model
detector.save_model('my_model.h5')
```

## Model Architecture

The model consists of:

1. **Base Model**: ResNet50 (pre-trained on ImageNet, frozen initially)
2. **Global Average Pooling Layer**: Reduces spatial dimensions
3. **Batch Normalization**: Normalizes activations
4. **Dense Layer**: 512 units with ReLU activation
5. **Dropout**: 0.5 rate for regularization
6. **Dense Layer**: 256 units with ReLU activation
7. **Dropout**: 0.3 rate for regularization
8. **Output Layer**: Softmax activation for multi-class classification

## Training Configuration

- **Image Size**: 224 x 224 pixels (ResNet50 standard input)
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Optimizer**: Adam with learning rate of 0.001
- **Loss Function**: Categorical Cross-Entropy
- **Metrics**: Accuracy, Precision, Recall

### Data Augmentation

Training images undergo the following augmentations:
- Rotation: ±20 degrees
- Width/Height Shift: 20%
- Shear: 20%
- Zoom: 20%
- Horizontal Flip
- Pixel Rescaling: 0-1 range

## Evaluation Metrics

The model is evaluated using:

1. **Overall Accuracy**: Percentage of correctly classified samples
2. **Confusion Matrix**: Detailed breakdown of predictions vs. actual labels
3. **Classification Report**: Precision, recall, and F1-score for each class
4. **Per-Class Accuracy**: Individual accuracy for each defect type

## Output Files

After training, the following files are generated:

- `fabric_defect_resnet_model.h5`: Final trained model
- `best_fabric_defect_model.h5`: Best model checkpoint based on validation accuracy
- `training_history.png`: Visualization of training metrics over epochs
- `confusion_matrix.png`: Heatmap of the confusion matrix
- `per_class_accuracy.png`: Bar chart showing accuracy per class

## Results

The model provides:
- **Accuracy Score**: Overall classification accuracy on validation set
- **Confusion Matrix**: Visual representation of model predictions
- **Classification Report**: Detailed metrics including precision, recall, and F1-score
- **Training History**: Plots showing model performance over training epochs

## Advanced Features

### Fine-Tuning

To improve model performance, you can fine-tune the ResNet50 base model:

```python
detector.fine_tune_model(train_gen, val_gen, epochs=20)
```

This unfreezes the last 20 layers of ResNet50 and continues training with a lower learning rate.

## Project Structure

```
.
├── fabric_defect_detection.py          # Main Python script
├── fabric_defect_detection.ipynb       # Jupyter notebook
├── requirements.txt                    # Python dependencies
├── README_FABRIC_DEFECT.md            # This file
└── outputs/                           # Generated outputs
    ├── fabric_defect_resnet_model.h5
    ├── best_fabric_defect_model.h5
    ├── training_history.png
    ├── confusion_matrix.png
    └── per_class_accuracy.png
```

## Troubleshooting

### Kaggle Authentication

If you encounter authentication issues with kagglehub:

1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account settings and create an API token
3. Place the `kaggle.json` file in `~/.kaggle/`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Memory Issues

If you run into memory issues:
- Reduce batch size (e.g., from 32 to 16)
- Reduce image size (not recommended as ResNet50 expects 224x224)
- Use a machine with more RAM/GPU memory

### GPU Support

To use GPU acceleration:
1. Install CUDA and cuDNN
2. Install tensorflow-gpu: `pip install tensorflow-gpu`
3. Verify GPU is detected: `tf.config.list_physical_devices('GPU')`

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- ResNet50 architecture from [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Fabric Defects Dataset from Kaggle
- TensorFlow and Keras teams for the excellent deep learning framework

## Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: Make sure you have proper authentication set up for Kaggle to download the dataset automatically.
