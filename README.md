# Social Media Analyzer & Fabric Defect Detection

This repository contains two main projects:

## 1. Social Media Post Evaluator
- Located in: `project_socialmedia_post_evaluator.ipynb`
- Analyzes social media posts and engagement metrics

## 2. Fabric Defect Detection (NEW)
A deep learning-based fabric defect detection system using ResNet50 CNN architecture.

### Quick Start - Fabric Defect Detection

#### Local Machine
```bash
# Install dependencies
pip install -r requirements.txt

# Run the model (choose one option)
python fabric_defect_detection.py          # Full pipeline
python quick_start.py                      # Quick start with defaults
jupyter notebook fabric_defect_detection.ipynb  # Interactive notebook
```

#### Kaggle (Recommended for GPU)
```bash
# Use the Kaggle-optimized script
python fabric_defect_kaggle.py             # Standalone script for Kaggle
```
See [KAGGLE_INSTRUCTIONS.md](KAGGLE_INSTRUCTIONS.md) for detailed Kaggle setup guide.

### Features
- ✅ ResNet50 transfer learning architecture
- ✅ Automatic Kaggle dataset download
- ✅ Data augmentation pipeline
- ✅ Accuracy and confusion matrix evaluation
- ✅ Training history visualization
- ✅ Model checkpointing and early stopping

### Documentation
- **Complete Guide**: See [README_FABRIC_DEFECT.md](README_FABRIC_DEFECT.md)
- **Implementation Details**: See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

### Files
- `fabric_defect_detection.py` - Main implementation (for local use)
- `fabric_defect_kaggle.py` - **Kaggle-optimized standalone script** ⭐
- `fabric_defect_detection.ipynb` - Interactive notebook
- `quick_start.py` - Quick training script
- `requirements.txt` - Dependencies
- `validate_implementation.py` - Validation script (✓ All checks passed)
- **[KAGGLE_INSTRUCTIONS.md](KAGGLE_INSTRUCTIONS.md)** - Complete Kaggle setup guide

### Dataset
Uses the [Fabric Defects Dataset](https://www.kaggle.com/datasets/nexuswho/fabric-defects-dataset) from Kaggle.

```python
import kagglehub
path = kagglehub.dataset_download("nexuswho/fabric-defects-dataset")
```

For detailed instructions, see [README_FABRIC_DEFECT.md](README_FABRIC_DEFECT.md).
