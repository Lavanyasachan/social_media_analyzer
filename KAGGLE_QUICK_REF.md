# Kaggle Quick Reference Card

## 🚀 Fastest Way to Run in Kaggle

### Step 1: Add Dataset
In Kaggle → **Add Data** → Search **"fabric-defects-dataset"** by nexuswho → **Add**

### Step 2: Upload Script
Upload **`fabric_defect_kaggle.py`** to your Kaggle notebook

### Step 3: Run
```python
%run fabric_defect_kaggle.py
```

That's it! ✅

---

## 📝 Alternative: Copy-Paste Method

1. Open `fabric_defect_kaggle.py` in any text editor
2. Copy ALL the code (Ctrl+A, Ctrl+C)
3. In Kaggle, create a new code cell
4. Paste the code (Ctrl+V)
5. Run the cell

---

## ⚙️ Enable GPU (Recommended)

Before running:
1. Click **Settings** (right panel in Kaggle)
2. Under **Accelerator** → Select **GPU**
3. Click **Save**

This makes training **10x faster**! 🚀

---

## 📊 What You'll Get

### Console Output
- Training progress bar for each epoch
- Validation accuracy after each epoch
- Final accuracy score and detailed metrics

### Downloaded Files (from `/kaggle/working/`)
✅ `fabric_defect_resnet_model.h5` - Trained model  
✅ `best_fabric_defect_model.h5` - Best checkpoint  
✅ `training_history.png` - Training curves  
✅ `confusion_matrix.png` - Confusion matrix  
✅ `per_class_accuracy.png` - Per-class accuracy  

---

## ⏱️ Expected Runtime

| Mode | Time per Epoch | Total Time (50 epochs) |
|------|----------------|------------------------|
| GPU  | 5-10 minutes   | ~30-60 minutes         |
| CPU  | 30-60 minutes  | ~3-5 hours             |

**💡 Tip**: Training usually stops early (20-30 epochs) due to early stopping!

---

## 🔧 Quick Configuration

Edit these lines in the script if needed:

```python
# Line ~70: Dataset path
DATASET_PATH = "/kaggle/input/fabric-defects-dataset"

# Line ~73-77: Hyperparameters
BATCH_SIZE = 32      # Lower if out of memory (try 16 or 8)
EPOCHS = 50          # Max epochs (will stop early if converged)
IMG_SIZE = (224, 224)  # Don't change (ResNet50 standard)
```

---

## 🐛 Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| "Dataset not found" | Make sure dataset is added via "Add Data" |
| Out of memory | Reduce BATCH_SIZE to 16 or 8 |
| Takes too long | Enable GPU in Settings |
| Low accuracy | Check dataset, train longer, or verify classes |

---

## 📥 Download Results

After script finishes:
1. Click **Output** tab (right panel)
2. See all generated files
3. Click download icon next to each file
4. Or click **Download All**

---

## 💻 Script Features

✅ Fully self-contained (no external dependencies on other project files)  
✅ Auto-detects Kaggle environment  
✅ Comprehensive error messages  
✅ Progress tracking and status updates  
✅ Beautiful visualizations  
✅ Production-ready trained model  

---

## 📖 Need More Help?

- **Detailed Guide**: See `KAGGLE_INSTRUCTIONS.md`
- **Technical Details**: See `README_FABRIC_DEFECT.md`
- **Architecture**: See `ARCHITECTURE.txt`

---

## 🎯 Quick Summary

```
1. Add dataset in Kaggle
2. Upload fabric_defect_kaggle.py
3. Run: %run fabric_defect_kaggle.py
4. Wait ~30-60 minutes (with GPU)
5. Download results from Output tab
```

**You'll have a trained ResNet50 model for fabric defect detection!** 🎉

---

**File**: `fabric_defect_kaggle.py`  
**Size**: ~15KB  
**Lines**: ~530  
**Type**: Standalone Python script  
**Platform**: Optimized for Kaggle  
**GPU**: Highly recommended ✅
