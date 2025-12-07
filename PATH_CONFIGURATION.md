# Path Configuration Guide

## 🗂️ Your Current Setup

### Google Drive Structure
```
/content/drive/MyDrive/BrainTumor/
├── Task01_BrainTumour/              # Dataset root
│   ├── imagesTr/                    # Training images (484 cases)
│   │   ├── BRATS_001.nii
│   │   ├── BRATS_002.nii
│   │   └── ...
│   ├── labelsTr/                    # Training labels (484 cases)
│   │   ├── BRATS_001.nii
│   │   ├── BRATS_002.nii
│   │   └── ...
│   └── dataset.json                 # Dataset metadata
└── models/                          # Output directory (created automatically)
    ├── best_3d_unet_model.pth
    ├── training_curves.png
    ├── test_results.json
    ├── predictions_visualization.png
    ├── failure_analysis.json
    └── experiments.json
```

---

## 📝 Key Path Variables in Notebook

### Main Paths (Cell 1)
```python
BASE_DIR = "/content/drive/MyDrive/BrainTumor/Task01_BrainTumour"
SAVE_DIR = "/content/drive/MyDrive/BrainTumor/models"
```

### Derived Paths
```python
imagesTr = os.path.join(BASE_DIR, "imagesTr")
labelsTr = os.path.join(BASE_DIR, "labelsTr")
dataset_json = os.path.join(BASE_DIR, "dataset.json")
```

### Output File Paths
```python
best_model_path = os.path.join(SAVE_DIR, 'best_3d_unet_model.pth')
curve_path = os.path.join(SAVE_DIR, 'training_curves.png')
results_path = os.path.join(SAVE_DIR, 'test_results.json')
pred_path = os.path.join(SAVE_DIR, 'predictions_visualization.png')
failure_path = os.path.join(SAVE_DIR, 'failure_analysis.json')
experiments_path = os.path.join(SAVE_DIR, 'experiments.json')
```

---

## 🔧 If You Need to Change Paths

### Option 1: Different Dataset Location
If your dataset is in a different folder:

```python
# Change this line in Cell 1
BASE_DIR = "/content/drive/MyDrive/YOUR_FOLDER/Task01_BrainTumour"
```

### Option 2: Different Output Location
If you want outputs in a different folder:

```python
# Change this line in Cell 1
SAVE_DIR = "/content/drive/MyDrive/YOUR_FOLDER/outputs"
```

### Option 3: Local Machine (Not Colab)
If running on your local machine:

```python
# Update to your local paths
BASE_DIR = "C:/Users/YourName/Documents/BrainTumor/Task01_BrainTumour"
SAVE_DIR = "C:/Users/YourName/Documents/BrainTumor/models"

# Comment out Google Drive mount in Cell 1
# from google.colab import drive
# drive.mount('/content/drive')
```

---

## ✅ Path Verification

The notebook automatically verifies paths in Cell 1:

```
================================================================================
📁 DATASET VERIFICATION
================================================================================
Dataset root: /content/drive/MyDrive/BrainTumor/Task01_BrainTumour
Exists: True
imagesTr: .../imagesTr -> True
labelsTr: .../labelsTr -> True
dataset.json: .../dataset.json -> True

📊 Found 484 training image files and 484 label files.
Sample image: BRATS_001.nii
Sample label: BRATS_001.nii
================================================================================
```

If you see `False` or `Found 0 files`, your paths need adjustment!

---

## 🎯 Path Best Practices

1. **Use absolute paths** for clarity
2. **Use `os.path.join()`** for cross-platform compatibility
3. **Create output directories** with `os.makedirs(SAVE_DIR, exist_ok=True)`
4. **Verify paths exist** before running long training
5. **Keep dataset and outputs separate** for organization

---

## 🚨 Common Path Issues

### Issue 1: "File not found"
**Cause**: Path doesn't match your actual Google Drive structure  
**Fix**: Verify folder names match exactly (case-sensitive!)

### Issue 2: "Permission denied"
**Cause**: Google Drive not properly mounted  
**Fix**: Re-run the drive.mount() cell and authorize

### Issue 3: "Found 0 files"
**Cause**: Wrong dataset path or files not extracted  
**Fix**: Check if dataset is properly extracted from ZIP

### Issue 4: "Cannot write to directory"
**Cause**: Output directory doesn't exist or no permissions  
**Fix**: Notebook auto-creates it, but verify write permissions

---

## 📥 Dataset Download

If you haven't downloaded the dataset yet:

```bash
# Download from Medical Segmentation Decathlon
# URL: http://medicaldecathlon.com/

# Upload to Google Drive at:
# /content/drive/MyDrive/BrainTumor/Task01_BrainTumour/
```

---

## 🔍 Quick Path Test

Run this in a Colab cell to test your paths:

```python
import os
from glob import glob

BASE_DIR = "/content/drive/MyDrive/BrainTumor/Task01_BrainTumour"

print("Checking paths...")
print(f"Dataset exists: {os.path.exists(BASE_DIR)}")
print(f"Images exist: {os.path.exists(os.path.join(BASE_DIR, 'imagesTr'))}")
print(f"Labels exist: {os.path.exists(os.path.join(BASE_DIR, 'labelsTr'))}")

img_files = glob(os.path.join(BASE_DIR, "imagesTr", "*.nii*"))
print(f"Found {len(img_files)} image files")

if len(img_files) > 0:
    print("✅ Paths are correct!")
else:
    print("❌ Fix your paths!")
```

---

**Note**: All paths in the updated notebook use your existing Google Drive structure. No additional changes needed unless you want to reorganize your files!

