# 🧠 Brain Tumor Segmentation - Final Project

## CSC 590 - Medical Image Segmentation using Deep Learning

---

## 📋 Project Overview

This project implements a complete pipeline for 3D brain tumor segmentation using the Medical Segmentation Decathlon (MSD) Task01_BrainTumour dataset. It includes:

- **Baseline Model**: 3D U-Net with DiceCE loss
- **Improvements**: Post-processing, TTA, enhanced architectures
- **Evaluation**: BraTS-style metrics (WT, TC, ET)
- **Analysis**: Training curves, predictions, failure analysis

---

## 🎯 Current Status

✅ **BASELINE COMPLETE** - Model trained and evaluated!

### Baseline Results:
- **Whole Tumor (WT)**: 0.9126 ± 0.0528 ✅ Excellent
- **Tumor Core (TC)**: 0.8126 ± 0.1859 ✅ Good
- **Enhancing Tumor (ET)**: 0.5985 ± 0.2443 ⚠️ Challenging
- **Mean Dice Score**: **0.7746** ✅ Solid baseline

---

## 🚀 Quick Start (For Final Submission)

### Option 1: Minimum Viable (15 minutes) ⚡
**Already have a baseline! Just add improvements:**

1. Open the notebook in Google Colab
2. Run **Cell 30** - Post-processing evaluation (~10 min)
3. Run **Cell 33** - Generate final comparison (~2 min)
4. Download results from Google Drive
5. Include in your report

**Expected Result**: Mean Dice **~0.79-0.80** (+2-3% improvement)

### Option 2: Recommended (1-2 hours) 🎯
All of Option 1, plus:

4. In Cell 31, set `RUN_TTA = True`
5. Re-run Cell 31 (~1 hour)
6. Re-run Cell 33

**Expected Result**: Mean Dice **~0.80-0.82** (+3-5% improvement)

### Option 3: Advanced (8-12 hours) 🚀
All of Option 2, plus:

7. In Cell 32, set `TRAIN_IMPROVED = True`
8. Let train overnight (~8-10 hours)
9. Re-run Cell 33

**Expected Result**: Mean Dice **~0.83-0.88** (+7-12% improvement)

---

## 📁 Project Structure

```
CSC 590 Final Project/
│
├── BrainTumor_Starter_Visualization_(4).ipynb  # Main notebook (READY TO RUN!)
│
├── FINAL_STAGE_SUMMARY.md      # Detailed explanation of new cells
├── QUICK_START.md              # Quick reference guide
├── README_FINAL_PROJECT.md     # This file
│
└── /content/drive/MyDrive/BrainTumor/
    ├── Task01_BrainTumour/     # Dataset
    │   ├── imagesTr/           # Training images (484 cases)
    │   ├── labelsTr/           # Training labels
    │   └── dataset.json        # Metadata
    │
    └── models/                 # All outputs (auto-generated)
        ├── best_3d_unet_model.pth              # Best model checkpoint ✅
        ├── training_curves.png                  # Training visualization ✅
        ├── predictions_visualization.png        # Sample predictions ✅
        ├── test_results.json                    # Baseline metrics ✅
        ├── failure_analysis.json                # Worst/best cases ✅
        ├── experiments.json                     # Experiment tracking ✅
        │
        ├── test_results_postprocessing.json     # Post-proc results (NEW)
        ├── final_summary_table.csv              # Summary table (NEW)
        └── final_experiments_comparison.png     # Comparison chart (NEW)
```

---

## 📊 Dataset

**Source**: Medical Segmentation Decathlon - Task01_BrainTumour

### Statistics:
- **Total Cases**: 484
- **Training**: 338 (70%)
- **Validation**: 72 (15%)
- **Test**: 74 (15%)

### Modalities:
- FLAIR (T2-weighted FLAIR)
- T1w (T1-weighted)
- T1ce (T1-weighted with contrast enhancement)
- T2w (T2-weighted)

### Labels:
- 0: Background
- 1: Edema (peritumoral edema)
- 2: Enhancing tumor
- 3: Non-enhancing tumor core

### BraTS Regions:
- **WT (Whole Tumor)**: All tumor classes (1+2+3)
- **TC (Tumor Core)**: Classes 2+3
- **ET (Enhancing Tumor)**: Class 2 only

---

## 🏗️ Model Architecture

### Baseline: 3D U-Net

```
Input: 4 channels × 128×128×128 (4 MRI modalities)
↓
Encoder: 5 levels with (32, 64, 128, 256, 512) channels
↓
Decoder: 5 levels with skip connections
↓
Output: 4 channels × 128×128×128 (background + 3 tumor classes)
```

**Parameters**: 19,225,897 (~19.2M)  
**Size**: ~73.3 MB

### Training Configuration:
- **Loss**: DiceCE (50% Dice + 50% Cross Entropy)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: Cosine Annealing (T_max=100)
- **Batch Size**: 2
- **Epochs**: 100
- **Precision**: Mixed (FP16)
- **Inference**: Sliding Window (128³ ROI, sw_batch_size=4)

---

## 🔧 Improvements Implemented

### 1. Post-Processing (Cell 30) ⭐
**No retraining required!**

Techniques:
- Remove small objects (< 100-200 voxels)
- Fill holes in tumor regions
- Morphological closing (smooth boundaries)
- Connected component analysis

**Expected Improvement**: +1-3% Dice

### 2. Test-Time Augmentation (Cell 31)
Strategy:
- Predict on original image
- Predict on 3 axis-flipped versions
- Average all predictions

**Expected Improvement**: Additional +1-2% Dice

### 3. Enhanced Model (Cell 32)
**Requires retraining**

Changes:
- **Architecture**: Attention U-Net (attention gates on skip connections)
- **Loss**: DiceFocal (gamma=2.0, focuses on hard examples)
- **Augmentation**: 
  - Elastic deformation
  - Gaussian noise & smoothing
  - Contrast adjustment
  - Coarse dropout
  - Multi-axis rotations and flips
- **Training**: 150 epochs

**Expected Improvement**: +5-10% Dice

---

## 📈 Results

### Baseline (Already Achieved ✅)

| Metric | Mean ± Std | Median |
|--------|-----------|--------|
| **WT Dice** | 0.9126 ± 0.0528 | 0.9286 |
| **TC Dice** | 0.8126 ± 0.1859 | 0.8845 |
| **ET Dice** | 0.5985 ± 0.2443 | 0.6500 |
| **Mean Dice** | **0.7746** | - |

### Post-Processing (Run Cell 30)
Expected improvements shown after execution

### With TTA (Run Cell 31)
Expected improvements shown after execution

### Enhanced Model (Run Cell 32)
Expected improvements shown after execution

---

## 📊 Files Generated for Report

### Metrics & Results
1. `test_results.json` - Baseline detailed metrics ✅
2. `test_results_postprocessing.json` - With post-processing
3. `test_results_tta_postprocessing.json` - With TTA
4. `test_results_improved_model.json` - Enhanced model results
5. `final_experiments_comparison.json` - All experiments compiled
6. `final_summary_table.csv` - Summary table for easy inclusion

### Visualizations
1. `training_curves.png` - Loss and metrics over epochs ✅
2. `predictions_visualization.png` - Sample predictions ✅
3. `final_experiments_comparison.png` - Performance comparison chart
4. `training_curves_improved.png` - Enhanced model training

### Analysis
1. `failure_analysis.json` - Worst 5 and best 5 cases ✅

### Models
1. `best_3d_unet_model.pth` - Baseline model checkpoint ✅
2. `best_attention_unet_focal.pth` - Enhanced model (if trained)

---

## 💡 Report Writing Guide

### 1. Introduction (1 page)
- Motivation: Why brain tumor segmentation is important
- Challenge: Multimodal MRI, complex tumor structures
- Approach: Deep learning with 3D U-Net
- Objectives: Accurate segmentation of WT, TC, ET

### 2. Related Work (0.5-1 page)
- U-Net architecture (Ronneberger et al.)
- BraTS challenge overview
- MONAI framework
- Recent advances (Attention mechanisms, Focal loss)

### 3. Dataset & Preprocessing (1 page)
- MSD Task01_BrainTumour dataset
- 484 cases, 4 modalities
- Preprocessing pipeline:
  - Resampling to 1mm isotropic
  - Orientation to RAS
  - Foreground cropping
  - Resizing to 128³
  - Intensity normalization
- Data augmentation (list the transforms)

### 4. Methodology (2 pages)

#### 4.1 Baseline Architecture
- 3D U-Net details
- Loss function (DiceCE)
- Training configuration
- Evaluation metrics (Dice for WT, TC, ET)

#### 4.2 Improvements
- Post-processing techniques (describe each)
- Test-time augmentation strategy
- Enhanced model (if trained):
  - Attention U-Net architecture
  - Focal loss for class imbalance
  - Enhanced augmentation

### 5. Experiments & Results (2-3 pages)

#### 5.1 Baseline Results
- Table with WT, TC, ET Dice scores
- Discuss which regions are easier/harder
- Show training curves

#### 5.2 Ablation Studies
- Effect of post-processing
- Effect of TTA
- Effect of enhanced model (if trained)
- Use the comparison table from Cell 33!

#### 5.3 Qualitative Results
- Include predictions_visualization.png
- Show best and worst cases
- Discuss what the model learns well vs struggles with

### 6. Discussion (1-2 pages)

#### 6.1 Analysis
- Why ET is hardest (smallest region, class imbalance)
- Why WT is easiest (largest region, clearer boundaries)
- Impact of each improvement

#### 6.2 Failure Analysis
- Cases with low Dice scores
- Possible reasons: small tumors, unclear boundaries, artifacts
- What could improve these cases

#### 6.3 Limitations
- Fixed input size (128³)
- Class imbalance (ET is rare)
- Computational cost (training time)
- Dataset size (484 cases)

### 7. Conclusion & Future Work (0.5-1 page)

#### Achievements
- Implemented complete segmentation pipeline
- Baseline: 0.77 Mean Dice
- Improvements: +X% with post-processing/TTA
- All code and experiments documented

#### Future Directions
- Ensemble methods (combine multiple models)
- Transformer architectures (recent advances)
- Multi-task learning (predict tumor grade)
- Uncertainty quantification (clinical safety)
- Larger datasets (more training data)

---

## 🎓 Academic Quality Checklist

- [ ] **Methodology clearly described** (reproducible)
- [ ] **Results properly presented** (tables and figures)
- [ ] **Comparisons are fair** (same test set, metrics)
- [ ] **Ablation studies included** (isolate each improvement)
- [ ] **Failure cases analyzed** (understand limitations)
- [ ] **References cited** (MONAI, U-Net, BraTS, etc.)
- [ ] **Code is documented** (all cells have comments)
- [ ] **Figures are clear** (labels, legends, captions)
- [ ] **Discussion is insightful** (not just reporting numbers)
- [ ] **Future work is specific** (not vague suggestions)

---

## 📚 Key References

1. **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
2. **BraTS**: Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", TMI 2015
3. **MONAI**: "MONAI: Medical Open Network for AI", https://monai.io/
4. **MSD**: Simpson et al., "A large annotated medical image dataset for the development and evaluation of segmentation algorithms", arXiv 2019
5. **Attention U-Net**: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", MIDL 2018
6. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

---

## 🔧 Technical Details

### System Requirements
- **GPU**: NVIDIA T4 or better (Google Colab free tier works!)
- **RAM**: 12-16 GB
- **Storage**: ~50 GB for dataset + models

### Software Dependencies
```
Python 3.8+
PyTorch 1.12+
MONAI 1.0+
nibabel
SimpleITK
matplotlib
seaborn
scikit-image
scipy
pandas
```

All dependencies are installed in Cell 1 of the notebook.

### Training Time
- **Baseline (100 epochs)**: ~3-5 hours on T4 GPU ✅ Done!
- **Post-processing eval**: ~10-15 minutes
- **TTA eval**: ~40-60 minutes
- **Enhanced model (150 epochs)**: ~7-10 hours on T4 GPU

### Inference Time
- **Standard**: ~3-5 seconds per case
- **With TTA**: ~12-20 seconds per case

---

## 🚨 Troubleshooting

### Issue: Out of Memory (OOM)
**Symptoms**: CUDA OOM error during training  
**Solution**: Reduce `BATCH_SIZE` from 2 to 1 in Cell 5

### Issue: Training Too Slow
**Symptoms**: Taking longer than expected  
**Solutions**:
- Reduce `cache_rate` from 1.0 to 0.5
- Use fewer epochs (50-75 instead of 100)
- Check GPU is being used (`device = cuda`)

### Issue: Google Drive Disconnects
**Symptoms**: Drive unmounts during long training  
**Solution**: Model checkpoints are saved. Remount Drive and load checkpoint to resume

### Issue: Poor ET Segmentation
**Symptoms**: ET Dice < 0.5  
**Solutions**:
- Use post-processing (Cell 30)
- Use focal loss (Cell 32)
- Train longer (more epochs)
- Check class weights

### Issue: Validation Plateaus
**Symptoms**: Val Dice stops improving  
**Solutions**:
- Check for overfitting (val loss increasing?)
- Reduce learning rate
- Add more augmentation
- Try early stopping

---

## ✅ Final Checklist

### Before Running
- [ ] Google Drive mounted
- [ ] Dataset path is correct
- [ ] GPU is available

### After Baseline (Already Done ✅)
- [ ] Model trained successfully
- [ ] Test results generated
- [ ] Training curves saved
- [ ] Predictions visualized

### For Final Submission
- [ ] Run Cell 30 (post-processing)
- [ ] Run Cell 33 (final comparison)
- [ ] Download all generated files
- [ ] Review all visualizations
- [ ] Check experiment tracker results

### For Report
- [ ] Include methodology description
- [ ] Include results table
- [ ] Include training curves figure
- [ ] Include predictions figure
- [ ] Include comparison chart
- [ ] Discuss failure cases
- [ ] Cite references
- [ ] Proofread everything

---

## 🎉 Success Criteria

Your project is complete when you have:

✅ **Baseline Model** 
- [x] 3D U-Net trained
- [x] Mean Dice > 0.70 (you have 0.7746!)
- [x] WT, TC, ET metrics reported
- [x] Training curves generated

✅ **Improvements**
- [ ] At least one improvement implemented
- [ ] Ablation study showing impact
- [ ] Comparison table generated

✅ **Analysis**
- [x] Sample predictions visualized
- [x] Failure analysis completed
- [ ] Discussion of results

✅ **Documentation**
- [x] Code is well-commented
- [x] Results are saved
- [ ] Report is written

---

## 🎓 Grading Considerations

### Likely Evaluation Criteria

**Implementation (40%)**
- ✅ Correct architecture
- ✅ Proper data handling
- ✅ Training pipeline works
- ✅ Evaluation metrics correct

**Results (30%)**
- ✅ Baseline performance reasonable
- [ ] Improvements demonstrated
- ✅ Results properly reported
- [ ] Comparisons are fair

**Analysis (20%)**
- ✅ Training curves analyzed
- ✅ Failure cases examined
- [ ] Insights are meaningful
- [ ] Limitations discussed

**Presentation (10%)**
- [ ] Report is well-written
- [ ] Figures are clear
- [ ] Code is documented
- [ ] Professional quality

---

## 📧 Support

If you encounter issues:
1. Check `FINAL_STAGE_SUMMARY.md` for detailed explanations
2. Check `QUICK_START.md` for quick reference
3. Review Cell 34 in the notebook for troubleshooting
4. All error messages are designed to be helpful!

---

## 🏆 Achievement Unlocked!

You have:
- ✅ Implemented a state-of-the-art medical image segmentation pipeline
- ✅ Trained a 3D U-Net model successfully
- ✅ Achieved competitive results (0.77 Mean Dice)
- ✅ Set up framework for improvements
- ✅ Generated all materials for final report

**Now go get that grade! 🎓✨**

---

**Project Status**: ✅ READY FOR FINAL SUBMISSION  
**Last Updated**: November 16, 2025  
**Author**: CSC 590 Final Project Team  
**License**: MIT (for educational purposes)

