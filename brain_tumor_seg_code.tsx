import React, { useState } from 'react';
import { FileCode, Database, Brain, BarChart3, Settings, BookOpen } from 'lucide-react';

const BrainTumorSegmentationCode = () => {
  const [activeTab, setActiveTab] = useState('setup');

  const tabs = [
    { id: 'setup', label: 'Setup', icon: Settings },
    { id: 'dataset', label: 'Dataset', icon: Database },
    { id: 'preprocessing', label: 'Preprocessing', icon: Settings },
    { id: 'dataloaders', label: 'DataLoaders', icon: Database },
    { id: 'model', label: 'Models', icon: Brain },
    { id: 'loss', label: 'Loss', icon: BarChart3 },
    { id: 'metrics', label: 'Metrics', icon: BarChart3 },
    { id: 'training', label: 'Training', icon: Brain },
    { id: 'evaluation', label: 'Evaluation', icon: BarChart3 },
    { id: 'medsam', label: 'MedSAM', icon: Brain },
    { id: 'failure', label: 'Failure Analysis', icon: FileCode },
    { id: 'experiments', label: 'Experiments', icon: BarChart3 },
    { id: 'utils', label: 'Utils', icon: BookOpen }
  ];

  const codeSnippets = {
    setup: `# Brain Tumor Segmentation Project Setup
# Install required packages (run in Colab)

!pip install monai[all]
!pip install nibabel
!pip install SimpleITK
!pip install matplotlib seaborn
!pip install tensorboard

import os
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import monai
from monai.data import CacheDataset, DataLoader as MonaiDataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, CropForegroundd, ResizeWithPadOrCropd,
    NormalizeIntensityd, RandRotate90d, RandFlipd,
    RandScaleIntensityd, RandShiftIntensityd, RandAffined
)
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from pathlib import Path
import matplotlib.pyplot as plt
import json

# Set random seeds
monai.utils.set_determinism(seed=42)
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")`,

    dataset: `# Dataset Preparation for MSD Task01_BrainTumour

class MSDDatasetPreparation:
    def __init__(self, data_root, train_ratio=0.7, val_ratio=0.15):
        self.data_root = Path(data_root)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
    def load_dataset_json(self):
        json_path = self.data_root / "dataset.json"
        with open(json_path, 'r') as f:
            dataset_info = json.load(f)
        return dataset_info
    
    def prepare_data_dicts(self):
        dataset_info = self.load_dataset_json()
        data_dicts = []
        
        for item in dataset_info['training']:
            image_path = self.data_root / item['image']
            label_path = self.data_root / item['label']
            
            data_dicts.append({
                'image': str(image_path),
                'label': str(label_path)
            })
        
        return data_dicts
    
    def split_dataset(self, data_dicts, seed=42):
        np.random.seed(seed)
        n_total = len(data_dicts)
        indices = np.random.permutation(n_total)
        
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        train_files = [data_dicts[i] for i in indices[:n_train]]
        val_files = [data_dicts[i] for i in indices[n_train:n_train+n_val]]
        test_files = [data_dicts[i] for i in indices[n_train+n_val:]]
        
        print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        return train_files, val_files, test_files

# Initialize
data_prep = MSDDatasetPreparation(data_root="/content/Task01_BrainTumour")
data_dicts = data_prep.prepare_data_dicts()
train_files, val_files, test_files = data_prep.split_dataset(data_dicts)`,

    preprocessing: `# Preprocessing Transforms

def get_preprocessing_transforms(mode='train'):
    common_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), 
                mode=("bilinear", "nearest")),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(128, 128, 128)),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    
    if mode == 'train':
        train_transforms = common_transforms + [
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ]
        return Compose(train_transforms)
    
    return Compose(common_transforms)

train_transforms = get_preprocessing_transforms(mode='train')
val_transforms = get_preprocessing_transforms(mode='val')
test_transforms = get_preprocessing_transforms(mode='test')`,

    dataloaders: `# Create DataLoaders

def create_dataloaders(train_files, val_files, test_files, 
                       train_transforms, val_transforms, test_transforms,
                       batch_size=2):
    
    train_ds = CacheDataset(data=train_files, transform=train_transforms, 
                           cache_rate=1.0, num_workers=4)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, 
                         cache_rate=1.0, num_workers=4)
    test_ds = CacheDataset(data=test_files, transform=test_transforms, 
                          cache_rate=1.0, num_workers=4)
    
    train_loader = MonaiDataLoader(train_ds, batch_size=batch_size, 
                                  shuffle=True, num_workers=0)
    val_loader = MonaiDataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader = MonaiDataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = create_dataloaders(
    train_files, val_files, test_files,
    train_transforms, val_transforms, test_transforms, batch_size=2
)`,

    model: `# Model Architecture

def create_3d_unet(in_channels=4, out_channels=4):
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.1
    )
    return model

model = create_3d_unet(in_channels=4, out_channels=4).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")`,

    loss: `# Loss Functions

loss_function = DiceCELoss(
    include_background=False,
    to_onehot_y=True,
    softmax=True,
    lambda_dice=0.5,
    lambda_ce=0.5
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scaler = GradScaler()`,

    metrics: `# Evaluation Metrics

def compute_region_metrics(pred, label):
    pred_np = pred.cpu().numpy()
    label_np = label.cpu().numpy()
    
    wt_pred = (pred_np > 0).astype(np.float32)
    wt_label = (label_np > 0).astype(np.float32)
    
    tc_pred = ((pred_np == 2) | (pred_np == 3)).astype(np.float32)
    tc_label = ((label_np == 2) | (label_np == 3)).astype(np.float32)
    
    et_pred = (pred_np == 2).astype(np.float32)
    et_label = (label_np == 2).astype(np.float32)
    
    def dice_score(pred, label):
        smooth = 1e-5
        intersection = np.sum(pred * label)
        union = np.sum(pred) + np.sum(label)
        return (2.0 * intersection + smooth) / (union + smooth)
    
    return {
        'dice_wt': dice_score(wt_pred, wt_label),
        'dice_tc': dice_score(tc_pred, tc_label),
        'dice_et': dice_score(et_pred, et_label)
    }`,

    training: `# Training Loop

def train_epoch(model, train_loader, optimizer, loss_function, scaler, device):
    model.train()
    epoch_loss = 0
    
    for batch_data in train_loader:
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader)

def validate(model, val_loader, loss_function, device):
    model.eval()
    val_loss = 0
    all_metrics = {'dice_wt': [], 'dice_tc': [], 'dice_et': []}
    
    with torch.no_grad():
        for batch_data in val_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            outputs = sliding_window_inference(
                inputs, roi_size=(128, 128, 128),
                sw_batch_size=4, predictor=model
            )
            
            val_loss += loss_function(outputs, labels).item()
            preds = torch.argmax(outputs, dim=1, keepdim=True)
            
            metrics = compute_region_metrics(preds[0, 0], labels[0, 0])
            for k, v in metrics.items():
                all_metrics[k].append(v)
    
    return val_loss / len(val_loader), {k: np.mean(v) for k, v in all_metrics.items()}

# Training
history = {'train_loss': [], 'val_loss': []}
best_dice = 0

for epoch in range(100):
    train_loss = train_epoch(model, train_loader, optimizer, loss_function, scaler, device)
    val_loss, metrics = validate(model, val_loader, loss_function, device)
    
    scheduler.step()
    
    print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
    print(f"  WT={metrics['dice_wt']:.4f}, TC={metrics['dice_tc']:.4f}, ET={metrics['dice_et']:.4f}")
    
    mean_dice = np.mean(list(metrics.values()))
    if mean_dice > best_dice:
        best_dice = mean_dice
        torch.save(model.state_dict(), 'best_model.pth')`,

    evaluation: `# Evaluation

def evaluate_test_set(model, test_loader, device):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    all_metrics = {'dice_wt': [], 'dice_tc': [], 'dice_et': []}
    
    with torch.no_grad():
        for batch_data in test_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            outputs = sliding_window_inference(
                inputs, roi_size=(128, 128, 128),
                sw_batch_size=4, predictor=model
            )
            
            preds = torch.argmax(outputs, dim=1, keepdim=True)
            metrics = compute_region_metrics(preds[0, 0], labels[0, 0])
            
            for k, v in metrics.items():
                all_metrics[k].append(v)
    
    results = {f'{k}_mean': np.mean(v) for k, v in all_metrics.items()}
    results.update({f'{k}_std': np.std(v) for k, v in all_metrics.items()})
    
    print("TEST RESULTS:")
    print(f"WT: {results['dice_wt_mean']:.4f} ± {results['dice_wt_std']:.4f}")
    print(f"TC: {results['dice_tc_mean']:.4f} ± {results['dice_tc_std']:.4f}")
    print(f"ET: {results['dice_et_mean']:.4f} ± {results['dice_et_std']:.4f}")
    
    return results

results = evaluate_test_set(model, test_loader, device)`,

    medsam: `# MedSAM Fine-tuning

!pip install git+https://github.com/bowang-lab/MedSAM.git
!wget https://github.com/bowang-lab/MedSAM/releases/download/v0.1/medsam_vit_b.pth

from segment_anything import sam_model_registry

class MedSAMFineTune(nn.Module):
    def __init__(self, checkpoint_path, num_classes=4):
        super().__init__()
        self.medsam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        
        for param in self.medsam.image_encoder.parameters():
            param.requires_grad = False
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, 1)
        )
    
    def forward(self, x):
        B, C, H, W, D = x.shape
        outputs = []
        for d in range(D):
            slice_x = x[:, :, :, :, d].mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            features = self.medsam.image_encoder(slice_x)
            outputs.append(self.seg_head(features))
        return torch.stack(outputs, dim=-1)

medsam_model = MedSAMFineTune('medsam_vit_b.pth', num_classes=4).to(device)`,

    failure: `# Failure Analysis

def analyze_failures(model, test_loader, device):
    model.eval()
    cases = []
    
    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            outputs = sliding_window_inference(inputs, (128, 128, 128), 4, model)
            preds = torch.argmax(outputs, dim=1, keepdim=True)
            
            metrics = compute_region_metrics(preds[0, 0], labels[0, 0])
            
            cases.append({
                'id': idx,
                'dice_wt': metrics['dice_wt'],
                'dice_tc': metrics['dice_tc'],
                'dice_et': metrics['dice_et'],
                'mean': np.mean(list(metrics.values()))
            })
    
    cases.sort(key=lambda x: x['mean'])
    
    print("WORST CASES:")
    for i, case in enumerate(cases[:5]):
        print(f"{i+1}. Case {case['id']}: Mean={case['mean']:.4f}")
        print(f"   WT={case['dice_wt']:.4f}, TC={case['dice_tc']:.4f}, ET={case['dice_et']:.4f}")
    
    return cases

failures = analyze_failures(model, test_loader, device)`,

    experiments: `# Experiment Tracking

class ExperimentTracker:
    def __init__(self):
        self.experiments = {}
    
    def log(self, name, config, results):
        self.experiments[name] = {'config': config, 'results': results}
        print(f"Logged: {name}")
    
    def compare(self):
        import pandas as pd
        rows = []
        for name, exp in self.experiments.items():
            row = {'Name': name}
            row.update(exp['results'])
            rows.append(row)
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        return df

tracker = ExperimentTracker()
tracker.log('3D_UNet_DiceCE', {'model': '3D U-Net'}, results)
tracker.compare()`,

    utils: `# Utility Functions

def save_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.get('val_dice', []), color='green')
    axes[1].set_title('Validation Dice')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()

def visualize_predictions(model, test_loader, device, n=5):
    model.eval()
    fig, axes = plt.subplots(n, 4, figsize=(16, n*4))
    
    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            if idx >= n:
                break
            
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            outputs = sliding_window_inference(inputs, (128, 128, 128), 4, model)
            preds = torch.argmax(outputs, dim=1)
            
            z = inputs.shape[-1] // 2
            
            axes[idx, 0].imshow(inputs[0, 0, :, :, z].cpu(), cmap='gray')
            axes[idx, 0].set_title('FLAIR')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(inputs[0, 1, :, :, z].cpu(), cmap='gray')
            axes[idx, 1].set_title('T1ce')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(labels[0, 0, :, :, z].cpu(), cmap='jet')
            axes[idx, 2].set_title('GT')
            axes[idx, 2].axis('off')
            
            axes[idx, 3].imshow(preds[0, :, :, z].cpu(), cmap='jet')
            axes[idx, 3].set_title('Pred')
            axes[idx, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150)
    plt.show()`
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center gap-3 mb-2">
            <Brain className="w-10 h-10 text-indigo-600" />
            <h1 className="text-3xl font-bold text-gray-800">
              Brain Tumor Segmentation Boilerplate
            </h1>
          </div>
          <p className="text-gray-600">
            MSD Task01 | 3D U-Net + MedSAM | MONAI + PyTorch
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-lg mb-6 overflow-x-auto">
          <div className="flex gap-1 p-2">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all whitespace-nowrap ${
                    activeTab === tab.id
                      ? 'bg-indigo-600 text-white shadow-md'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="font-medium">{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="bg-gray-900 rounded-lg shadow-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold text-white">
              {tabs.find(t => t.id === activeTab)?.label}
            </h2>
            <button
              onClick={() => {
                navigator.clipboard.writeText(codeSnippets[activeTab]);
                alert('Code copied!');
              }}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg font-medium"
            >
              Copy Code
            </button>
          </div>
          
          <pre className="overflow-x-auto">
            <code className="text-sm text-gray-100 font-mono leading-relaxed whitespace-pre">
              {codeSnippets[activeTab]}
            </code>
          </pre>
        </div>

        <div className="mt-6 bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-bold text-gray-800 mb-3">Quick Start</h3>
          <div className="space-y-2 text-gray-700">
            <p><strong>1. Setup:</strong> Run installation in Google Colab</p>
            <p><strong>2. Dataset:</strong> Download MSD Task01_BrainTumour</p>
            <p><strong>3. Train:</strong> Start with 3D U-Net baseline</p>
            <p><strong>4. Fine-tune:</strong> Run MedSAM with frozen encoder</p>
            <p><strong>5. Analyze:</strong> Use failure analysis for report</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BrainTumorSegmentationCode;