# MRI-Based Alzheimer’s Disease Classification using 3D CNN

This project implements a **3D Convolutional Neural Network (CNN)** for classifying MRI brain scans into two categories: **Healthy Controls (CN)** and **Alzheimer’s Disease (AD)**. The model is trained on structural MRI volumes in `.nii` or `.nii.gz` format and uses preprocessing techniques such as **MNI152 registration**, **intensity normalization**, and **data augmentation** to improve performance and generalization.

---
## Features

- MRI data preprocessing with **MNI152 registration** 
- Robust normalization and light augmentations (random flip, rotation, gamma jitter, noise).  
- Modular dataset loader `MRIVolumeDataset` 
- Efficient **3D CNN** architecture with global average pooling and dropout for stability.  
- Configurable YAML-based training and testing.  
- Slice visualization (axial, coronal, sagittal) and confusion matrix generation.

---

## Directory Structure

```
project/
│
├── data/
│   ├── Training/
│   │   ├── health/        # Healthy control MRIs (.nii/.nii.gz)
│   │   └── patient/       # Alzheimer’s patient MRIs (.nii/.nii.gz)
│   └── Testing/
│       ├── health/
│       └── patient/
│
├── src/
│   ├── data/dataset.py         # MRI dataset class with preprocessing and augmentation
│   ├── data/registrations.py   # MNI registration utilities
│   ├── models/cnn3d.py         # 3D CNN model architecture
│   ├── utils/metrics.py        # Accuracy, F1-score, confusion matrix computations
│   ├── utils/seed.py           # Seed value
│   ├── vis/plots.py            # Confusion matrix and loss visualization
│   ├── vis/slices.py           # Axial, coronal, sagittal slice visualization       
│
├── train.py                    # Model training and validation
├── test.py                     # Model inference and evaluation
├── config.yaml                 # Config file for parameters
├── requirements.txt            # requirements file
├── runs/                       # Folder for saved models and logs
└── README.md                   # This file
```

---

##  Installation

```bash
git clone https://github.com/AbhijitChallapalli/CSE-6389-PA1.git
```

Ensure the following dependencies are installed:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. **Training the Model**

```bash
python train.py --config config.yaml
```

Training outputs include:
- Epoch-wise loss values printed to console
- Saved best model weights in `runs/best_model.pt`
- Automatically determined classification threshold
- Plots of training curves and confusion matrix

### 2. **Testing the Model**

```bash
python test.py --config config.yaml
```

Outputs include:
- Test accuracy and confusion matrix (saved as `confusion_matrix.png`)
- Predicted probabilities for each MRI volume in `test_predictions.csv`
- Slice visualizations in `runs/slices/`

---

##  Model Architecture

| Layer | Type | Output Channels | Kernel | Operation |
|-------|------|-----------------|---------|------------|
| 1 | Conv3D + LeakyReLU | 32 | 3x3x3 | Feature extraction |
| 2 | MaxPool3D | – | 2x2x2 | Downsampling |
| 3 | Conv3D + LeakyReLU | 64 | 3x3x3 | Feature extraction |
| 4 | MaxPool3D | – | 2x2x2 | Downsampling |
| 5 | Conv3D + LeakyReLU | 128 | 3x3x3 | Deep spatial features |
| 6 | MaxPool3D | – | 2x2x2 | Downsampling |
| 7 | Conv3D + LeakyReLU | 192 | 3x3x3 | Rich volumetric representation |
| 8 | Global Avg Pool + Dropout | – | – | Regularization |
| 9 | Fully Connected | 2 | – | Classification (AD vs CN) |

---

##  Preprocessing Pipeline

- **Registration**: Align MRI volumes to **MNI152 template** (2mm isotropic resolution).  
- **Normalization**: Robust z-score normalization after percentile clipping (1–99%).  
- **Cropping/Padding**: Center crop or pad to `(128,128,128)` grid.  
- **Augmentations**:  
  - Random flips (x, y, z axes)  
  - Random small rotations (±7°)  
  - Gamma intensity jitter (±0.1)  
  - Additive Gaussian noise (σ=0.01)  

---

##  Training Configuration

| Parameter | Value |
|------------|--------|
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam (lr=0.001) |
| Epochs | 10 |
| Batch Size | 1–2 |
| Device | CPU / CUDA |
| Seed | Fixed for reproducibility |

---

## Evaluation & Results

| Metric | Value |
|---------|--------|
| Accuracy | **60.00%** |
| Confusion Matrix | [[2, 3], [1, 4]] |

Interpretation:

The model achieved 60% classification accuracy on the test dataset.

True Negatives (CN correctly identified): 2
 
False Positives (CN misclassified as AD): 3
 
False Negatives (AD misclassified as CN): 1
 
True Positives (AD correctly identified): 4

---

##  References

1. MNI152 Template, Montreal Neurological Institute (https://nist.mni.mcgill.ca/)  
2. Krizhevsky, A. et al., *ImageNet Classification with Deep Convolutional Neural Networks*, NeurIPS 2012.  
3. FSL and ANTs Documentation for MRI Registration Workflows  
4. Basaia et al., *Automated classification of Alzheimer’s disease and mild cognitive impairment using a single MRI and deep neural networks*, NeuroImage 2019.  
5. Wen et al., *Convolutional Neural Networks for Classification of Alzheimer’s Disease: Overview and Reproducible Evaluation*, Medical Image Analysis 2020.  
6. Suk et al., *Deep ensemble learning of sparse regression models for brain disease diagnosis*, MICCAI 2017.

---

##  Future Work
 
- Incorporation of **multi-modal MRI (T1, FLAIR, DTI)**.  
- Domain adaptation and cross-cohort evaluation using **ADNI** dataset.

---

##  License

This project is distributed under the **MIT License**. See `LICENSE` for more information.
