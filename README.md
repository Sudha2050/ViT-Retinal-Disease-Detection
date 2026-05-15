# 🩺 Retinal Disease Detection using Vision Transformer (ViT)

**A multi‑disease retinal diagnosis system** that classifies **Diabetic Retinopathy (DR)** from fundus photographs and **OCT retinal diseases** (CNV, DME, DRUSEN, NORMAL) from OCT scans. Built with a shared **ViT‑B/16** backbone and two specialised classification heads, trained with **3‑stage transfer learning** and deployed as a **FastAPI web application**.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📸 Demo

![Web App Demo](demo.gif)  
*Real‑time diagnosis: left – fundus DR grading, right – OCT disease classification*

---

## 🧠 Features

- **Dual‑head Vision Transformer (ViT‑B/16)** – single backbone, two classification heads for **DR (5 grades)** and **OCT (4 classes)**.
- **State‑of‑the‑art preprocessing** – CLAHE contrast enhancement, resizing to 224×224, normalisation.
- **Robust training pipeline** – 3‑stage fine‑tuning:
  1. Linear probing (head only)
  2. Partial unfreezing (last 4 transformer blocks)
  3. Joint fine‑tuning with **focal loss** to handle class imbalance.
- **Explainability** – Grad‑CAM heatmaps highlight regions the model focuses on.
- **Deterministic inference** – full reproducibility with fixed seeds and TTA (Test‑Time Augmentation).
- **Production‑ready API** – FastAPI backend with a clean, modern web interface.
- **Container‑ready** – simple deployment with Docker (optional).

---

## 📂 Datasets

| Dataset | Modality | Classes | Images | Source |
|---------|----------|---------|--------|--------|
| **APTOS 2019** | Fundus photos | No DR, Mild, Moderate, Severe, Proliferative DR | 3,662 | [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection) |
| **Kermany OCT** | OCT scans | CNV, DME, DRUSEN, NORMAL | 84,495 | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) |

Both datasets are split into 70/15/15 (train/val/test) for DR, and pre‑defined splits for OCT.

---

## 🏗️ Architecture

![Architecture](architecture.png)

- **Backbone**: `vit_base_patch16_224` from `timm`, pre‑trained on ImageNet‑21k.
- **Heads**: Two independent linear layers – one for DR (5 units) and one for OCT (4 units).
- **Input**: 224×224 RGB images (OCT converted to 3‑channel by stacking grayscale).
- **Parameters**: ~86M total.

---

## 📈 Training & Results

The model was trained in **3 phases** on an **NVIDIA T4 GPU** using mixed precision (FP16):

| Phase | Trainable Layers | Epochs | Optimizer | LR | DR Val Acc | OCT Val Acc |
|-------|------------------|--------|-----------|----|-------------|--------------|
| Head only | Classification head | 5 | AdamW | 1e-3 | 80.6% | – |
| Joint (last 4 blocks) | Backbone blocks + heads | 10 | AdamW | 1e-5 | 83.3% | 100% |
| **Focal‑loss joint** | Last 4 blocks + heads | 5 extra | AdamW | 1e-5 | **83.9%** | 100% |

**Final Test Performance (epoch 9):**

| Task | Accuracy | F1 Score (macro) | Notes |
|------|----------|------------------|-------|
| Diabetic Retinopathy (5 classes) | 83.9% | 0.68 | Focal loss improved recall on mild/severe classes |
| OCT Disease (4 classes) | 100% | 1.00 | Near‑perfect on test set |

**ROC Curves & Confusion Matrices** are available in the `notebooks/` folder.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/retinal-disease-detection.git
cd retinal-disease-detection
