
#  EyEar – Sign-to-Text Translation with Transformer-based Edge AI

This repository presents the **Sign Language to Text** recognition module of **EyEar**, a real-time bidirectional wearable translator designed for the Deaf and Mute communities.  
Developed as part of my Final Year Project at HKUST.

---

##  My Role

- Developed the **sign-to-text translation module**
- Built and trained a **Transformer-based deep learning model** on MediaPipe hand keypoints
- Integrated the trained model into an **edge AI system** for real-time inference
- Designed experiments and performance evaluation

---

## Project Summary

| Component       | Description |
|------------------|-------------|
| Input            | `.npy` files of 30 frames per gesture, each frame 1662 keypoint values|
| Model            | Transformer Encoder with linear embedding + adaptive pooling |
| Output           | Word-level classification among 2,000 possible signs |
| Dataset          | Custom recorded sign gesture dataset using MediaPipe |
| Accuracy (Test)  | **85.59%** after early stopping |
| Epochs Trained   | 76 (early stopped) |

---

## Model Architecture

```python
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, input_size=1662, d_model=768, num_heads=12, num_layers=6, num_classes=2000):
        ...
```

- Transformer Encoder (6 layers, 12 heads)
- Positional features embedded via linear layer
- Adaptive Average Pooling
- Dropout + FC + Softmax for classification

---

## Training Details

| Setting         | Value       |
|------------------|-------------|
| Optimizer        | AdamW       |
| Scheduler        | CosineAnnealingWarmRestarts |
| Loss Function    | CrossEntropy with class weights |
| Gradient Clip    | Max norm = 1.0 |
| Early Stop       | Patience = 20 |
| Batch Size       | dynamic (Colab based) |

### Final Results

- **Val Accuracy**: 85.57% @ Epoch 76  
- **Test Accuracy**: **85.59%**, **Test Loss**: 0.3176

---

## How to Train

```bash
Open sign_to_text_model_training.ipynb

# Preprocess .npy files and load to train_loader, val_loader, test_loader
# Train and evaluate using PyTorch
```

The best model will be saved as `./best_model.pt`.

---

## Awards & Recognition

- **2nd Runner-up, Best Final Year Project** (HKUST ECE Department)
- **Best Poster Award** at ECE industrial day 2025

---

## Author

Lee Gyumin – HKUST ECE  
Final Year Project 2024-25  
Email: gleeag@connect.ust.hk

---

## Dataset Citation

This project utilizes the **WLASL (Word-Level American Sign Language)** dataset, introduced in:

> Dongxu Li, Cristian Rodriguez Opazo, Xin Yu, and Hongdong Li,  
> **"Word‑level Deep Sign Language Recognition from Video:  
> A New Large‑scale Dataset and Methods Comparison,"**  
> *arXiv:1910.11006 (2020)* :contentReference[oaicite:1]{index=1}

- Dataset from GitHub: [dxli94/WLASL](https://github.com/dxli94/WLASL)
- License: Provided under the dataset's original licensing terms

Please cite the above when using WLASL for academic research.
