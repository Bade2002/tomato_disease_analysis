```markdown
# 🍅 Tomato Leaf Disease Classification & Diagnostic System

[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1%2BCU121-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/NVIDIA%20CUDA-12.1-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end Computer Vision project and diagnostic application to classify tomato leaf conditions into **10 distinct categories** (9 diseases + healthy). Accelerated using **NVIDIA CUDA (Ampere Architecture)** with Automatic Mixed Precision (AMP), featuring both high-accuracy **Transfer Learning** and a **Custom CNN designed from scratch**, connected to a native desktop GUI.

---

## 📌 Key Highlights

- **Dual-Model Architecture**:
  - **MobileNetV3-Large**: Transfer learning with Cosine Annealing learning rate schedule achieving **99.91% validation accuracy**.
  - **Custom 4-Block CNN**: Built from scratch using PyTorch primitives with Global Average Pooling (~520k parameters) achieving **~95% validation accuracy**.
- **Hardware-Accelerated Pipeline**: Uses `torch.amp` (FP16 mixed precision) with Tensor Core acceleration on RTX GPUs, cutting per-epoch training to ~15–20 seconds.
- **Interactive Multi-Model Desktop GUI**: Built using Tkinter and PIL—runs 100% offline with live image previews, dynamic model toggling, top-3 confidence meters, and hardware detection badges.
- **Robust Augmentation Strategy**: Photometric jitter, random affine transformations, rotations, flips, and random erasing to prevent background/lighting overfitting.

---

## 📂 Project Directory Structure

```text
tomato-leaf-disease-detection/
├── dataset_split/                               # Dataset Root (PlantVillage Split)
│   ├── train/                                   # 12,808 Training Images (80%)
│   │   ├── Tomato___Bacterial_spot/
│   │   ├── Tomato___Early_blight/
│   │   ├── Tomato___Late_blight/
│   │   ├── Tomato___Leaf_Mold/
│   │   ├── Tomato___Septoria_leaf_spot/
│   │   ├── Tomato___Spider_mites_Two_spotted_spider_mite/
│   │   ├── Tomato___Target_Spot/
│   │   ├── Tomato___Tomato_YellowLeaf__Curl_Virus/
│   │   ├── Tomato___Tomato_mosaic_virus/
│   │   └── Tomato___healthy/
│   │
│   └── val/                                     # 3,203 Validation Images (20%)
│       ├── Tomato___Bacterial_spot/
│       ├── Tomato___Early_blight/
│       └── ... (matching 10 class folders)
│
├── check_dataset.py                             # Dataset health & corruption verification
├── train_advanced.py                            # MobileNetV3 transfer learning training pipeline
├── train_custom_cnn.py                          # Custom CNN from-scratch training pipeline
├── predict.py                                   # CLI single-image inference script
├── app_gui_multimodel.py                        # Standalone multi-model desktop GUI application
│
├── best_tomato_model.pth                        # Trained weights (MobileNetV3)
├── custom_tomato_cnn_best.pth                   # Trained weights (Custom CNN)
├── class_labels.json                            # Class label mapping file
├── requirements.txt                             # Python dependencies
└── README.md                                    # Project documentation

```

---

## 🔬 Supported Disease Classes (10 Total)

| Index | Class Name | Pathogen Type / Condition |
| --- | --- | --- |
| **0** | `Tomato_Bacterial_spot` | Bacterial (*Xanthomonas*) |
| **1** | `Tomato_Early_blight` | Fungal (*Alternaria solani*) |
| **2** | `Tomato_Late_blight` | Oomycete (*Phytophthora infestans*) |
| **3** | `Tomato_Leaf_Mold` | Fungal (*Passalora fulva*) |
| **4** | `Tomato_Septoria_leaf_spot` | Fungal (*Septoria lycopersici*) |
| **5** | `Tomato_Spider_mites_Two_spotted_spider_mite` | Pest (*Tetranychus urticae*) |
| **6** | `Tomato_Target_Spot` | Fungal (*Corynespora cassiicola*) |
| **7** | `Tomato_Tomato_YellowLeaf_Curl_Virus` | Viral (Begomovirus) |
| **8** | `Tomato_Tomato_mosaic_virus` | Viral (Tobamovirus) |
| **9** | `Tomato_healthy` | Healthy Leaf Tissue |

---

## 📊 Benchmark & Performance Comparison

| Metric / Attribute | MobileNetV3-Large (Transfer Learning) | Custom CNN (From Scratch) |
| --- | --- | --- |
| **Base Architecture** | MobileNetV3-Large (ImageNet) | 4-Stage Conv-BN-ReLU-Pool |
| **Trainable Parameters** | ~4.2 Million | ~520,000 |
| **Training Time (15 Epochs on RTX 3070 Ti)** | ~4–5 minutes | ~3 minutes |
| **Input Resolution** | $224 \times 224 \times 3$ | $224 \times 224 \times 3$ |
| **Optimization** | AdamW ($lr=3\times10^{-4}$) + Cosine Annealing | Adam ($lr=10^{-3}$) + ReduceLROnPlateau |
| **Regularization** | Label Smoothing ($0.1$) + Dropout ($0.2$) | Dropout ($0.4, 0.2$) + BatchNorm |
| **Validation Accuracy** | **99.91%** | **~95.50%** |
| **GPU Inference Latency** | $< 8 \text{ ms}$ / image | $< 3 \text{ ms}$ / image |

---

## 🛠️ Installation & Setup

### 1. Prerequisites

* **Python 3.10+ / 3.12 (64-bit)**
* **NVIDIA GPU** with CUDA support (e.g., RTX 30/40 series) or CPU fallback

### 2. Environment Setup

Clone the repository and install dependencies:

```bash
git clone [https://github.com/your-username/tomato-leaf-disease-detection.git](https://github.com/your-username/tomato-leaf-disease-detection.git)
cd tomato-leaf-disease-detection

```

Install GPU-accelerated PyTorch:

```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

```

Install supporting packages:

```bash
pip install pillow numpy matplotlib scikit-learn tqdm

```

### 3. Verify Hardware Detection

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"

```

---

## 🚀 How to Run

### Step 1: Verify Dataset

Check image integrity and distribution across splits:

```bash
python check_dataset.py

```

### Step 2: Train the Models

To train **MobileNetV3-Large** (Transfer Learning):

```bash
python train_advanced.py

```

To train the **Custom CNN** from scratch:

```bash
python train_custom_cnn.py

```

### Step 3: Run Single Image CLI Prediction

Run inference directly on a target image:

```bash
python predict.py "dataset_split/val/Tomato_Early_blight/sample.jpg"

```

### Step 4: Launch the Multi-Model Desktop Application

Run the native desktop GUI to upload images and toggle between models in real time:

```bash
python app_gui_multimodel.py

```

---

## 🧠 Architectural Insights

### Custom CNN Architecture Pipeline

```text
Input (3 x 224 x 224)
   │
   ├── Block 1: [Conv2d(32) -> BatchNorm -> ReLU] x2 -> MaxPool(2x2)  ==> (32, 112, 112)
   ├── Block 2: [Conv2d(64) -> BatchNorm -> ReLU] x2 -> MaxPool(2x2)  ==> (64, 56, 56)
   ├── Block 3: [Conv2d(128) -> BatchNorm -> ReLU]   -> MaxPool(2x2)  ==> (128, 28, 28)
   ├── Block 4: [Conv2d(256) -> BatchNorm -> ReLU]   -> MaxPool(2x2)  ==> (256, 14, 14)
   │
   ├── Global Average Pooling: AdaptiveAvgPool2d((1, 1))                ==> (256, 1, 1)
   └── Classification Head:    Flatten -> Dropout(0.4) -> Linear(128) -> ReLU -> Dropout(0.2) -> Linear(10)

```

* **Batch Normalization**: Placed after every convolution to reduce internal covariate shift and stabilize training.
* **Global Average Pooling**: Squeezes spatial feature maps into a 1D vector, eliminating giant parameter matrices from standard flat linear heads.
* **Mixed Precision (`torch.amp.GradScaler`)**: Scales gradients to prevent numeric underflow during FP16 forward and backward passes.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

```

```
