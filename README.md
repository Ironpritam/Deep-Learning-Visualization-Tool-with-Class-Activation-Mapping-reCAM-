# Re-CAM: Deep Learning Visualization & Explainable AI (XAI) Suite

> A production-grade Explainable AI (XAI) and CNN model diagnostic toolkit developed at **IIT Ropar**. Features multi-algorithm visual explainers (**Vanilla CAM**, **Grad-CAM**, **Grad-CAM++**), multi-backbone vision neural networks (**ResNet-50**, **ResNet-18**, **VGG-16**, **MobileNet-V2**), automated Region of Interest (ROI) bounding box extraction, interactive Tkinter Desktop GUI, FastAPI REST microservice, and automated PDF audit report generation.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-Vision%20Models-FF6F00.svg)](https://pytorch.org/vision/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green.svg)](https://opencv.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Overview

Deep neural networks deployed in high-stakes domain applications (such as medical diagnostics, autonomous driving, and industrial quality inspection) suffer from a fundamental challenge: **model opacity** ("the black box problem"). Understanding *why* a deep convolutional network outputs a specific classification decision is critical for AI safety, fairness, and debugging.

Developed as a semester project during M.Tech studies at the **Indian Institute of Technology (IIT) Ropar**, **Re-CAM** addresses this challenge by projecting the internal feature activations of convolutional layers onto the raw input space. 

By analyzing the linear combination of final feature maps weighted by target class parameters or backpropagated gradients, **Re-CAM** pinpoints the exact pixel regions that drive model predictions, turning black-box networks into interpretable, audited AI assets.

---

## 🚀 Key Features & Capabilities

### 🎯 1. Multi-Algorithm XAI Suite
* **Vanilla Class Activation Mapping (CAM)**: Computes class-specific activation maps for architectures leveraging Global Average Pooling (GAP) layers.
* **Grad-CAM (Gradient-Weighted CAM)**: Uses backpropagated gradients of target class scores with respect to feature maps; compatible with **any** CNN architecture (VGG, ResNet, MobileNet).
* **Grad-CAM++**: Utilizes second-order gradients to provide fine-grained localization for multiple object instances in a single frame.

### 🔀 2. Multi-Model Vision Backbone Loader
* Instant switching across pre-trained computer vision models:
  * **ResNet-50** & **ResNet-18** (Deep Residual Networks)
  * **VGG-16** (Deep Convolutional Network)
  * **MobileNet-V2** (Efficient Edge-Optimized Architecture)

### 🔍 3. Automated Region of Interest (ROI) Bounding Box Extraction
* Calculates activation intensity contours (>70% peak intensity threshold).
* Automatically extracts and draws **Bounding Boxes** around localized high-activation regions, acting as a weakly-supervised object detector.

### 🏷️ 4. ImageNet 1,000 Class Inspector & Confidence Scorer
* Built-in dictionary mapping 1,000 ImageNet category indices to human-readable labels (e.g. `281: "tabby, tabby cat"`).
* Calculates Top-5 class probabilities and allows 1-click target class explanation switching.

### 🖥️ 5. Interactive Desktop GUI (`app_gui.py`)
* Real-time **Side-by-Side Canvas Visualizer** displaying `[Original Image] | [Raw Heatmap] | [Blended Overlay + ROI Bounding Box]`.
* Live sliders for **Alpha Transparency Blending** ($0.1 \rightarrow 1.0$) and **ROI Thresholding**.
* 6 Perceptual Colormaps (`JET`, `VIRIDIS`, `PLASMA`, `INFERNO`, `HOT`, `RAINBOW`).

### ⚡ 6. Enterprise REST API Microservice (`api_server.py`)
* FastAPI endpoint (`/api/v1/explain`) accepting image uploads and returning structured JSON predictions, bounding box coordinates, and Base64-encoded visual heatmaps.

### 📄 7. Automated PDF Diagnostic Report Generator (`src/reporting/`)
* Exports publication-ready PDF audit reports featuring execution metadata, Top-5 prediction tables, visual canvases, and detected ROI bounding box coordinates.

---

## 🔄 End-to-End System Pipeline

```text
                   Input Image / Tensor Feed
                               │
                               ▼
            PyTorch Vision Backbone Selection (ResNet / VGG / MobileNet)
                               │
                               ▼
                 Forward Pass & Top-5 Softmax Predictions
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   Target Class Selection              Feature Map & Gradient Hooks
   (Auto Top-1 or Custom)             (Conv Layer Activations A_k)
            │                                     │
            └──────────────────┬──────────────────┘
                               │
                               ▼
        XAI Engine Formulation (Vanilla CAM / Grad-CAM / Grad-CAM++)
                               │
                               ▼
             Bilinear Interpolation & OpenCV Colormap (JET/VIRIDIS)
                               │
                               ▼
          Alpha Blending (1-α Image + α Heatmap) & ROI Thresholding
                               │
                               ▼
             Automated ROI Contour Bounding Box Extraction
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
     Interactive GUI      REST API JSON      PDF Report
     (Tkinter Canvas)    (Base64 Payload)   Audit Export
```

---

## 📐 Mathematical Foundation

### 1. Vanilla Class Activation Mapping (CAM)
For networks with Global Average Pooling (GAP), the activation score for target class $c$ at spatial coordinate $(x, y)$ is given by:

$$S_c(x, y) = \text{ReLU}\left(\sum_{k} w_k^c A_k(x, y)\right)$$

Where $A_k(x, y)$ represents activation map $k$ of the final conv layer, and $w_k^c$ is the weight of class $c$ corresponding to channel $k$.

### 2. Grad-CAM (Gradient-Weighted CAM)
For arbitrary CNN architectures, the importance weight $\alpha_k^c$ of feature map $A_k$ for class score $Y^c$ is defined by global average pooling over gradients:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_k(i, j)}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A_k\right)$$

---

## 📋 Technology Stack

| Domain | Technology | Purpose |
|---|---|---|
| **Deep Learning** | PyTorch, Torchvision | Vision Backbones (ResNet, VGG, MobileNet) & Tensor Hooks |
| **Explainable AI** | Custom PyTorch Engine | Vanilla CAM, Grad-CAM, Grad-CAM++ Algorithms |
| **Computer Vision** | OpenCV, NumPy, Pillow | Colormaps, Alpha Blending, ROI Box Contour Detection |
| **User Interface** | Tkinter, PIL | Desktop GUI Visualization Dashboard |
| **Web API Service** | FastAPI, Uvicorn, Pydantic | Enterprise RESTful Microservice & JSON Endpoints |
| **Audit & Reporting** | ReportLab | Automated PDF Diagnostic Report Generation |

---

## 📁 Repository Structure

```text
Deep-Learning-Visualization-Tool-reCAM/
├── README.md                           # Main Portfolio Documentation
├── README_ANPR.md                      # Reference ANPR Project Documentation
├── main.py                             # Unified CLI / Application Launcher
├── app_gui.py                          # Modern Desktop GUI Application
├── api_server.py                       # FastAPI RESTful Microservice
├── requirements.txt                    # Unified Dependency Specification
│
├── src/                                # Core Engine Package
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── backbones.py                # Multi-Backbone PyTorch Loader (ResNet, VGG, MobileNet)
│   ├── xai/
│   │   ├── __init__.py
│   │   └── recam_engine.py             # Vanilla CAM, Grad-CAM, Grad-CAM++ Engine
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── imagenet_labels.py          # 1,000 ImageNet Class Lookup Dictionary
│   │   └── image_processing.py         # Transforms, Colormaps, Alpha Blending, ROI Extraction
│   └── reporting/
│       ├── __init__.py
│       └── report_generator.py         # PDF Diagnostic Audit Exporter
│
├── assets/                             # Benchmark Images & Visual Assets
│   └── sample.jpg
│
└── main-everything clubbed.py          # Legacy Single Script (Preserved for Academic Context)
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/PritamMahajan/Deep-Learning-Visualization-Tool-with-Class-Activation-Mapping-reCAM-.git
cd Deep-Learning-Visualization-Tool-with-Class-Activation-Mapping-reCAM-

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Interactive Desktop GUI

```bash
python main.py --gui
# Or directly:
python app_gui.py
```

### 3. Launch REST API Microservice

```bash
python main.py --api
# Or directly:
python api_server.py
```
Access interactive OpenAPI documentation at `http://localhost:8000/docs`.

### 4. CLI Batch Inference

Run heatmap explanation on target images via command line:

```bash
# Run Grad-CAM on sample image with ResNet-50
python main.py --image assets/sample.jpg --model resnet50 --method gradcam --output output_side_by_side.jpg

# Run Grad-CAM++ on specific class (e.g. 281 = Tabby Cat) using VGG16 backbone
python main.py --image assets/sample.jpg --model vgg16 --method gradcam_pp --target 281 --colormap VIRIDIS
```

### 5. REST API Usage Example

Send an image file to the REST API microservice:

```python
import requests

url = "http://localhost:8000/api/v1/explain"
files = {"file": open("assets/sample.jpg", "rb")}
data = {
    "model_name": "resnet50",
    "xai_method": "gradcam",
    "target_class": 281,
    "colormap": "JET",
    "alpha": 0.5
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Status: {result['status']}")
print(f"Target Class: {result['target_class_name']}")
print(f"Detected ROI Boxes: {len(result['roi_bounding_boxes'])}")
print(f"Canvas Base64 Payload Length: {len(result['canvas_base64'])}")
```

---

## 👤 Author & Academic Background

* **Pritam Sunil Mahajan**
  * **M.Tech in Artificial Intelligence** — *Indian Institute of Technology (IIT) Ropar*
  * **B.Tech in Computer Engineering** — *Ramrao Adik Institute of Technology, D.Y. Patil Deemed to be University*


---

## 📄 License

This repository is released under the [MIT License](LICENSE).
