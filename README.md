# 🌾 Rice Leaf Disease Prediction App

A deep-learning web application that helps farmers identify rice leaf diseases from a simple photo. Upload a picture of a sick rice leaf and get an instant diagnosis with confidence scores, a farmer-friendly explanation, and recommended actions.

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Training the Model](#-training-the-model)
- [Running the Web App](#-running-the-web-app)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)

## ✨ Features

- **Image Upload** — Drag-and-drop or browse to upload a rice leaf photo (JPG / PNG / BMP)
- **Disease Prediction** — Classifies into 3 diseases: Bacterial Leaf Blight, Brown Spot, Leaf Smut
- **Confidence Bars** — Visual confidence scores for every class
- **Farmer-Friendly Explanation** — Plain-language description of the detected disease
- **Recommended Action** — Actionable treatment and prevention advice
- **Beautiful UI** — Premium dark-green agricultural theme with glassmorphism effects

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| Model | EfficientNet-B0 (via `timm`) |
| Framework | PyTorch + torchvision |
| Web UI | Streamlit |
| Data | PlantVillage Rice Leaf Dataset (2 subsets) |

## 📁 Project Structure

```
smai A3/
├── Rice Leaf Dataset1/          # Small dataset (120 images)
│   ├── Bacterial Leaf Blight/
│   ├── Brown Spot/
│   └── Leaf smut/
├── Rice Leaf Dataset2/          # Large dataset (4,684 images)
│   ├── Bacterial Leaf Blight/
│   ├── Brown Spot/
│   └── Leaf smut/
├── train.py                     # Training script
├── app.py                       # Streamlit web app
├── disease_info.json            # Cached disease descriptions & actions
├── requirements.txt             # Python dependencies
├── rice_disease_model.pth       # Saved model weights (generated after training)
├── class_names.json             # Class label mapping (generated after training)
└── README.md                    # This file
```

## 🚀 Setup & Installation

### 1. Prerequisites

- Python 3.9+
- pip

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `torch`, `torchvision` — Deep learning framework
- `timm` — Pretrained image models (EfficientNet-B0)
- `streamlit` — Web app framework
- `Pillow` — Image loading
- `matplotlib` — Plotting
- `scikit-learn` — Train/val/test split & metrics
- `numpy` — Numerical operations

## 🏋️ Training the Model

Run the training script from the project directory:

```bash
python train.py
```

### Training Options

```bash
python train.py \
  --data1 "Rice Leaf Dataset1" \
  --data2 "Rice Leaf Dataset2" \
  --epochs 3 \
  --batch-size 32 \
  --lr 0.001 \
  --output-dir "."
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data1` | `Rice Leaf Dataset1` | Path to first dataset directory |
| `--data2` | `Rice Leaf Dataset2` | Path to second dataset directory |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `32` | Batch size for training |
| `--lr` | `0.001` | Learning rate |
| `--output-dir` | `.` | Where to save model and class names |

### Training Output

After training completes, two files are generated:

- **`rice_disease_model.pth`** — Saved model weights (best validation accuracy checkpoint)
- **`class_names.json`** — Ordered list of class labels

The script also prints:
- Per-epoch train/val loss and accuracy
- Test accuracy, classification report, and confusion matrix

## 🌐 Running the Web App

> **⚠️ Important:** You must train the model first (see above) so that `rice_disease_model.pth` and `class_names.json` exist.

```bash
streamlit run app.py
```

This launches the app at **http://localhost:8501**. Open it in your browser, upload a rice leaf photo, and get an instant diagnosis.

## 📊 Dataset

The app uses two combined rice leaf disease datasets:

| Dataset | Bacterial Leaf Blight | Brown Spot | Leaf Smut | Total |
|---------|----------------------|------------|-----------|-------|
| Rice Leaf Dataset1 | 40 | 40 | 40 | 120 |
| Rice Leaf Dataset2 | 1,604 | 1,620 | 1,460 | 4,684 |
| **Combined** | **1,644** | **1,660** | **1,500** | **4,804** |

The combined dataset is split **80/10/10** (train / validation / test) with stratification.

### Data Augmentation (Training Only)

- Random resized crop (224×224)
- Random horizontal & vertical flip
- Random rotation (±20°)
- Color jitter (brightness, contrast, saturation, hue)
- ImageNet normalization

## 🧠 Model Architecture

- **Base model:** EfficientNet-B0 (pretrained on ImageNet, via `timm`)
- **Transfer learning:** All layers fine-tuned, classifier head replaced for 3 classes
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** Cosine annealing
- **Loss:** Cross-entropy
- **Epochs:** 3 (with best-validation-accuracy checkpoint)

---

*Built for SMAI Assignment 3 — For educational purposes. Always verify diagnoses with an agricultural expert.*
