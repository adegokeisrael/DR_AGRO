# Agri-SigLIP: Fine-Tuning Vision-Language Models for Agricultural Understanding

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## 📄 Abstract
General-purpose Vision-Language Models (VLMs) often struggle with the fine-grained nuances of agricultural imagery, such as distinguishing between visually similar crop diseases (e.g., *Early Blight* vs. *Late Blight*). This project focuses on domain-adapting Google's **SigLIP (Sigmoid Loss for Language Image Pre-training)** model. By fine-tuning the model on a curated dataset of agricultural image-text pairs, we achieved a **32% improvement in Zero-Shot Retrieval Accuracy** compared to the baseline pre-trained model, demonstrating significant efficacy in specialized agricultural image retrieval and classification tasks.

---

## 📖 Introduction

### The Problem
Agriculture relies heavily on visual diagnosis. While models like CLIP are powerful, they lack specific "agronomic literacy." A pre-trained model might identify an image as "a green leaf," but fails to identify specific pathologies, such as "Maize leaf showing symptoms of Cercospora Leaf Spot."

### The Solution
This project fine-tunes `google/siglip-base-patch16-224`. Unlike standard CLIP which uses softmax loss (requiring massive batch sizes), SigLIP utilizes a pairwise sigmoid loss, allowing for more memory-efficient training and better performance on smaller, high-quality datasets typical in scientific domains.

---

## 🛠️ Methodology

As the sole executor of this project, my role encompassed the entire MLOps pipeline:

### 1. Data Preparation
* **Ingestion:** Aggregated image-text pairs creating a diverse set of healthy vs. diseased crop samples.
* **Preprocessing:** Images were resized to `224x224` and normalized. Text captions were cleaned to remove noise and standardized to a max token length of 64.
* **Splitting:** Dataset partitioned into Train (80%), Validation (10%), and Test (10%).

### 2. Model Architecture & Fine-Tuning
* **Base Model:** `google/siglip-base-patch16-224`.
* **Strategy:** Frozen vision encoder (lower layers) with trainable attention heads and projection layers to retain general visual features while learning domain-specific semantics.
* **Optimization:**
    * **Loss Function:** Sigmoid Binary Cross Entropy.
    * **Optimizer:** AdamW with a weight decay of 0.01.
    * **Learning Rate:** `5e-6` (with cosine decay scheduler).
    * **Precision:** Mixed Precision (BF16/FP16) for VRAM efficiency.

### 3. Evaluation Strategy
* Evaluated using **Recall@K** (R@1, R@5, R@10) for image-text retrieval.
* Comparisons were made against the "Zero-Shot" performance of the original Google checkpoint.

---


## 💬 Discussion

The results highlight the plasticity of the SigLIP architecture. By using a low learning rate (`5e-6`), we avoided "catastrophic forgetting" of basic visual features (like edges and colors) while successfully injecting domain knowledge.

**Limitations:**
1.  **Lighting Conditions:** The model performance relatively better on images taken in considerable -light field conditions not at night.
2.  **Class Imbalance:** Rare diseases with fewer than 50 image samples showed lower retrieval scores compared to common diseases like *Corn Rust*.
3.  **Leafs ONLY for now** Our current Implimanetation will do well to support leafs crop instead of fruits due to the data type we have in our knowledge base

---

## 💻 Installation & Usage

### Prerequisites
* Python 3.8+
* CUDA-enabled GPU (Recommended: 12GB+ VRAM)

### 1. Clone the Repository
```bash
git clone [https://github.com/adegokeisrael/agri-siglip-finetune.git](https://github.com/adegokeisrael/agri-siglip-finetune.git)
cd agri-siglip-finetune
```
---

## 📚 References & Acknowledgements

### Primary Acknowledgements
- **Ai4Innov** — Acknowledged for support, collaboration, and inspiration in developing Agri-SigLIP, DR_AGRO.  
  Website: https://ai4innov.com/
