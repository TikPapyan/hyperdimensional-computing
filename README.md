# Hyperdimensional Computing (HDC)
**A Unified Framework for Symbolic Learning Across Vision and Language Tasks**

This repository implements and compares a set of high-dimensional computing (HDC) techniques applied to three core recognition tasks:

- **Language Identification** (Europarl)
- **Digit Classification** (MNIST)
- **Face Recognition** (Olivetti)

The goal is to explore HDC as a computational paradigm that enables efficient, interpretable, and incremental learning across different data modalities. This work draws inspiration from recent advances in combining neural feature extraction with high-dimensional symbolic representations.

---

## Repository Structure

| Folder       | Description                                           |
|--------------|-------------------------------------------------------|
| `lang-rec/`  | Language classification using character trigrams + HDC |
| `digit-rec/` | Pixel-level and patch-based digit classification      |
| `face-rec/`  | Face recognition using OnlineHD and hybrid pipelines  |

Each subfolder contains an independent implementation, including data handling, model training, and evaluation routines tailored to that domain.

---

## Installation

To set up the environment, install the required dependencies:

```
pip install -r requirements.txt
```

## Results Summary

| Task                 | Dataset        | Method                    | Accuracy        | Encoding Time | Training Time     | Prediction Time  |
|----------------------|----------------|----------------------------|------------------|----------------|--------------------|------------------|
| Language ID          | Europarl       | Trigrams + HDC            | 98.35%           | 43.1s          | 0.16s              | 3.15s            |
| Digit Classification | MNIST          | Pixel-Level (HoloGN)      | 80.14%           | 36.5s          | 40.7s              | 7.5s             |
| Digit Classification | MNIST          | Patch-Based (LLM + POI)   | 97.24%           | 2520s          | 2027s              | 409s             |
| Digit Classification | MNIST (64×64)  | Patch-Based (LLM + POI)   | 95.31%           | 13974s         | 1125s              | 2272s            |
| Face Recognition     | Olivetti       | LLM + OnlineHD            | 91.25%           | 45.5s          | 2.33s              | 0.29s            |
| Face Recognition     | Olivetti       | Hybrid (CNN + HDC)        | 95.24% / 100.0%  | 3818s          | 18.95s / 7.98s     | 8.65s / 3.72s     |

## Language Identification

 - **Module**: lang-rec
 - **Dataset**: [European Parliament Proceedings Parallel Corpus](https://www.statmt.org/europarl/).
 - **Approach**: Character-level Trigrams + HDC

This module performs written language classification using character-level statistical patterns combined with HDC vector encoding.

### Pipeline Overview

1. Preprocessing:

- Cleans raw text and handles multilingual abbreviations
- Extracts character trigrams from cleaned strings

2. Hyperdimensional Encoding:

- Assigns a unique random vector to each trigram
- Constructs sentence vectors by summing trigram vectors
- Normalizes the resulting vector to produce a text hypervector

3. Training & Classification:

- Trains a prototype vector for each language via averaging
- Predicts the language of new texts using cosine similarity

## Digit Classification (MNIST)

 - **Module**: digit-rec
 - **Dataset**: [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

This module implements digit classification using Hyperdimensional Computing (HDC), comparing two distinct HDC frameworks drawn from recent research.

### Method 1: Associative Memory using Item Memory and Pixel-Wise Encoding

Based on: Manabat et al. (2019) "*Performance Analysis of Hyperdimensional Computing for Character Recognition*"

This approach uses simple pixel-based encoding and a majority-vote bundling strategy to generate digit hypervectors.

#### Pipeline Overview:

1. Binarization:

 - Input images are thresholded into binary form (0 or 1).

2. Pixel-Wise Encoding:

 - Each pixel is assigned a unique position hypervector and a binary value hypervector.
 - The overall image is encoded by binding value and position vectors, then bundling all pixel hypervectors.

3. Training:

 - Class hypervectors are formed by bundling encoded training examples of each digit class (0–9).

4. Classification:

 - For each test image, a query hypervector is generated.
 - The predicted class is determined by the nearest class hypervector using cosine similarity.

This method offers fast encoding and classification with minimal computational overhead, and serves as a baseline HDC system for binarized character inputs.

### Method 2: Patch-Based Encoding with Global Position Binding

Based on: Smets et al. (2024) - "*An Encoding Framework for Binarized Images Using Hyperdimensional Computing*"

This more advanced approach incorporates patch-level encoding, global position-aware binding, and an iterative prototype refinement loop, resulting in high recognition accuracy.

#### Key Components:

1. Patch-Level Processing:

 - Binarized input images are scanned for Points of Interest (POIs), i.e., foreground pixels with value 1.
 - For each POI, a 7×7 patch is extracted around it.

2. Local Patch Encoding:

 - Each pixel in the patch is encoded using:
    - A value hypervector (0 or 1),
    - A local position hypervector (within the patch),
    - Bound together using XOR and bundled across the patch.

3. Global Position Binding:

- The encoded patch vector is then bound with global position hypervectors based on its absolute location in the image.

4. Image-Level Representation:

- All patch vectors in the image are bundled using a majority rule to form a single image hypervector.

5. Classification via Iterative Prototype Learning:

- Class prototypes are initialized from labeled data.
- Misclassified examples iteratively update class prototypes in bipolar space to improve accuracy over time.
- Prediction is based on Hamming distance to learned class hypervectors.

This method is computationally more intensive but achieves competitive accuracy on the MNIST test set. It is particularly robust to spatial variability and sparse input patterns due to the position-aware design.

## Face Recognition (Olivetti Dataset)

 - **Module**: digit-rec
 - **Dataset**: [Olivetti](https://scikit-learn.org/0.19/datasets/olivetti_faces.html)

The `face-rec` directory implements and compares two Hyperdimensional Computing (HDC) pipelines for face recognition on the Olivetti dataset.

### 1. CNN+HDC Pipeline

#### Reference:
Yasser et al. (2022) "*An Efficient Hyperdimensional Computing Paradigm for Face Recognition*"

This approach uses a hybrid of deep learning and HDC, featuring:

 - InceptionV3: Pretrained CNN for extracting patch-level visual descriptors
 - KMeans: Clustering visual descriptors into symbolic visual "words"
 - Random Item Memory: Encodes spatial and symbolic information into binary hypervectors
 - Associative Memory: Classifies based on similarity in Hamming space
 - Incremental Learning: Supports addition of new identities without retraining from scratch

Evaluation consists of two phases:

 - Phase 1: Train/test on a fixed base set of subjects
 - Phase 2: Incrementally add and evaluate new subjects

### 2. LLM+POI-Inspired HDC Pipeline

#### Reference:
Smets et al. (2024) "*An encoding framework for binarized images using hyperdimensional computing*"

This second method applies the encoding framework proposed by Smets et al. to facial edge images derived from Canny filtering.

#### Key Steps:
 - Preprocessing:
     - Apply Canny edge detection to binarize grayscale face images
 - Local+Global Encoding:
     - Combine pixel-level patches with global location encoding
     - Use HDC operations (binding and bundling) to form image hypervectors

 - Online Learning:
    - An online OnlineHD classifier accumulates and adapts class hypervectors
    - Includes adaptive updates and convergence-checked iterative retraining

#### Notable Features:
 - No deep network or pretraining required
 - Fully symbolic and interpretable
 - Achieves robust recognition with a purely HDC-based system

### Future Work

This project focused primarily on evaluating classification accuracy. Several directions are open for future improvements:

 - **Parallelization**: Current implementations are mostly serial and CPU-based. Parallel or GPU versions could improve scalability.
 - **Fine-Grained Recognition**: Future studies could explore complex datasets with higher-resolution inputs and more subtle class distinctions.

### Citation

If you find this work useful, consider citing the internship report or referencing the associated papers.

### Author
**Tigran Papyan**

M1 Artificial Intelligence – French University in Armenia & Université Toulouse III – Paul Sabatier