# 🧠 Hyperdimensional Computing (HDC)
**A Unified Framework for Symbolic Learning Across Vision and Language Tasks**

This repository implements and compares a set of high-dimensional computing (HDC) techniques applied to three core recognition tasks:

- **Language Identification** (Europarl)
- **Digit Classification** (MNIST)
- **Face Recognition** (Olivetti)

The goal is to explore HDC as a lightweight, interpretable, and flexible alternative to traditional deep learning—capable of both efficient learning and symbolic reasoning. This work combines classical HDC methods with recent advances such as patch-based encoding, global binding, and hybrid CNN-HDC models.

---

## Repository Structure

| Folder       | Description                                           |
|--------------|-------------------------------------------------------|
| `lang-rec/`  | Language classification using character trigrams + HDC |
| `digit-rec/` | Pixel-level and patch-based digit classification      |
| `face-rec/`  | Face recognition using OnlineHD and hybrid pipelines  |

Each folder contains independent implementations, preprocessing, training scripts, and evaluation logic.

---

## Installation

To install required dependencies:

```
pip install -r requirements.txt
```

## 📈 Results Summary

| Task                 | Dataset        | Method                    | Accuracy        | Encoding Time | Training Time     | Prediction Time  |
|----------------------|----------------|----------------------------|------------------|----------------|--------------------|------------------|
| Language ID          | Europarl       | Trigrams + HDC            | 98.35%           | 43.1s          | 0.16s              | 3.15s            |
| Digit Classification | MNIST          | Pixel-Level (HoloGN)      | 80.14%           | 36.5s          | 40.7s              | 7.5s             |
| Digit Classification | MNIST          | Patch-Based (LLM + POI)   | 97.24%           | 2520s          | 2027s              | 409s             |
| Digit Classification | MNIST (64×64)  | Patch-Based (LLM + POI)   | 95.31%           | 13974s         | 1125s              | 2272s            |
| Face Recognition     | Olivetti       | LLM + OnlineHD            | 91.25%           | 45.5s          | 2.33s              | 0.29s            |
| Face Recognition     | Olivetti       | Hybrid (CNN + HDC)        | 95.24% / 100.0%  | 3818s          | 18.95s / 7.98s     | 8.65s / 3.72s     |

## Tasks and Methods

### Language Identification (`lang-rec/`)

 - **Dataset**: [Europarl Parallel Corpus](https://www.statmt.org/europarl/)
 - **Features**: Character-level trigrams
 - **Encoding**: Random hypervectors per trigram
 - **Training**: Prototype vectors per language via averaging
 - **Prediction**: Cosine similarity

### Digit Classification (`digit-rec/`)

 - **Dataset**: [Europarl Parallel Corpus](https://www.statmt.org/europarl/)
 - **Features**: Character-level trigrams
 - **Encoding**: Random hypervectors per trigram
 - **Training**: Prototype vectors per language via averaging
 - **Prediction**: Cosine similarity

### Language Identification (`lang-rec/`)

#### Method 1: Pixel-Level Encoding (HoloGN)

 - Based on Manabat et al. (2019)
 - Fast but low-accuracy baseline
 - Uses direct binding of binary pixel values with positions

#### Method 2: Patch-Based Encoding with Global Binding

 - Based on Smets et al. (2024)
 - Detects Points of Interest → extracts patches → encodes local structure
 - Includes global spatial encoding and iterative class refinement

### Face Recognition (`face-rec/`)

#### Method 1: CNN + HDC Hybrid

 - Based on Yasser et al. (2022)
 - Uses InceptionV3 to extract CNN patch features
 - Clusters features via KMeans into symbolic "visual words"
 - Encodes images via symbolic patch bundling
 - Supports incremental learning

#### Method 2: Edge-Based + OnlineHD

 - Based on Smets et al. (2024)
 - Applies Canny edge detection → patch encoding with POI + OnlineHD classifier
 - Fully symbolic; no pretrained networks
 - Fast, interpretable, and compact

### Future Work

This project focused primarily on evaluating classification accuracy. Several directions are open for future improvements:

 - **Parallelization**: Current implementations are mostly serial and CPU-based. Parallel or GPU versions could improve scalability.
 - **Fine-Grained Recognition**: Future studies could explore complex datasets with higher-resolution inputs and more subtle class distinctions.

### Citation

If you find this work useful, consider citing the internship report or referencing the associated papers.

### Author
**Tigran Papyan**

M1 Artificial Intelligence – French University in Armenia & Université Toulouse III – Paul Sabatier