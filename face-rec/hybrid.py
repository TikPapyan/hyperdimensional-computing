"""
Hyperdimensional Computing (HDC) Image Classification Framework
Based on: Yasser et al. (2022) "An Efficient Hyperdimensional Computing Paradigm for Face Recognition"
"""

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchvision import models, transforms
from tqdm import tqdm
from torchvision.models import Inception_V3_Weights

# ——— PARAMETERS ———
weights = Inception_V3_Weights.IMAGENET1K_V1  # Pretrained weights for InceptionV3
D = 10000                                     # Dimensionality of hypervectors
n_clusters = 1000                             # Number of patch-level visual words (KMeans clusters)
phase1_ratio = 0.7                            # Ratio of subjects used for Phase-1 (base classes)
img_train_ratio = 0.75                        # Percentage of subject images used for training

# ——— DATA LOADING ———
# Fetch grayscale facial images (64x64) and subject labels (40 individuals, 10 images each)
data = fetch_olivetti_faces()
images, targets = data.images, data.target
subjects = np.unique(targets)

# Split subjects into Phase-1 (base training) and Phase-2 (incremental generalization)
phase1_subjs, phase2_subjs = train_test_split(subjects, train_size=phase1_ratio, random_state=0)

# ——— INCEPTIONV3 FEATURE EXTRACTOR SETUP ———
# Load pretrained InceptionV3 model and attach a hook to capture intermediate CNN patch features
model = models.inception_v3(weights=weights, aux_logits=True)
model.eval()
_feature_buffer = []  # Buffer to store extracted CNN features per image

def _hook(module, inp, out):
    _feature_buffer.append(out.detach().cpu())

# Tap into the output of "Mixed_5c" layer, suitable for dense spatial features
model.Mixed_5c.register_forward_hook(_hook)

# Image preprocessing: resize -> convert to RGB -> normalize
_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# Function to extract CNN patch-level features from an image
def extract_cnn_patches(img):
    _feature_buffer.clear()
    x = _preprocess(img).unsqueeze(0)
    with torch.no_grad():
        model(x)
    feat = _feature_buffer[0][0]  # Shape: (C, H, W)
    return feat.reshape(feat.shape[0], -1).T  # Reshape to (H*W, C)

# ——— CNN PATCH CLUSTERING (KMeans for Visual Words) ———
# Extract patch features from all images, flatten, and cluster into K visual words
all_patches = []
for img in tqdm(images, desc="Extracting CNN patches"):
    patches = extract_cnn_patches(img).numpy().astype(np.float64)
    all_patches.append(patches)
all_patches = np.vstack(all_patches)
print(f"Total extracted patches: {all_patches.shape}")

kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
kmeans.fit(all_patches.astype(np.float64))  # Learn centroids of visual words

# ——— ITEM MEMORY (IM) ———
# Assign a random bipolar hypervector to each of the K visual words
IM = np.random.choice([-1, 1], size=(n_clusters, D))

# ——— PHASE 1 TRAINING (BASE SUBJECTS) ———
# Accumulator Memory (AM): stores class hypervectors for each known subject
AM = {s: np.zeros(D, dtype=int) for s in phase1_subjs}

# Encode each training image of base subjects into a hypervector and accumulate in AM
for subj in tqdm(phase1_subjs, desc="Phase-1 training"):
    subj_imgs = images[targets == subj]
    train_imgs, test_imgs = train_test_split(subj_imgs, train_size=img_train_ratio, random_state=0)

    for img in train_imgs:
        patches = extract_cnn_patches(img)
        words = kmeans.predict(patches.numpy().astype(np.float64))  # Visual word indices
        AM[subj] += IM[words].sum(axis=0)  # Accumulate patch hypervectors for subject

# Binarize subject hypervectors
for subj in AM:
    AM[subj] = np.where(AM[subj] > 0, 1, -1)

# ——— PHASE 1 TESTING ———
# Evaluate classifier using the base (phase-1) subjects only
correct = total = 0
for subj in tqdm(phase1_subjs, desc="Phase-1 testing"):
    subj_imgs = images[targets == subj]
    train_imgs, test_imgs = train_test_split(subj_imgs, train_size=img_train_ratio, random_state=0)

    # Ensure test images are unseen
    train_hashes = {hash(img.tobytes()) for img in train_imgs}
    test_hashes = {hash(img.tobytes()) for img in test_imgs}
    assert train_hashes.isdisjoint(test_hashes), f"❌ Data leakage for subject {subj}!"

    for img in test_imgs:
        patches = extract_cnn_patches(img)
        words = kmeans.predict(patches.numpy().astype(np.float64))
        test_vec = np.where(IM[words].sum(axis=0) > 0, 1, -1)
        sims = {s: cosine_similarity(test_vec.reshape(1, -1), AM[s].reshape(1, -1))[0, 0]
                for s in AM}
        pred = max(sims, key=sims.get)
        correct += (pred == subj)
        total += 1

print(f"\nPhase-1 accuracy: {correct/total*100:.2f}%")

# ——— PHASE 2 TRAINING (INCREMENTAL SUBJECTS) ———
# New subjects added incrementally by encoding their training images
for subj in tqdm(phase2_subjs, desc="Phase-2 training"):
    AM[subj] = np.zeros(D, dtype=int)
    subj_imgs = images[targets == subj]
    train_imgs, test_imgs = train_test_split(subj_imgs, train_size=img_train_ratio, random_state=0)

    for img in train_imgs:
        patches = extract_cnn_patches(img)
        words = kmeans.predict(patches.numpy().astype(np.float64))
        AM[subj] += IM[words].sum(axis=0)

# Binarize hypervectors for new subjects
for subj in phase2_subjs:
    AM[subj] = np.where(AM[subj] > 0, 1, -1)

# ——— PHASE 2 TESTING ———
# Evaluate all new subjects against the full AM (including base subjects)
correct = total = 0
for subj in tqdm(phase2_subjs, desc="Phase-2 testing"):
    subj_imgs = images[targets == subj]
    train_imgs, test_imgs = train_test_split(subj_imgs, train_size=img_train_ratio, random_state=0)

    for img in test_imgs:
        patches = extract_cnn_patches(img)
        words = kmeans.predict(patches.numpy().astype(np.float64))
        test_vec = np.where(IM[words].sum(axis=0) > 0, 1, -1)
        # Use entire AM for classification (supports incremental learning)
        sims = {s: cosine_similarity(test_vec.reshape(1, -1), AM[s].reshape(1, -1))[0, 0]
                for s in AM}
        pred = max(sims, key=sims.get)
        correct += (pred == subj)
        total += 1

print(f"Phase-2 accuracy: {correct/total*100:.2f}%")