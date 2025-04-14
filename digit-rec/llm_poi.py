"""
Hyperdimensional Computing (HDC) Image Classification Framework
Based on: Smets et al. (2024) "An encoding framework for binarized images using hyperdimensional computing"
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from tqdm import tqdm

np.random.seed(42)

d = 10000
patch_size = 7
S_global = 4
max_iter = 1000
check_interval = 100
accuracy_threshold = 0.99

def random_hv(dim):
    # dense binary HVs...elements are 0 or 1 with equal probability" (Section 2.1)
    return np.random.randint(0, 2, dim).astype(np.int32)

def bind(a, b):
    # binding ⊗: H × H → H: (v₁, v₂) → v₁ XOR v₂" (Section 2.1)
    return np.logical_xor(a, b).astype(np.int32)

def bundle(hv_list):
    # v[d] = [B[d]] = {1 if B[d] > n/2, 0 otherwise}" (Section 2.1)
    if len(hv_list) == 0:
        return np.zeros(d, dtype=np.int32)
    hv_sum = np.sum(np.array(hv_list), axis=0)
    threshold = len(hv_list) / 2.0
    return (hv_sum >= threshold).astype(np.int32)

def bin2bipolar(hv):
    return np.where(hv == 0, -1, 1)

def hamming_distance(hv1, hv2):
    return np.mean(hv1 != hv2)

def generate_local_linear_hvs(num_positions, d, S):
    # for values ranging from -100 to 100 with 21 levels four splits create edge vectors at -100, -50, 0, 50, 100 (Figure 4)
    pos_hvs = {}
    splits = np.linspace(0, num_positions - 1, S + 1, dtype=int)
    
    for s in range(S):
        start, end = splits[s], splits[s + 1]
        # Create edge vectors with D/2 bit difference (Section 2.2.3)
        edge0 = random_hv(d)
        edge1 = edge0.copy()
        flip_indices = np.random.choice(d, size=d//2, replace=False)
        edge1[flip_indices] = 1 - edge1[flip_indices]
        
        # Linear interpolation within split
        for pos in range(start, end + 1):
            alpha = (pos - start) / max((end - start), 1)
            # linear mapping is applied between edge vectors (Section 2.2.3)
            combined = (1 - alpha)*edge0 + alpha*edge1
            pos_hvs[pos] = (combined >= 0.5).astype(np.int32)  # Majority rule
    for pos in range(num_positions):
        if pos not in pos_hvs:
            pos_hvs[pos] = random_hv(d)
    return pos_hvs

# Initialize global position hypervectors (Section 2.3.2.4)
global_x_hvs = generate_local_linear_hvs(28, d, S_global)  # CIM_{x,w} in paper
global_y_hvs = generate_local_linear_hvs(28, d, S_global)  # CIM_{y,h} in paper

def generate_patch_pos_hvs(patch_size, d):
    # x and y position HVs are stored in two separate CIMs...mapped with orthogonal mapping (Section 2.3.2.3)
    return {pos: random_hv(d) for pos in range(patch_size)}

patch_x_hvs = generate_patch_pos_hvs(patch_size, d)  # CIM_{x,z} in paper
patch_y_hvs = generate_patch_pos_hvs(patch_size, d)  # CIM_{y,z} in paper

# Value Hypervectors (Section 2.3.2.3)
value_hvs = {
    0: random_hv(d),  # IM for value 0
    1: random_hv(d)   # IM for value 1
}

def encode_image(image, patch_size=patch_size):
    h, w = image.shape
    offset = patch_size // 2
    patch_vectors = []
    
    # POI Selection (Section 2.3.2.2)
    points = np.argwhere(image == 1)

    for (x, y) in points:
        # Handle edges (not specified in the paper)
        if x - offset < 0 or x + offset >= h or y - offset < 0 or y + offset >= w:
            continue

        # Extract patch (Section 2.3.2.2)
        patch = image[x - offset: x + offset + 1, y - offset: y + offset + 1]

        # Patch Vector Encoding (Equation 9)
        local_pixel_hvs = []
        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                # Paper: "v_pixel = v_value ⊗ v_x ⊗ v_y" (Section 2.3.2.3)
                pixel_val = patch[i, j]
                hv_val = value_hvs[pixel_val]
                hv_px = patch_x_hvs[i]
                hv_py = patch_y_hvs[j]
                local_pos = bind(hv_px, hv_py)
                hv_local = bind(hv_val, local_pos)
                local_pixel_hvs.append(hv_local)

        # Bundle patch pixels with majority rule (Equation 3)
        patch_hv = bundle(local_pixel_hvs)

        # 4. Bind with Global Position (Equation 10)
        hv_global_x = global_x_hvs[x]
        hv_global_y = global_y_hvs[y]
        patch_with_global = bind(bind(patch_hv, hv_global_x), hv_global_y)
        patch_vectors.append(patch_with_global)

    if len(patch_vectors) == 0:
        return np.zeros(d, dtype=np.int32)

    return bundle(patch_vectors)

class HDClassifier:
    def __init__(self, d):
        self.d = d
        # Bipolar accumulators for prototype refinement
        self.accumulators = {cls: np.zeros(d, dtype=np.int32) for cls in range(10)}
    
    def initialize(self, hv_list, labels):
        # bundling all sample HVs belonging to the same class
        for hv, lbl in zip(hv_list, labels):
            bhv = bin2bipolar(hv)
            self.accumulators[lbl] += bhv
    
    def get_prototypes(self):
        # Binarize accumulators (Equation 3)
        prototypes = {}
        for cls, acc in self.accumulators.items():
            prototypes[cls] = (acc >= 0).astype(np.int32)
        return prototypes
    
    def predict(self, hv, prototypes=None):
        # Similarity-based prediction (Equation 1)
        if prototypes is None:
            prototypes = self.get_prototypes()
        distances = {cls: hamming_distance(hv, prototypes[cls]) for cls in prototypes}
        return min(distances, key=distances.get)
    
    def update(self, hv, true_label, pred_label):
        # updating these class bundles using misclassified samples
        bhv = bin2bipolar(hv)
        self.accumulators[true_label] += bhv
        self.accumulators[pred_label] -= bhv

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
X = X.reshape(-1, 28, 28)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
print(f"Loaded {len(X_train)} training and {len(X_test)} test images.")

# Binarization (Equation 8) (threshold T_bin = 0)
X_train_bin = (X_train > 0).astype(np.int32)
X_test_bin  = (X_test > 0).astype(np.int32)

print("Encoding training images...")
encoded_train = Parallel(n_jobs=-1)(
    delayed(encode_image)(img, patch_size=patch_size) 
    for img in tqdm(X_train_bin, desc="Encoding Training")
)

hd_classifier = HDClassifier(d)
hd_classifier.initialize(encoded_train, y_train)

print("Training...")
best_train_acc = 0.0
best_prototypes = None
for iteration in range(1, max_iter + 1):
    prototypes = hd_classifier.get_prototypes()
    correct = 0
    for hv, true_label in zip(encoded_train, y_train):
        pred_label = hd_classifier.predict(hv, prototypes)
        if pred_label == true_label:
            correct += 1
        else:
            hd_classifier.update(hv, true_label, pred_label)
    train_acc = correct / len(y_train)
    if train_acc > best_train_acc:
        best_train_acc = train_acc
        best_prototypes = hd_classifier.get_prototypes()
    if iteration % check_interval == 0:
        print(f"Iteration {iteration}: Training accuracy = {train_acc*100:.2f}%")
        if train_acc >= accuracy_threshold:
            print("Training accuracy threshold reached. Stopping training.")
            break

print("Encoding test images...")
encoded_test = Parallel(n_jobs=-1)(
    delayed(encode_image)(img, patch_size=patch_size)
    for img in tqdm(X_test_bin, desc="Encoding Test")
)

print("Evaluating...")
test_preds = []
for hv in encoded_test:
    pred = hd_classifier.predict(hv, best_prototypes)
    test_preds.append(pred)
test_acc = accuracy_score(y_test, test_preds)
print(f"Test Accuracy: {test_acc*100:.2f}%")