"""
Hyperdimensional Computing (HDC) Image Classification Framework
Based on: Smets et al. (2024) "An encoding framework for binarized images using hyperdimensional computing"
"""

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from tqdm import tqdm
from skimage.feature import canny

d = 10000
patch_size = 7
S_global = 4
learning_rate = 1.0
max_iter = 5
tol = 1e-6

faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X = faces.images
y = faces.target
n_classes = len(faces.target)

X_edges = Parallel(n_jobs=-1)(delayed(canny)(img, sigma=1.0) for img in X)
X_bin = np.array(X_edges, dtype=np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    X_bin, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Dataset: {len(X_train)} train, {len(X_test)} test, {n_classes} classes")

np.random.seed(42)

def random_hv(dim):
    return np.random.randint(0,2,dim,dtype=np.int32)

def bind(a,b):
    return np.bitwise_xor(a,b)

def bundle(hvs):
    if not hvs:
        return np.zeros(d, dtype=np.int32)
    s = np.sum(hvs, axis=0)
    return (s >= (len(hvs)/2)).astype(np.int32)

def bin2bipolar(hv):
    return np.where(hv==0, -1, 1)

def hamming(a,b):
    return np.mean(a!=b)

def gen_local_cim(npos, dim, splits):
    splits_idx = np.linspace(0, npos-1, splits+1, dtype=int)
    cim = {}
    for s in range(splits):
        e0 = random_hv(dim)
        e1 = e0.copy()
        mask = np.random.choice(dim, dim//2, replace=False)
        e1[mask] ^= 1
        start, end = splits_idx[s], splits_idx[s+1]
        for p in range(start, end+1):
            alpha = (p-start)/max(end-start,1)
            mix = (1-alpha)*e0 + alpha*e1
            cim[p] = (mix>=0.5).astype(np.int32)
    for p in range(npos):
        cim.setdefault(p, random_hv(dim))
    return cim

h, w = X_train.shape[1], X_train.shape[2]
global_x = gen_local_cim(h, d, S_global)
global_y = gen_local_cim(w, d, S_global)

patch_x = {i: random_hv(d) for i in range(patch_size)}
patch_y = {j: random_hv(d) for j in range(patch_size)}
value_hvs = {0: random_hv(d), 1: random_hv(d)}
off = patch_size//2

def encode(img):
    pts = np.argwhere(img==1)
    vecs = []
    for x,y in pts:
        if x-off<0 or x+off>=h or y-off<0 or y+off>=w:
            continue
        patch = img[x-off:x+off+1, y-off:y+off+1]
        pix = []
        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                hv = bind(value_hvs[patch[i,j]], bind(patch_x[i], patch_y[j]))
                pix.append(hv)
        p_hv = bundle(pix)
        vecs.append(bind(p_hv, bind(global_x[x], global_y[y])))
    return bundle(vecs)

class OnlineHD:
    def __init__(self, dim, n_cls, lr=1.0):
        self.d = dim
        self.lr = lr
        self.acc = {c: np.zeros(dim, dtype=np.float32) for c in range(n_cls)}

    def initial_train(self, H_list, labels):
        for H, lbl in zip(H_list, labels):
            self.acc[lbl] += bin2bipolar(H)

    def normalize(self):
        self.model = {c: (self.acc[c]>=0).astype(np.int32) for c in self.acc}

    def predict(self, H):
        distances = {c: hamming(H, self.model[c]) for c in self.model}
        return min(distances, key=distances.get), distances

    def adaptive_train(self, H_list, labels):
        for H, true in zip(H_list, labels):
            pred, dists = self.predict(H)
            if pred == true: continue
            δ_true, δ_pred = 1-dists[true], 1-dists[pred]
            bH = bin2bipolar(H)
            self.acc[true] += self.lr*(1-δ_true)*bH
            self.acc[pred] -= self.lr*(1-δ_pred)*bH
        self.normalize()

    def iterative_retrain(self, H_list, labels, epochs):
        for e in range(epochs):
            prev_acc = {c: self.acc[c].copy() for c in self.acc}
            for H, true in zip(H_list, labels):
                pred, dists = self.predict(H)
                if pred == true: continue
                δ_true, δ_pred = 1-dists[true], 1-dists[pred]
                bH = bin2bipolar(H)
                self.acc[true] += self.lr*(1-δ_true)*bH
                self.acc[pred] -= self.lr*(1-δ_pred)*bH
            self.normalize()
            diff = max(np.linalg.norm(self.acc[c]-prev_acc[c]) for c in self.acc)
            if diff < tol:
                print(f"Converged after {e+1} iterations (diff={diff:.2e})")
                break

enc_train = Parallel(n_jobs=-1)(delayed(encode)(img) for img in tqdm(X_train, desc="Enc Train"))
enc_test  = Parallel(n_jobs=-1)(delayed(encode)(img) for img in tqdm(X_test, desc="Enc Test"))

hd = OnlineHD(d, n_classes, lr=learning_rate)
hd.initial_train(enc_train, y_train)
hd.normalize()

hd.adaptive_train(enc_train, y_train)

hd.iterative_retrain(enc_train, y_train, epochs=max_iter)

y_pred = [hd.predict(H)[0] for H in enc_test]
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")