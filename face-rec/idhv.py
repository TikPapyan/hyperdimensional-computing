import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = fetch_olivetti_faces(shuffle=True, random_state=42)
X = data.images
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
num_classes = len(np.unique(y_train))
print(f"Dataset: {len(X_train)} train, {len(X_test)} test, {num_classes} classes")

D = 10000
L = 21
H, W = X.shape[1:]

pos_vectors = np.random.choice([-1, 1], size=(H * W, D))
level_vectors = np.random.choice([-1, 1], size=(L, D))

def encode_image(img):
    flat = img.ravel()
    levels = np.floor(flat * (L - 1)).astype(int)

    bundled = np.zeros(D, dtype=int)
    for idx, q in enumerate(levels):
        bundled += pos_vectors[idx] * level_vectors[q]

    return np.where(bundled >= 0, 1, -1)

prototypes = np.zeros((num_classes, D), dtype=int)

for c in range(num_classes):
    imgs_c = X_train[y_train == c]
    hvs = np.array([encode_image(im) for im in imgs_c])
    summed = hvs.sum(axis=0)
    prototypes[c] = np.where(summed >= 0, 1, -1)

def predict(img):
    hv = encode_image(img)
    sims = prototypes @ hv
    return np.argmax(sims)

y_pred = np.array([predict(im) for im in X_test])

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")