import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Parameters
D = 10000  # hypervector dimensionality
η = 0.7    # learning rate

# Load and flatten images
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X = data.images.reshape((-1, 64 * 64))
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
num_classes = len(np.unique(y_train))
print(f"Dataset: {len(X_train)} train, {len(X_test)} test, {num_classes} classes")

# Random base hypervectors
np.random.seed(42)
base_hv = np.random.choice([-1, 1], size=(X.shape[1], D))  # Now X.shape[1] = 4096

def encode(x):
    """Encodes input into a hypervector"""
    return np.dot(x, base_hv)

def cosine(vec1, vec2):
    """Efficient cosine similarity"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

# OnlineHD Training
class_hv = np.zeros((num_classes, D))
for epoch in tqdm(range(5)):
    for i in range(len(X_train)):
        x = X_train[i]
        y_true = y_train[i]
        h = encode(x)

        sims = np.array([cosine(h, c) for c in class_hv])
        y_pred = np.argmax(sims)

        if y_pred == y_true:
            continue

        δ_true = sims[y_true]
        δ_pred = sims[y_pred]

        # Adaptive retraining update
        class_hv[y_true] += η * (1 - δ_true) * h
        class_hv[y_pred] -= η * (1 - δ_pred) * h

# Inference
y_preds = []
for x in X_test:
    h = encode(x)
    sims = np.array([cosine(h, c) for c in class_hv])
    y_preds.append(np.argmax(sims))

acc = accuracy_score(y_test, y_preds)
print(f"Accuracy: {acc * 100:.2f}%")