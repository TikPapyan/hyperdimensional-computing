import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

POI = 2048
PATCH_SIZE = (16, 16)
LBP_P = 8
LBP_R = 1
LBP_METHOD = 'uniform'

data = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = data.images, data.target
n_classes = len(data.target)

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

print(f"Dataset: {len(X_train)} train, {len(X_test)} test, {n_classes} classes")

def extract_pois(images, poi=2048):
    np.random.seed(0)
    all_pois = []
    n_imgs = len(images)
    pois_per_img = poi // n_imgs
    for img in images:
        h, w = img.shape
        ys = np.random.randint(0, h - PATCH_SIZE[0], pois_per_img)
        xs = np.random.randint(0, w - PATCH_SIZE[1], pois_per_img)
        all_pois.extend([(y, x) for y, x in zip(ys, xs)])
    return all_pois[:poi]

def encode_patch_lbp(patch):
    patch = (patch * 255).astype(np.uint8)
    lbp = local_binary_pattern(patch, LBP_P, LBP_R, method=LBP_METHOD)
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_P + 3),
        range=(0, LBP_P + 2)
    )
    return hist.astype(np.float32) / (np.linalg.norm(hist) + 1e-6)

def encode_image_shift_lbp(img, pois):
    features = []
    for (y, x) in pois:
        patch = img[y:y+PATCH_SIZE[0], x:x+PATCH_SIZE[1]]
        feat = encode_patch_lbp(patch)
        features.append(feat)
    return np.concatenate(features)

pois = extract_pois(X_train[:10], poi=POI)

print("Encoding training set...")
X_train_enc = np.array([encode_image_shift_lbp(im, pois) for im in tqdm(X_train)])
print("Encoding test set...")
X_test_enc = np.array([encode_image_shift_lbp(im, pois) for im in tqdm(X_test)])

def get_prototypes(X, y):
    classes = np.unique(y)
    prototypes = {}
    for c in classes:
        class_vecs = X[y == c]
        prototypes[c] = class_vecs.mean(axis=0)
    return prototypes

def predict(X, prototypes):
    preds = []
    for vec in X:
        sims = {cls: np.dot(vec, proto) / (np.linalg.norm(proto) * np.linalg.norm(vec) + 1e-6)
                for cls, proto in prototypes.items()}
        pred = max(sims, key=sims.get)
        preds.append(pred)
    return np.array(preds)

prototypes = get_prototypes(X_train_enc, y_train)
y_pred = predict(X_test_enc, prototypes)

acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%")