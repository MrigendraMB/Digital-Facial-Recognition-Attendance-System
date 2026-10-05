import os
import cv2
import numpy as np
import pickle

MODEL_PATH = "model.pkl"


class FaceCentroidModel:
    def __init__(self, centroids):
        self.centroids = centroids
        self.classes_ = np.array(sorted(centroids.keys()), dtype=int)

    def predict_proba(self, X):
        probabilities = []
        for emb in X:
            scores = []
            for class_id in self.classes_:
                centroid = self.centroids[int(class_id)]
                distance = float(np.linalg.norm(emb - centroid))
                score = 1.0 / (distance + 1e-6)
                scores.append(score)
            scores = np.array(scores, dtype=np.float64)
            scores_sum = scores.sum()
            if scores_sum == 0:
                probabilities.append(np.ones(len(scores), dtype=np.float64) / len(scores))
            else:
                probabilities.append(scores / scores_sum)
        return np.asarray(probabilities, dtype=np.float64)


def _face_cascade():
    cascade_cls = getattr(cv2, "CascadeClassifier", None)
    if cascade_cls is None:
        return None
    cascade_root = getattr(cv2, "data", None)
    if cascade_root is None or not hasattr(cascade_root, "haarcascades"):
        return None
    cascade_path = os.path.join(cascade_root.haarcascades, "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        return None
    try:
        detector = cascade_cls(cascade_path)
        if hasattr(detector, "empty") and detector.empty():
            return None
        return detector
    except Exception:
        return None


def _center_bbox(bgr_image):
    h, w = bgr_image.shape[:2]
    box_w = max(40, int(w * 0.6))
    box_h = max(40, int(h * 0.6))
    x1 = max(0, (w - box_w) // 2)
    y1 = max(0, (h - box_h) // 2)
    return x1, y1, min(w, x1 + box_w), min(h, y1 + box_h)


def detect_face_bbox(bgr_image):
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    detector = _face_cascade()
    if detector is None:
        return _center_bbox(bgr_image)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return _center_bbox(bgr_image)
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    return x, y, x + w, y + h

# ---- Utility: extract face crop -> small grayscale vector (embedding) ----
def crop_face_and_embed(bgr_image, bbox):
    h, w = bgr_image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    face = bgr_image[y1:y2, x1:x2]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (32,32), interpolation=cv2.INTER_AREA)
    emb = face.flatten().astype(np.float32) / 255.0
    return emb

def extract_embedding_for_image(stream_or_bytes):
    # accepts a file-like stream (werkzeug FileStorage.stream)
    # read image from stream into numpy BGR
    data = stream_or_bytes.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    bbox = detect_face_bbox(img)
    if bbox is None:
        return None
    emb = crop_face_and_embed(img, bbox)
    return emb

# ---- Load model helpers ----
def load_model_if_exists():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict_with_model(clf, emb):
    # returns label and confidence (max probability)
    proba = clf.predict_proba([emb])[0]
    idx = np.argmax(proba)
    label = clf.classes_[idx]
    conf = float(proba[idx])
    return label, conf

# ---- Training function used in background ----
def train_model_background(dataset_dir, progress_callback=None):
    """
    dataset_dir/
        student_id/
            img1.jpg
            img2.jpg
    progress_callback(progress_percent, message) -> optional
    """
    X = []
    y = []
    student_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    total_students = max(1, len(student_dirs))
    processed = 0

    for sid in student_dirs:
        folder = os.path.join(dataset_dir, sid)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))]
        for fn in files:
            path = os.path.join(folder, fn)
            img = cv2.imread(path)
            if img is None:
                continue
            bbox = detect_face_bbox(img)
            if bbox is None:
                continue
            emb = crop_face_and_embed(img, bbox)
            if emb is None:
                continue
            X.append(emb)
            y.append(int(sid))
        processed += 1
        if progress_callback:
            pct = int((processed/total_students)*80)  # training progress up to 80% during feature extraction
            progress_callback(pct, f"Processed {processed}/{total_students} students")

    if len(X) == 0:
        if progress_callback:
            progress_callback(0, "No training data found")
        return

    # convert
    X = np.stack(X)
    y = np.array(y)

    # fit centroid-based classifier for fast, dependency-light recognition
    if progress_callback:
        progress_callback(85, "Training face centroids...")
    centroids = {}
    for class_id in sorted(set(y.tolist())):
        centroids[int(class_id)] = np.mean(X[y == class_id], axis=0)
    clf = FaceCentroidModel(centroids)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    if progress_callback:
        progress_callback(100, "Training complete")
