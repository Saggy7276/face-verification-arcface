"""Shared paths, preprocessing, data loading and embedding helpers used by every script."""
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "dataset"                  # selected LFW images, one folder per person
PROC_DIR = DATA_DIR / "dataset_processed"       # detected + cropped + resized faces
SPLITS_PATH = DATA_DIR / "splits.json"
LABELS_PATH = DATA_DIR / "labels.csv"
STATS_PATH = DATA_DIR / "dataset_stats.json"
PAIRS_DIR = DATA_DIR / "pairs"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
RESULTS_JSON = RESULTS_DIR / "evaluation_results.json"

# --------------------------------------------------------------------------- preprocessing constants
IMG_SIZE = 112                                   # model input resolution (H = W)
MEAN = (0.485, 0.456, 0.406)                     # ImageNet statistics: the backbone was pretrained with them
STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- face detection / cropping
class FaceCropper:
    """OpenCV Haar-cascade face detector + square crop + resize.

    LFW labels the face in the centre of each 250x250 image, so when several faces are
    detected the one closest to the centre is kept. If no face is found near the centre
    the cropper falls back to a fixed centre crop sized to LFW's known face scale.
    """

    def __init__(self, size: int = IMG_SIZE, margin: float = 1.25):
        cascade = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.detector = cv2.CascadeClassifier(cascade)
        if self.detector.empty():
            raise RuntimeError(f"Could not load Haar cascade from {cascade}")
        self.size = size
        self.margin = margin

    def detect(self, img_bgr):
        """Return (x, y, w, h) of the centre-most face, or None."""
        h, w = img_bgr.shape[:2]
        gray = cv2.equalizeHist(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
        min_side = int(0.2 * min(h, w))
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
        if len(faces) == 0:
            return None
        cx, cy = w / 2, h / 2
        x, y, bw, bh = min(faces, key=lambda f: ((f[0] + f[2] / 2 - cx) ** 2 + (f[1] + f[3] / 2 - cy) ** 2, -f[2] * f[3]))
        # A face far from the centre is a background person, not the labelled identity.
        if abs(x + bw / 2 - cx) > 0.2 * w or abs(y + bh / 2 - cy) > 0.2 * h:
            return None
        return int(x), int(y), int(bw), int(bh)

    @staticmethod
    def _square_crop(img, cx, cy, side):
        h, w = img.shape[:2]
        side = int(round(side))
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        x1, y1 = x0 + side, y0 + side
        pad = max(0, -x0, -y0, x1 - w, y1 - h)
        if pad:
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
        return img[y0:y1, x0:x1]

    def __call__(self, img_bgr):
        """Return (face_crop_bgr, detected: bool)."""
        h, w = img_bgr.shape[:2]
        box = self.detect(img_bgr)
        if box is not None:
            x, y, bw, bh = box
            cx, cy, side = x + bw / 2, y + bh / 2, max(bw, bh) * self.margin
        else:
            # LFW crops were enlarged 2.2x around a Viola-Jones box, so the face spans ~1/2.2 of the image.
            cx, cy, side = w / 2, h / 2, min(h, w) / 2.2 * self.margin
        crop = self._square_crop(img_bgr, cx, cy, side)
        crop = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return crop, box is not None


# --------------------------------------------------------------------------- tensors / datasets
def train_transform():
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def eval_transform():
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


def load_splits() -> dict:
    if not SPLITS_PATH.exists():
        raise SystemExit(f"{SPLITS_PATH} not found. Run dataset_preparation.py first.")
    with open(SPLITS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def split_samples(split: str):
    """List of (relative_image_path, person_id) for one split, in deterministic order."""
    ids = load_splits()["splits"][split]
    return [(rel, pid) for pid in sorted(ids) for rel in ids[pid]]


class FaceDataset(Dataset):
    def __init__(self, relpaths, labels=None, transform=None):
        self.relpaths = list(relpaths)
        self.labels = labels
        self.transform = transform or eval_transform()

    def __len__(self):
        return len(self.relpaths)

    def __getitem__(self, i):
        img = Image.open(PROC_DIR / self.relpaths[i]).convert("RGB")
        x = self.transform(img)
        return (x, self.labels[i]) if self.labels is not None else x


# --------------------------------------------------------------------------- model loading / embeddings
def load_model(checkpoint, device):
    from model import FaceEmbeddingNet

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = FaceEmbeddingNet(arch=ckpt["arch"], embedding_dim=ckpt["embedding_dim"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def compute_embeddings(model, relpaths, device, batch_size=128, flip_tta=True, num_workers=0):
    """L2-normalised embeddings (N, D) for images in PROC_DIR.

    With flip_tta the embeddings of the image and its horizontal mirror are summed
    before normalisation (a standard, label-free test-time augmentation).
    """
    model.eval()
    loader = DataLoader(FaceDataset(relpaths), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        e = model(x)
        if flip_tta:
            e = e + model(torch.flip(x, dims=[3]))
        out.append(F.normalize(e.float(), dim=1).cpu())
    return torch.cat(out).numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two (N, D) arrays: a.b / (|a| |b|)."""
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.clip(den, 1e-12, None)


def update_results_json(section: str, values: dict) -> None:
    """Merge one section into results/evaluation_results.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON, encoding="utf-8") as fh:
            data = json.load(fh)
    data[section] = values
    with open(RESULTS_JSON, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
