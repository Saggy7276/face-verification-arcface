"""Embeddings & cosine similarity for every evaluation pair.

    Image A -> ResNet -> embedding A \
                                      cosine similarity = A.B / (|A| |B|)
    Image B -> ResNet -> embedding B /

Each unique image is embedded once, then the similarity of every pair in
data/pairs/{split}_pairs.csv is written to results/{split}_pair_scores.csv.

Single-pair mode (any two face photos, cropped with the same Haar detector):
    python evaluate_pairs.py --image-a a.jpg --image-b b.jpg

Usage
    python evaluate_pairs.py                         # scores val + test pairs
"""
import argparse
import csv
import json

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from common import (CKPT_DIR, IMG_SIZE, PAIRS_DIR, RESULTS_DIR, RESULTS_JSON, FaceCropper, compute_embeddings,
                    cosine_similarity, eval_transform, get_device, load_model)


def score_split(model, device, split, flip_tta):
    path = PAIRS_DIR / f"{split}_pairs.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run generate_pairs.py first.")
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    images = sorted({r["img_a"] for r in rows} | {r["img_b"] for r in rows})
    index = {p: i for i, p in enumerate(images)}
    emb = compute_embeddings(model, images, device, flip_tta=flip_tta)
    a = emb[[index[r["img_a"]] for r in rows]]
    b = emb[[index[r["img_b"]] for r in rows]]
    sims = cosine_similarity(a, b)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{split}_pair_scores.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id", "img_a", "img_b", "id_a", "id_b", "label", "cosine_similarity"])
        for r, s in zip(rows, sims):
            w.writerow([r["pair_id"], r["img_a"], r["img_b"], r["id_a"], r["id_b"], r["label"], f"{s:.6f}"])

    labels = np.array([int(r["label"]) for r in rows])
    print(f"[{split}] {len(rows)} pairs, {len(images)} unique images, embedding dim {emb.shape[1]} | "
          f"mean genuine {sims[labels == 1].mean():.3f}  mean impostor {sims[labels == 0].mean():.3f}  "
          f"AUC {roc_auc_score(labels, sims):.4f} -> {out}")


def crop_file(path, cropper):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Cannot read {path}")
    return cropper(img)


@torch.no_grad()
def embed_crop(model, device, crop, flip_tta):
    x = eval_transform()(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    e = model(x)
    if flip_tta:
        e = e + model(torch.flip(x, dims=[3]))
    return torch.nn.functional.normalize(e, dim=1).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(CKPT_DIR / "best_model.pth"))
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--no-flip-tta", action="store_true", help="disable horizontal-flip test-time augmentation")
    p.add_argument("--image-a", help="single-pair mode: first image")
    p.add_argument("--image-b", help="single-pair mode: second image")
    args = p.parse_args()

    single = bool(args.image_a or args.image_b)
    if single:
        if not (args.image_a and args.image_b):
            raise SystemExit("Give both --image-a and --image-b")
        # Detect faces before CUDA is initialised: OpenCV's thread pool occasionally raised an
        # "Unknown C++ exception" when detectMultiScale ran after torch had set up CUDA.
        cv2.setNumThreads(0)
        cropper = FaceCropper(size=IMG_SIZE)
        crop_a, da = crop_file(args.image_a, cropper)
        crop_b, db = crop_file(args.image_b, cropper)

    device = get_device()
    model, ckpt = load_model(args.checkpoint, device)
    print(f"Loaded {ckpt['arch']} (epoch {ckpt['epoch']}, val AUC {ckpt['val_auc']:.4f}) on {device}")

    if single:
        ea = embed_crop(model, device, crop_a, not args.no_flip_tta)
        eb = embed_crop(model, device, crop_b, not args.no_flip_tta)
        sim = float(cosine_similarity(ea, eb)[0])
        thr = None
        if RESULTS_JSON.exists():
            thr = json.load(open(RESULTS_JSON, encoding="utf-8")).get("pair_verification", {}).get("threshold")
        print(f"face detected: A={da} B={db}")
        print(f"cosine similarity = {sim:.4f}")
        if thr is not None:
            print(f"threshold = {thr:.2f} -> {'MATCH (same person)' if sim >= thr else 'NON-MATCH (different people)'}")
        return

    for split in args.splits:
        score_split(model, device, split, not args.no_flip_tta)


if __name__ == "__main__":
    main()
