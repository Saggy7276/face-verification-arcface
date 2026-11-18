"""Step 1 - Dataset preparation.

Steps
  1. Download LFW (deep-funneled version) from Kaggle via kagglehub.
  2. Select identities with >= --min-images images, keep at most --max-images per identity.
  3. Copy them into a per-person layout:  data/dataset/person_001/img_01.jpg ...
  4. Face detection (OpenCV Haar cascade) -> square crop with margin -> resize to 112x112
     -> data/dataset_processed/ (same layout). Pixel normalisation (ImageNet mean/std) is
     applied at load time in common.py, so the processed images stay viewable JPEGs.
  5. Identity-disjoint train / val / test split -> data/splits.json
  6. Name mapping (data/labels.csv), statistics (data/dataset_stats.json) and an
     example figure (results/preprocessing_examples.png).

Usage
    python dataset_preparation.py                 # defaults: >=4 images, <=30 per identity
    python dataset_preparation.py --overwrite     # rebuild from scratch
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import Counter

import cv2
import numpy as np

from common import (DATA_DIR, IMG_SIZE, LABELS_PATH, PROC_DIR, RAW_DIR, RESULTS_DIR, SPLITS_PATH, STATS_PATH,
                    FaceCropper)

KAGGLE_DATASET = "jessicali9530/lfw-dataset"
IMG_EXTS = (".jpg", ".jpeg", ".png")


def find_people_root(base):
    """Return the directory whose subfolders are the per-person folders (LFW has 5,749)."""
    for dirpath, dirnames, _ in os.walk(base):
        if len(dirnames) > 1000:
            return dirpath
    sys.exit(f"Could not find LFW person folders under {base}")


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


def select_identities(src, min_images, max_images, max_ids):
    selected = []
    for name in sorted(os.listdir(src)):
        folder = os.path.join(src, name)
        if not os.path.isdir(folder):
            continue
        imgs = list_images(folder)
        if len(imgs) >= min_images:
            selected.append((name, imgs[:max_images]))
    if max_ids:
        selected = selected[:max_ids]
    return selected


def split_identities(pids, val_frac, test_frac, seed):
    shuffled = list(pids)
    random.Random(seed).shuffle(shuffled)
    n_test = round(len(shuffled) * test_frac)
    n_val = round(len(shuffled) * val_frac)
    return {
        "test": sorted(shuffled[:n_test]),
        "val": sorted(shuffled[n_test:n_test + n_val]),
        "train": sorted(shuffled[n_test + n_val:]),
    }


def save_examples_figure(examples, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(examples), figsize=(2 * len(examples), 4.4))
    for col, (raw, crop, detected) in enumerate(examples):
        axes[0, col].imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        axes[1, col].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[1, col].set_title("detected" if detected else "fallback", fontsize=8)
        for r in range(2):
            axes[r, col].axis("off")
    axes[0, 0].set_title("raw 250x250", fontsize=8, loc="left")
    fig.suptitle(f"Preprocessing: LFW image (top) -> face crop {IMG_SIZE}x{IMG_SIZE} (bottom)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-images", type=int, default=4, help="minimum images per identity (default 4)")
    p.add_argument("--max-images", type=int, default=30, help="cap on images per identity (default 30)")
    p.add_argument("--max-ids", type=int, default=0, help="limit number of identities (0 = all eligible)")
    p.add_argument("--val-frac", type=float, default=0.10, help="fraction of identities for validation")
    p.add_argument("--test-frac", type=float, default=0.20, help="fraction of identities for test")
    p.add_argument("--margin", type=float, default=1.25, help="crop size relative to detected face box")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="delete existing data/ outputs first")
    args = p.parse_args()

    for d in (RAW_DIR, PROC_DIR):
        if d.exists() and any(d.iterdir()):
            if not args.overwrite:
                sys.exit(f"'{d}' already exists. Use --overwrite to rebuild.")
            shutil.rmtree(d)

    import kagglehub
    download_dir = kagglehub.dataset_download(KAGGLE_DATASET)
    src = find_people_root(download_dir)
    print(f"LFW source folder: {src}")

    selected = select_identities(src, args.min_images, args.max_images, args.max_ids)
    if len(selected) < 50:
        sys.exit(f"Only {len(selected)} identities qualify; at least 50 are needed.")

    cropper = FaceCropper(size=IMG_SIZE, margin=args.margin)
    rng = random.Random(args.seed)
    labels, per_person_images = [], {}
    n_detected = n_total = 0
    examples = []

    for i, (name, files) in enumerate(selected, 1):
        pid = f"person_{i:03d}"
        (RAW_DIR / pid).mkdir(parents=True)
        (PROC_DIR / pid).mkdir(parents=True)
        rels = []
        for j, f in enumerate(files, 1):
            out_name = f"img_{j:02d}.jpg"
            raw_path = RAW_DIR / pid / out_name
            shutil.copy2(os.path.join(src, name, f), raw_path)

            img = cv2.imread(str(raw_path))
            if img is None:
                sys.exit(f"Could not read {raw_path}")
            crop, detected = cropper(img)
            cv2.imwrite(str(PROC_DIR / pid / out_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n_total += 1
            n_detected += detected
            rels.append(f"{pid}/{out_name}")
            if len(examples) < 8 and rng.random() < 0.01:
                examples.append((img, crop, detected))
        per_person_images[pid] = rels
        labels.append((pid, name.replace("_", " ")))
        if i % 100 == 0:
            print(f"  processed {i}/{len(selected)} identities")

    splits = split_identities(per_person_images.keys(), args.val_frac, args.test_frac, args.seed)
    assert not (set(splits["train"]) & set(splits["test"])) and not (set(splits["train"]) & set(splits["val"]))
    split_of = {pid: s for s, ids in splits.items() for pid in ids}

    DATA_DIR.mkdir(exist_ok=True)
    with open(SPLITS_PATH, "w", encoding="utf-8") as fh:
        json.dump({
            "seed": args.seed,
            "note": "Identity-disjoint: no person appears in more than one split.",
            "splits": {s: {pid: per_person_images[pid] for pid in ids} for s, ids in splits.items()},
        }, fh, indent=1)

    with open(LABELS_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["person_id", "name", "split", "num_images"])
        for pid, name in labels:
            w.writerow([pid, name, split_of[pid], len(per_person_images[pid])])

    counts = [len(v) for v in per_person_images.values()]
    stats = {
        "source": "Labeled Faces in the Wild (LFW), deep-funneled version",
        "source_urls": ["http://vis-www.cs.umass.edu/lfw/", f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"],
        "selection": {"min_images_per_identity": args.min_images, "max_images_per_identity": args.max_images,
                      "max_ids": args.max_ids or "all eligible"},
        "num_identities": len(per_person_images),
        "num_images": n_total,
        "images_per_identity": {"min": min(counts), "max": max(counts), "mean": round(float(np.mean(counts)), 2),
                                "median": float(np.median(counts))},
        "face_detection": {"detector": "OpenCV Haar cascade (haarcascade_frontalface_default)",
                           "detected": n_detected, "fallback_centre_crop": n_total - n_detected,
                           "detection_rate": round(n_detected / n_total, 4)},
        "preprocessing": {"crop": f"square, {args.margin}x detected box", "resize": f"{IMG_SIZE}x{IMG_SIZE}",
                          "alignment": "LFW deep-funneled images are pre-aligned (Huang et al., NIPS 2012); "
                                       "no additional landmark alignment",
                          "normalisation": "ToTensor (0-1) then ImageNet mean/std, applied at load time"},
        "splits": {s: {"identities": len(ids), "images": sum(len(per_person_images[p]) for p in ids)}
                   for s, ids in splits.items()},
        "split_seed": args.seed,
        "image_count_histogram": dict(sorted(Counter(counts).items())),
    }
    with open(STATS_PATH, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    RESULTS_DIR.mkdir(exist_ok=True)
    if examples:
        save_examples_figure(examples, RESULTS_DIR / "preprocessing_examples.png")

    print(f"\nIdentities: {stats['num_identities']}   images: {stats['num_images']}")
    print(f"Face detected in {n_detected}/{n_total} images ({100 * n_detected / n_total:.1f}%), rest centre-cropped")
    for s in ("train", "val", "test"):
        print(f"  {s:5s}: {stats['splits'][s]['identities']:4d} identities, {stats['splits'][s]['images']:5d} images")
    print(f"Wrote {RAW_DIR}, {PROC_DIR}, {SPLITS_PATH}, {LABELS_PATH}, {STATS_PATH}")


if __name__ == "__main__":
    main()
