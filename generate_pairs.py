"""Positive / negative pair generation.

Pairs are drawn ONLY from one held-out split (val or test); those identities never appear
in training (checked below), so there is no train/test leakage.

Positive pairs (label 1): two different images of the same identity.
  * All unordered combinations (i < j) per identity are enumerated, shuffled, then taken
    round-robin across identities until the target is met, so identities with many photos
    (up to 30 -> 435 combinations) cannot dominate identities with 4 photos (6 combinations).
Negative pairs (label 0): two images of different identities.
  * Draw an identity pair uniformly at random, then one image from each.
  * Same number as positives (balanced).
Duplicates: every pair is stored in canonical (sorted) order and kept in a set, so (a, b)
and (b, a) count once; a pair never contains the same image twice.

Usage
    python generate_pairs.py                       # val (2000+2000) and test (5000+5000)
    python generate_pairs.py --split test --num-pos 5000
"""
import argparse
import csv
import random
from itertools import combinations

from common import PAIRS_DIR, load_splits

DEFAULT_TARGETS = {"val": 2000, "test": 5000}


def positive_pairs(id_to_imgs, target, rng):
    pools = {}
    for pid in sorted(id_to_imgs):
        combos = list(combinations(sorted(id_to_imgs[pid]), 2))
        rng.shuffle(combos)
        pools[pid] = combos
    order = sorted(pools)
    pairs = []
    while len(pairs) < target and any(pools.values()):
        for pid in order:
            if pools[pid]:
                a, b = pools[pid].pop()
                pairs.append((a, b, pid, pid, 1))
                if len(pairs) == target:
                    break
    return pairs


def negative_pairs(id_to_imgs, target, rng):
    pids = sorted(id_to_imgs)
    sizes = [len(id_to_imgs[p]) for p in pids]
    total = sum(sizes)
    max_possible = (total * total - sum(s * s for s in sizes)) // 2
    target = min(target, max_possible)
    seen, pairs = set(), []
    while len(pairs) < target:
        pa, pb = rng.sample(pids, 2)
        a, b = rng.choice(id_to_imgs[pa]), rng.choice(id_to_imgs[pb])
        if b < a:
            a, b, pa, pb = b, a, pb, pa
        if (a, b) in seen:
            continue
        seen.add((a, b))
        pairs.append((a, b, pa, pb, 0))
    return pairs


def generate(split, num_pos, num_neg, seed):
    splits = load_splits()["splits"]
    id_to_imgs = splits[split]
    leaked = set(id_to_imgs) & set(splits["train"])
    assert not leaked, f"{split} identities found in train split: {sorted(leaked)[:5]}"

    rng = random.Random(seed)
    pos = positive_pairs(id_to_imgs, num_pos, rng)
    neg = negative_pairs(id_to_imgs, num_neg if num_neg else len(pos), rng)

    keys = [(a, b) for a, b, *_ in pos + neg]
    assert len(keys) == len(set(keys)), "duplicate pair found"
    assert all(a != b for a, b in keys), "self-pair found"

    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    out = PAIRS_DIR / f"{split}_pairs.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id", "img_a", "img_b", "id_a", "id_b", "label"])
        for i, row in enumerate(pos + neg):
            w.writerow([i, *row])

    pos_ids = {r[2] for r in pos}
    print(f"[{split}] identities={len(id_to_imgs)} images={sum(map(len, id_to_imgs.values()))} | "
          f"positives={len(pos)} (from {len(pos_ids)} identities) negatives={len(neg)} -> {out}")
    if len(pos) < num_pos:
        print(f"  note: only {len(pos)} distinct positive pairs exist in this split (target {num_pos})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["val", "test", "all"], default="all")
    p.add_argument("--num-pos", type=int, default=None, help="target positives (default: val 2000, test 5000)")
    p.add_argument("--num-neg", type=int, default=None, help="negatives (default: same as positives)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    for split in (["val", "test"] if args.split == "all" else [args.split]):
        generate(split, args.num_pos or DEFAULT_TARGETS[split], args.num_neg, args.seed)


if __name__ == "__main__":
    main()
