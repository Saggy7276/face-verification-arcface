"""Gallery / Probe identification on the held-out TEST identities.

For every test identity, --gallery-per-id images (chosen with a fixed seed) are enrolled;
their embeddings are averaged and re-normalised into one gallery template. All remaining
images of that identity become probes. Every probe is compared with every gallery
template by cosine similarity; the most similar template is the predicted identity.

Reports Rank-1, Rank-5, Rank-10 and the CMC curve. Default is single-shot enrolment
(1 image per identity), the harder and more common protocol.

Outputs: results/gallery_probe_results.csv, results/cmc_curve.png,
         evaluation_results.json ("gallery_probe" section)

Usage
    python gallery_probe.py
    python gallery_probe.py --gallery-per-id 2
    python gallery_probe.py --gallery-image person_009/img_02.jpg person_013/img_01.jpg
"""
import argparse
import csv
import random

import matplotlib
import numpy as np

from common import CKPT_DIR, RESULTS_DIR, compute_embeddings, get_device, load_model, load_splits, update_results_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(CKPT_DIR / "best_model.pth"))
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--gallery-per-id", type=int, default=1)
    p.add_argument("--max-rank", type=int, default=20)
    p.add_argument("--no-flip-tta", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gallery-image", nargs="+", default=[], metavar="PERSON/IMG",
                   help="enrol these exact images (e.g. person_009/img_02.jpg) instead of the seeded random "
                        "choice for their identities; other identities are unaffected")
    args = p.parse_args()

    id_to_imgs = load_splits()["splits"][args.split]

    # Explicit gallery choices, grouped by identity. Paths are relative to data/dataset_processed.
    owner = {img: pid for pid, imgs in id_to_imgs.items() for img in imgs}
    chosen = {}
    for raw in args.gallery_image:
        img = raw.replace("\\", "/").strip("/")
        if img not in owner:
            raise SystemExit(f"--gallery-image {raw}: not an image of the {args.split} split. "
                             f"Use the form person_XXX/img_YY.jpg with a {args.split} identity.")
        if img not in chosen.setdefault(owner[img], []):
            chosen[owner[img]].append(img)
    for pid, picks in chosen.items():
        if len(picks) > args.gallery_per_id:
            raise SystemExit(f"{len(picks)} gallery images given for {pid} but --gallery-per-id is "
                             f"{args.gallery_per_id}. Raise --gallery-per-id or give fewer images.")

    rng = random.Random(args.seed)
    gallery_ids, gallery_sets, probes = [], [], []
    for pid in sorted(id_to_imgs):
        imgs = sorted(id_to_imgs[pid])
        rng.shuffle(imgs)                             # always shuffled, so other identities keep their choice
        if pid in chosen:
            imgs = chosen[pid] + [i for i in imgs if i not in chosen[pid]]
        if len(imgs) <= args.gallery_per_id:
            continue                                  # needs at least one probe left
        gallery_ids.append(pid)
        gallery_sets.append(imgs[:args.gallery_per_id])
        probes += [(img, pid) for img in imgs[args.gallery_per_id:]]

    device = get_device()
    model, ckpt = load_model(args.checkpoint, device)
    flip = not args.no_flip_tta
    g_flat = [img for s in gallery_sets for img in s]
    g_emb = compute_embeddings(model, g_flat, device, flip_tta=flip)
    g_emb = g_emb.reshape(len(gallery_ids), args.gallery_per_id, -1).mean(1)
    g_emb /= np.linalg.norm(g_emb, axis=1, keepdims=True)           # one unit-length template per identity
    p_emb = compute_embeddings(model, [img for img, _ in probes], device, flip_tta=flip)

    sim = p_emb @ g_emb.T                                             # cosine similarity (unit vectors)
    order = np.argsort(-sim, axis=1)
    gallery_ids = np.array(gallery_ids)
    true_idx = np.array([np.where(gallery_ids == pid)[0][0] for _, pid in probes])
    ranks = np.array([int(np.where(order[i] == true_idx[i])[0][0]) + 1 for i in range(len(probes))])

    max_rank = min(args.max_rank, len(gallery_ids))
    cmc = [float(np.mean(ranks <= k)) for k in range(1, max_rank + 1)]
    results = {
        "split": args.split, "gallery_identities": len(gallery_ids), "gallery_images_per_identity": args.gallery_per_id,
        "probe_images": len(probes), "rank1": cmc[0], "rank5": cmc[min(4, max_rank - 1)],
        "rank10": cmc[min(9, max_rank - 1)], "chance_rank1": 1 / len(gallery_ids),
        "mean_rank_of_true_identity": float(ranks.mean()), "cmc": cmc, "flip_tta": flip,
    }
    if chosen:
        results["gallery_images_chosen_by_user"] = sorted(i for picks in chosen.values() for i in picks)
    update_results_json("gallery_probe", results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "gallery_probe_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["probe_image", "true_id", "predicted_id", "top1_similarity", "true_id_similarity",
                    "rank_of_true_id", "correct_rank1", "top5_ids"])
        for i, (img, pid) in enumerate(probes):
            top = order[i]
            w.writerow([img, pid, gallery_ids[top[0]], f"{sim[i, top[0]]:.4f}", f"{sim[i, true_idx[i]]:.4f}",
                        ranks[i], int(ranks[i] == 1), " ".join(gallery_ids[top[:5]])])

    fig, a = plt.subplots(figsize=(7, 4.4))
    a.plot(range(1, max_rank + 1), cmc, marker="o", ms=4)
    a.set_xlabel("rank k"); a.set_ylabel("identification rate (true id in top-k)")
    a.set_title(f"CMC - {len(gallery_ids)} gallery ids, {len(probes)} probes "
                f"(Rank-1 {cmc[0]:.3f}, Rank-5 {results['rank5']:.3f})")
    a.set_xticks(range(1, max_rank + 1)); a.set_ylim(0, 1.01); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "cmc_curve.png", dpi=130); plt.close(fig)

    for pid in sorted(chosen):
        print(f"Gallery for {pid} set by --gallery-image: {' '.join(gallery_sets[list(gallery_ids).index(pid)])}")
    print(f"Gallery: {len(gallery_ids)} identities x {args.gallery_per_id} image(s)   Probes: {len(probes)}")
    print(f"Rank-1 {cmc[0]:.4f}   Rank-5 {results['rank5']:.4f}   Rank-10 {results['rank10']:.4f}   "
          f"(chance {results['chance_rank1']:.4f})")


if __name__ == "__main__":
    main()
