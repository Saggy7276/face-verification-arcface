"""ROC / AUC and threshold analysis from real cosine-similarity scores.

* Threshold is chosen on the VALIDATION pairs (max Youden's J = TPR - FPR, i.e. best
  balanced accuracy), rounded to 2 decimals, then applied unchanged to the TEST pairs.
  The test set is not used to pick it.
* Reports on test: ROC curve, AUC, EER, TAR @ FAR = 1% / 0.1%, and accuracy, precision,
  recall (= TAR), F1, FAR and FRR at the chosen threshold, plus a threshold sweep that
  shows the false-accept / false-reject trade-off.

Outputs: results/roc_curve.png, similarity_distribution.png, far_frr_curve.png,
         evaluation_results.json ("pair_verification" section)

Usage
    python roc_analysis.py
"""
import argparse
import csv

import matplotlib
import numpy as np
from sklearn.metrics import auc, roc_curve

from common import RESULTS_DIR, update_results_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_scores(split):
    path = RESULTS_DIR / f"{split}_pair_scores.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run evaluate_pairs.py first.")
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return np.array([int(r["label"]) for r in rows]), np.array([float(r["cosine_similarity"]) for r in rows])


def equal_error_rate(fpr, tpr, thr):
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def tar_at_far(fpr, tpr, thr, far):
    i = int(np.where(fpr <= far)[0][-1])
    return {"far_target": far, "tar": float(tpr[i]), "actual_far": float(fpr[i]),
            "threshold": float(thr[i]) if np.isfinite(thr[i]) else None}


def metrics_at(y, s, t):
    pred = s >= t
    tp = int(np.sum(pred & (y == 1))); fn = int(np.sum(~pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0))); tn = int(np.sum(~pred & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": t, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / len(y), "precision": precision, "recall_tar": recall, "f1": f1,
            "far": fp / (fp + tn), "frr": fn / (fn + tp)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args()

    yv, sv = load_scores("val")
    yt, st = load_scores("test")

    # ---- threshold selection on validation
    fpr_v, tpr_v, thr_v = roc_curve(yv, sv)
    j = int(np.argmax(tpr_v - fpr_v))
    threshold = round(float(thr_v[j]), 2)
    val_eer, val_eer_thr = equal_error_rate(fpr_v, tpr_v, thr_v)

    # ---- test ROC
    fpr, tpr, thr = roc_curve(yt, st)
    roc_auc = float(auc(fpr, tpr))
    test_eer, test_eer_thr = equal_error_rate(fpr, tpr, thr)
    at_thr = metrics_at(yt, st, threshold)
    sweep = [{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics_at(yt, st, round(t, 2)).items()
              if k in ("threshold", "far", "frr", "accuracy")} for t in np.arange(0.0, 0.951, 0.05)]

    results = {
        "test_pairs": {"positives": int(yt.sum()), "negatives": int((yt == 0).sum())},
        "val_pairs": {"positives": int(yv.sum()), "negatives": int((yv == 0).sum())},
        "threshold": threshold,
        "threshold_selection": "max Youden J (TPR-FPR) on validation pairs, rounded to 2 decimals",
        "roc_auc_test": roc_auc,
        "roc_auc_val": float(auc(fpr_v, tpr_v)),
        "eer_test": test_eer, "eer_threshold_test": test_eer_thr,
        "eer_val": val_eer, "eer_threshold_val": val_eer_thr,
        "tar_at_far_1pct": tar_at_far(fpr, tpr, thr, 0.01),
        "tar_at_far_0.1pct": tar_at_far(fpr, tpr, thr, 0.001),
        "tar_at_far_note": f"FAR=0.1% corresponds to ~{(yt == 0).sum() * 0.001:.0f} false accepts out of "
                           f"{(yt == 0).sum()} impostor pairs, so that estimate is noisy.",
        "test_metrics_at_threshold": at_thr,
        "val_metrics_at_threshold": metrics_at(yv, sv, threshold),
        "genuine_similarity_test": {"mean": float(st[yt == 1].mean()), "std": float(st[yt == 1].std())},
        "impostor_similarity_test": {"mean": float(st[yt == 0].mean()), "std": float(st[yt == 0].std())},
        "threshold_sweep_test": sweep,
    }
    update_results_json("pair_verification", results)

    # ---- plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, logx in zip(ax, (False, True)):
        a.plot(fpr, tpr, lw=2, label=f"test  AUC = {roc_auc:.4f}")
        a.plot(fpr_v, tpr_v, lw=1.2, ls="--", label=f"val   AUC = {results['roc_auc_val']:.4f}")
        a.scatter([at_thr["far"]], [at_thr["recall_tar"]], c="red", zorder=5,
                  label=f"threshold {threshold:.2f}: TPR {at_thr['recall_tar']:.3f}, FPR {at_thr['far']:.3f}")
        a.set_xlabel("False Positive Rate (FAR)"); a.set_ylabel("True Positive Rate (TAR)")
        a.grid(alpha=0.3, which="both")
        if logx:
            a.set_xscale("log"); a.set_xlim(1e-4, 1); a.set_title("ROC (log FAR)")
        else:
            a.plot([0, 1], [0, 1], c="grey", lw=0.8, ls=":"); a.set_title("ROC curve - face verification")
        a.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "roc_curve.png", dpi=130); plt.close(fig)

    fig, a = plt.subplots(figsize=(8, 4.6))
    bins = np.linspace(min(st.min(), -0.2), 1.0, 70)
    a.hist(st[yt == 1], bins=bins, alpha=0.6, density=True, label=f"genuine (same person), n={int(yt.sum())}")
    a.hist(st[yt == 0], bins=bins, alpha=0.6, density=True, label=f"impostor (different), n={int((yt == 0).sum())}")
    a.axvline(threshold, c="k", ls="--", label=f"threshold = {threshold:.2f}")
    a.set_xlabel("cosine similarity"); a.set_ylabel("density"); a.set_title("Test-pair similarity distribution")
    a.legend(); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "similarity_distribution.png", dpi=130); plt.close(fig)

    ts = np.linspace(-0.2, 1.0, 241)
    far = [np.mean(st[yt == 0] >= t) for t in ts]
    frr = [np.mean(st[yt == 1] < t) for t in ts]
    fig, a = plt.subplots(figsize=(8, 4.4))
    a.plot(ts, far, label="FAR (false accepts)"); a.plot(ts, frr, label="FRR (false rejects)")
    a.axvline(threshold, c="k", ls="--", label=f"chosen threshold {threshold:.2f}")
    a.scatter([test_eer_thr], [test_eer], c="red", zorder=5, label=f"EER {test_eer:.3f} @ {test_eer_thr:.2f}")
    a.set_xlabel("cosine-similarity threshold"); a.set_ylabel("error rate"); a.grid(alpha=0.3); a.legend()
    a.set_title("Raising the threshold trades false accepts for false rejects (test)")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "far_frr_curve.png", dpi=130); plt.close(fig)

    m = at_thr
    print(f"Threshold (chosen on val) = {threshold:.2f}")
    print(f"Test ROC-AUC = {roc_auc:.4f}   EER = {test_eer:.4f} (thr {test_eer_thr:.3f})")
    print(f"TAR@FAR=1%   = {results['tar_at_far_1pct']['tar']:.4f}   "
          f"TAR@FAR=0.1% = {results['tar_at_far_0.1pct']['tar']:.4f}")
    print(f"At threshold on test: acc {m['accuracy']:.4f}  precision {m['precision']:.4f}  recall {m['recall_tar']:.4f}  "
          f"F1 {m['f1']:.4f}  FAR {m['far']:.4f}  FRR {m['frr']:.4f}")
    print("\nthreshold   FAR     FRR")
    for r in sweep[::2]:
        print(f"   {r['threshold']:.2f}   {r['far']:.4f}  {r['frr']:.4f}")


if __name__ == "__main__":
    main()
