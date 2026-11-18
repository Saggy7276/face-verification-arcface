"""Build report.pdf from the JSON results and figures produced by the pipeline.

Every number in the report is read from data/dataset_stats.json, results/training_history.json
and results/evaluation_results.json, so the report always matches the latest run.

Usage
    python make_report.py
"""
import json

import matplotlib
import matplotlib.image as mpimg

from common import RESULTS_DIR, RESULTS_JSON, ROOT, STATS_PATH

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

A4 = (8.27, 11.69)
LEFT, RIGHT = 0.07, 0.93      # text margins in figure coordinates
FONT_SIZE = 9
LINE_H = 0.0155


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _text_width(fig, renderer, s, cache):
    """Width of string s in figure coordinates, measured with the real font."""
    if s not in cache:
        t = fig.text(0, 0, s, fontsize=FONT_SIZE)
        cache[s] = t.get_window_extent(renderer).width / fig.bbox.width
        t.remove()
    return cache[s]


def justified_paragraph(fig, text, y):
    """Draw text fully justified between LEFT and RIGHT; the last line stays left-aligned. Returns new y."""
    renderer, cache = fig.canvas.get_renderer(), {}
    width = RIGHT - LEFT
    space = _text_width(fig, renderer, "a a", cache) - _text_width(fig, renderer, "aa", cache)
    lines, cur, cur_w = [], [], 0.0
    for word in (w for w in text.split(" ") if w):     # split on spaces only, so " " keeps words together
        w = _text_width(fig, renderer, word, cache)
        if cur and cur_w + space + w > width:
            lines.append(cur)
            cur, cur_w = [word], w
        else:
            cur_w = w if not cur else cur_w + space + w
            cur.append(word)
    if cur:
        lines.append(cur)
    for i, words in enumerate(lines):
        widths = [_text_width(fig, renderer, w, cache) for w in words]
        gap = (width - sum(widths)) / (len(words) - 1) if len(words) > 1 else 0.0
        # Last line, single word, or a line that would need very wide gaps (e.g. before a long URL): left-align.
        if i == len(lines) - 1 or len(words) == 1 or gap > 4 * space:
            fig.text(LEFT, y, " ".join(words), fontsize=FONT_SIZE, va="top")
        else:
            x = LEFT
            for w, wd in zip(words, widths):
                fig.text(x, y, w, fontsize=FONT_SIZE, va="top")
                x += wd + gap
        y -= LINE_H
    return y if lines else y - LINE_H


def page(pdf, title, blocks, images=(), heights=None):
    """blocks: list of str (paragraphs are justified; lines starting with '  ' are kept verbatim in monospace)."""
    fig = plt.figure(figsize=A4)
    fig.text(LEFT, 0.955, title, fontsize=15, weight="bold", va="top")
    y = 0.92
    for block in blocks:
        for raw in block.split("\n"):
            if raw.startswith("  "):
                fig.text(LEFT, y, raw, fontsize=FONT_SIZE, va="top", family="monospace")
                y -= LINE_H
            else:
                if raw.startswith("- "):
                    raw = "- " + raw[2:]                   # keep the bullet attached to its first word
                y = justified_paragraph(fig, raw, y)
        y -= LINE_H * 0.6
    if images:
        avail = y - 0.04
        heights = heights or [1] * len(images)
        tot = sum(heights)
        for img, h in zip(images, heights):
            hh = avail * h / tot
            ax = fig.add_axes([0.06, y - hh, 0.88, hh - 0.01])
            ax.imshow(mpimg.imread(img)); ax.axis("off")
            y -= hh
    pdf.savefig(fig); plt.close(fig)


def main():
    stats = load(STATS_PATH)
    train = load(RESULTS_DIR / "training_history.json")
    ev = load(RESULTS_JSON)
    pv, gp = ev["pair_verification"], ev["gallery_probe"]
    a = train["args"]
    m = pv["test_metrics_at_threshold"]
    sp = stats["splits"]
    fd = stats["face_detection"]

    out = ROOT / "report.pdf"
    with PdfPages(out) as pdf:
        page(pdf, "Face Verification / Re-Identification with a ResNet", [
            "Project report. A ResNet backbone pretrained on ImageNet is fine-tuned with an ArcFace "
            "(additive angular margin) loss to produce 512-D L2-normalised face embeddings. Verification compares "
            "two embeddings by cosine similarity; identification searches a gallery of enrolled identities. "
            "All evaluation uses identities that were never seen in training.",
            "Key results (held-out test identities)",
            f"  Dataset            LFW deep-funneled, {stats['num_identities']} identities, {stats['num_images']} images\n"
            f"  Model              {a['arch']} -> 512-D embedding, ArcFace m={a['margin']}, s={a['scale']}\n"
            f"  Test pairs         {pv['test_pairs']['positives']} genuine + {pv['test_pairs']['negatives']} impostor\n"
            f"  ROC-AUC            {pv['roc_auc_test']:.4f}\n"
            f"  EER                {pv['eer_test']:.4f}\n"
            f"  TAR @ FAR=1%       {pv['tar_at_far_1pct']['tar']:.4f}\n"
            f"  TAR @ FAR=0.1%     {pv['tar_at_far_0.1pct']['tar']:.4f}\n"
            f"  Threshold          {pv['threshold']:.2f}  (chosen on validation identities)\n"
            f"  At threshold       acc {m['accuracy']:.4f}  P {m['precision']:.4f}  R {m['recall_tar']:.4f}  "
            f"F1 {m['f1']:.4f}\n"
            f"                     FAR {m['far']:.4f}  FRR {m['frr']:.4f}\n"
            f"  Gallery/probe      {gp['gallery_identities']} ids x {gp['gallery_images_per_identity']} gallery img, "
            f"{gp['probe_images']} probes\n"
            f"  Rank-1 / Rank-5    {gp['rank1']:.4f} / {gp['rank5']:.4f}   (chance {gp['chance_rank1']:.4f})",
            "Pipeline",
            "  LFW -> Haar face detection -> square crop -> 112x112 -> ImageNet normalisation\n"
            "      -> identity-disjoint train / val / test split\n"
            "      -> ResNet -> Linear+BN -> 512-D -> L2 norm   (ArcFace head only for training)\n"
            "      -> (a) balanced genuine/impostor pairs -> cosine similarity -> ROC/AUC -> threshold\n"
            "      -> (b) gallery templates vs probes     -> cosine ranking   -> Rank-1/5, CMC",
        ])

        page(pdf, "1. Dataset and preprocessing", [
            "Source: Labeled Faces in the Wild (LFW), University of Massachusetts Amherst, deep-funneled version, "
            "downloaded from the Kaggle mirror jessicali9530/lfw-dataset. LFW is a public research benchmark of news "
            "photographs of public figures; it contains no criminal-record or mugshot data. LFW is distributed for "
            "research use without a formal licence; images remain the copyright of their original owners. It is "
            "used here only for non-commercial research evaluation. Official page: http://vis-www.cs.umass.edu/lfw/",
            f"Selection: every identity with at least {stats['selection']['min_images_per_identity']} images, capped "
            f"at {stats['selection']['max_images_per_identity']} per identity (first files in sorted order) so a few "
            f"heavily photographed people do not dominate training. Result: {stats['num_identities']} identities, "
            f"{stats['num_images']} images (min {stats['images_per_identity']['min']}, median "
            f"{stats['images_per_identity']['median']:.0f}, max {stats['images_per_identity']['max']} per identity).",
            f"Split (identity-disjoint, seed {stats['split_seed']}):",
            "\n".join(f"  {s:5s}  {sp[s]['identities']:4d} identities  {sp[s]['images']:5d} images"
                      for s in ("train", "val", "test")),
            f"Face detection: OpenCV Haar cascade; the detection closest to the image centre is kept (LFW labels "
            f"the centred face). Detected in {fd['detected']}/{fd['detected'] + fd['fallback_centre_crop']} images "
            f"({100 * fd['detection_rate']:.1f}%); the rest use a centre crop sized from LFW's known face scale. "
            f"Crop: {stats['preprocessing']['crop']}, resized to {stats['preprocessing']['resize']}. "
            f"Alignment: the deep-funneled images are already aligned by congealing; no extra landmark "
            f"alignment. Normalisation: pixels scaled to [0,1] then ImageNet mean/std. Training "
            f"augmentation: random resized crop (80-100%), horizontal flip, colour jitter, 10% greyscale.",
        ], images=[RESULTS_DIR / "preprocessing_examples.png"])

        best = next(r for r in train["history"] if r["epoch"] == train["best_epoch"])
        page(pdf, "2. Model and training", [
            f"Architecture: torchvision {a['arch']} with ImageNet weights; the 1000-way classifier is replaced by "
            f"Dropout({a['dropout']}) -> Linear(-> {a['embedding_dim']}, no bias) -> BatchNorm1d, followed by L2 "
            f"normalisation. Output: a 512-D unit-length embedding, so cosine similarity = dot product.",
            f"Loss: ArcFace over the {sp['train']['identities']} training identities. Each embedding is "
            f"compared with one learnable unit-length centre per identity; the target-class angle is penalised by an "
            f"additive margin m={a['margin']} (scale s={a['scale']}), ramped up from 0 over "
            f"{a['margin_warmup_epochs']} epochs.",
            "Why ArcFace: (1) it optimises exactly the quantity used at test time, the angle between unit "
            "embeddings, and the margin forces same-person embeddings into tight cones that are separated from "
            "other identities, which carries over to unseen people; (2) unlike contrastive or triplet loss it needs "
            "no pair/triplet mining. Every step compares each sample with all class centres, which is stable and "
            "data-efficient on a small dataset with few images per person; (3) plain softmax cross-entropy "
            "separates the training classes but does not enforce compact clusters, so its embeddings transfer "
            "worse to new identities.",
            f"Optimisation: AdamW, LR {a['lr']} for embedding+head and {a['lr'] * a['backbone_lr_mult']:g} for the "
            f"pretrained backbone, weight decay {a['weight_decay']}, {a['warmup_epochs']}-epoch warm-up then cosine "
            f"decay, batch {a['batch_size']}, {a['epochs']} epochs, "
            f"{'fp16 mixed' if a.get('amp') else 'fp32'} precision, seed {a['seed']}.",
            f"Model selection: after every epoch, verification AUC over all image pairs of the validation "
            f"identities. Best epoch {train['best_epoch']}: val AUC {best['val_auc']:.4f}, val leave-one-out rank-1 "
            f"{best['val_rank1']:.3f}. ImageNet features before fine-tuning gave val AUC "
            f"{train['history'][0]['val_auc']:.4f}.",
        ], images=[RESULTS_DIR / "training_curves.png"])

        page(pdf, "3. Pairs, cosine similarity and ROC", [
            f"Pair generation (test identities only, seed 42): positives = all unordered image combinations per "
            f"identity, shuffled and taken round-robin across identities, so people with many photos do not "
            f"dominate; negatives = random identity pair, then one random image of each. Balanced: "
            f"{pv['test_pairs']['positives']} genuine + {pv['test_pairs']['negatives']} impostor test pairs "
            f"({pv['val_pairs']['positives']} + {pv['val_pairs']['negatives']} on validation). Positives are "
            f"unique by construction and negatives are kept in sorted order in a set; an assertion checks that no "
            f"pair appears twice and no image is paired with itself. Pair identities are asserted disjoint from "
            f"training identities.",
            "Scoring: each unique image is embedded once (embedding of the image plus its mirror, re-normalised); "
            "similarity = A.B / (|A||B|) on embeddings, never on classifier logits.",
            f"Genuine similarity {pv['genuine_similarity_test']['mean']:.3f} +/- "
            f"{pv['genuine_similarity_test']['std']:.3f}, impostor {pv['impostor_similarity_test']['mean']:.3f} "
            f"+/- {pv['impostor_similarity_test']['std']:.3f}. Test ROC-AUC {pv['roc_auc_test']:.4f}, "
            f"EER {pv['eer_test']:.4f}.",
        ], images=[RESULTS_DIR / "roc_curve.png", RESULTS_DIR / "similarity_distribution.png"], heights=[1, 1])

        sweep = "\n".join(f"  {r['threshold']:9.2f}  {r['far']:8.4f}  {r['frr']:8.4f}  {r['accuracy']:8.4f}"
                          for r in pv["threshold_sweep_test"] if 0.05 <= r["threshold"] <= 0.75)
        page(pdf, "4. Threshold analysis", [
            f"Decision rule: similarity >= {pv['threshold']:.2f} -> Match, otherwise Non-Match. The threshold "
            f"maximises Youden's J (TPR - FPR) on the validation pairs and is then fixed for the test set.",
            "Effect of the threshold: raising it accepts fewer pairs, so false accepts (impostors let in, FAR) "
            "fall and false rejects (genuine users refused, FRR) rise; lowering it does the opposite. The ROC curve "
            "is this trade-off traced over every threshold. A security-critical deployment would pick a high "
            "threshold from a TAR@FAR target (e.g. FAR=0.1%); a convenience application would sit nearer the EER.",
            f"At {pv['threshold']:.2f} on test: TP {m['tp']}  FP {m['fp']}  TN {m['tn']}  FN {m['fn']}; "
            f"accuracy {m['accuracy']:.4f}, precision {m['precision']:.4f}, recall/TAR {m['recall_tar']:.4f}, "
            f"F1 {m['f1']:.4f}, FAR {m['far']:.4f}, FRR {m['frr']:.4f}.",
            f"TAR @ FAR=1%: {pv['tar_at_far_1pct']['tar']:.4f} (threshold {pv['tar_at_far_1pct']['threshold']:.3f}); "
            f"TAR @ FAR=0.1%: {pv['tar_at_far_0.1pct']['tar']:.4f} (threshold "
            f"{pv['tar_at_far_0.1pct']['threshold']:.3f}). {pv['tar_at_far_note']}",
            "  threshold       FAR       FRR  accuracy\n" + sweep,
        ], images=[RESULTS_DIR / "far_frr_curve.png"])

        page(pdf, "5. Gallery / Probe identification", [
            f"Gallery: each of the {gp['gallery_identities']} test identities is enrolled with "
            f"{gp['gallery_images_per_identity']} image chosen with a fixed seed (single-shot enrolment). "
            f"With --gallery-per-id k, k embeddings are averaged and re-normalised into one template. Probes: the "
            f"remaining {gp['probe_images']} test images. Each probe embedding is ranked against all gallery "
            f"templates by cosine similarity; the top match is the predicted identity.",
            f"Rank-1 {gp['rank1']:.4f}, Rank-5 {gp['rank5']:.4f}, Rank-10 {gp['rank10']:.4f} (chance "
            f"{gp['chance_rank1']:.4f}); mean rank of the true identity {gp['mean_rank_of_true_identity']:.2f}. "
            f"Per-probe predictions are in results/gallery_probe_results.csv.",
        ], images=[RESULTS_DIR / "cmc_curve.png"])

        page(pdf, "6. Limitations and next steps", [
            "- Small training set (a few thousand images). Production face models train on millions of images "
            "of 10k-100k identities (e.g. MS1M, WebFace), so absolute accuracy here is far below the "
            "state of the art on LFW.",
            "- ImageNet pretraining at 112x112: the backbone was designed for 224x224 object images; a face-specific "
            "stem or a larger input size would help.",
            "- Haar detection with no landmark alignment; a CNN detector with 5-point alignment (e.g. RetinaFace or "
            "MTCNN) would make crops more consistent. LFW's deep funnelling partly compensates.",
            "- LFW bias: mostly public figures in news photos, skewed in age, gender and ethnicity, often frontal "
            "and well lit. Results will not transfer directly to surveillance, low-light or extreme-pose images.",
            "- Statistical precision: TAR@FAR=0.1% is based on only a few impostor pairs; confidence intervals "
            "(bootstrap) and multiple seeds or folds would be needed for firm conclusions.",
            "- Gallery/probe is closed-set; open-set identification (rejecting probes of unknown people with the "
            "threshold) is not evaluated.",
            "- Identity selection takes the first images by filename, so the per-identity cap is not random.",
        ])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
