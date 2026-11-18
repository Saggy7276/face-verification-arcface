"""Fine-tune an ImageNet ResNet into a 512-D face-embedding model.

Loss: ArcFace (additive angular margin softmax) over the TRAIN identities only.
Model selection: verification ROC-AUC on the held-out VALIDATION identities (all image
pairs among them), computed from cosine similarity of embeddings. Test identities are
never touched here.

Usage
    python train.py                                  # ResNet50, 30 epochs
    python train.py --arch resnet18 --epochs 20 --batch-size 128
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from common import (CKPT_DIR, RESULTS_DIR, FaceDataset, compute_embeddings, get_device, set_seed, split_samples,
                    train_transform)
from model import ArcMarginHead, FaceEmbeddingNet


def validation_metrics(model, device, val_samples):
    """All-pairs verification AUC + leave-one-out Rank-1 on the validation identities."""
    paths = [r for r, _ in val_samples]
    ids = np.array([p for _, p in val_samples])
    emb = compute_embeddings(model, paths, device, flip_tta=False)
    sim = emb @ emb.T
    iu = np.triu_indices(len(paths), k=1)
    same = (ids[:, None] == ids[None, :])[iu]
    auc = roc_auc_score(same.astype(int), sim[iu])
    np.fill_diagonal(sim, -np.inf)
    rank1 = float(np.mean(ids[sim.argmax(1)] == ids))
    return float(auc), rank1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--embedding-dim", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="LR for embedding layer + ArcFace head")
    p.add_argument("--backbone-lr-mult", type=float, default=0.1, help="backbone LR = lr * mult")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--margin", type=float, default=0.3, help="ArcFace angular margin m (0 = plain cosine softmax)")
    p.add_argument("--scale", type=float, default=30.0, help="ArcFace scale s")
    p.add_argument("--margin-warmup-epochs", type=int, default=3, help="ramp margin linearly from 0")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--warmup-epochs", type=int, default=1, help="linear LR warm-up before cosine decay")
    p.add_argument("--no-pretrained", action="store_true", help="train from scratch instead of ImageNet weights")
    p.add_argument("--amp", action="store_true",
                   help="fp16 mixed precision (off by default: it produced NaNs on a GTX 1650 / Turing TU117)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(CKPT_DIR / "best_model.pth"))
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    use_amp = device.type == "cuda" and args.amp
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_samples = split_samples("train")
    val_samples = split_samples("val")
    classes = sorted({pid for _, pid in train_samples})
    cls_idx = {c: i for i, c in enumerate(classes)}
    print(f"Train: {len(classes)} identities / {len(train_samples)} images   "
          f"Val: {len({p for _, p in val_samples})} identities / {len(val_samples)} images")

    train_ds = FaceDataset([r for r, _ in train_samples], [cls_idx[p] for _, p in train_samples], train_transform())
    g = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=g,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)

    model = FaceEmbeddingNet(args.arch, args.embedding_dim, pretrained=not args.no_pretrained,
                             dropout=args.dropout).to(device)
    head = ArcMarginHead(args.embedding_dim, len(classes), scale=args.scale, margin=0.0).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * args.backbone_lr_mult},
        {"params": list(model.embedding.parameters()) + list(head.parameters()), "lr": args.lr},
    ], weight_decay=args.weight_decay)
    steps_per_epoch = len(loader)
    total_steps, warmup_steps = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    history, best_auc, best_epoch = [], -1.0, -1

    auc0, r10 = validation_metrics(model, device, val_samples)
    print(f"epoch  0 | ImageNet features only        | val AUC {auc0:.4f} | val rank-1 {r10:.3f}")
    history.append({"epoch": 0, "loss": None, "train_acc": None, "val_auc": auc0, "val_rank1": r10})

    for epoch in range(1, args.epochs + 1):
        head.margin = args.margin * min(1.0, epoch / max(1, args.margin_warmup_epochs))
        model.train(); head.train()
        t0, tot_loss, correct, seen = time.time(), 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                emb = model(x)
            logits = head(emb, y)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tot_loss += loss.item() * len(y)
            with torch.no_grad():
                correct += (head(emb.detach()).argmax(1) == y).sum().item()   # accuracy without margin
            seen += len(y)

        val_auc, val_r1 = validation_metrics(model, device, val_samples)
        rec = {"epoch": epoch, "loss": tot_loss / seen, "train_acc": correct / seen, "val_auc": val_auc,
               "val_rank1": val_r1, "margin": head.margin, "lr": optimizer.param_groups[1]["lr"]}
        history.append(rec)
        flag = ""
        if val_auc > best_auc:
            best_auc, best_epoch, flag = val_auc, epoch, "  * saved"
            torch.save({"arch": args.arch, "embedding_dim": args.embedding_dim, "model": model.state_dict(),
                        "epoch": epoch, "val_auc": val_auc, "val_rank1": val_r1,
                        "num_train_classes": len(classes), "args": vars(args)}, args.out)
        print(f"epoch {epoch:2d} | loss {rec['loss']:.3f} | train acc {rec['train_acc']:.3f} | m {head.margin:.2f} "
              f"| val AUC {val_auc:.4f} | val rank-1 {val_r1:.3f} | {time.time() - t0:.0f}s{flag}")

    with open(RESULTS_DIR / "training_history.json", "w", encoding="utf-8") as fh:
        json.dump({"args": vars(args), "best_epoch": best_epoch, "best_val_auc": best_auc, "history": history},
                  fh, indent=2)
    plot_history(history, RESULTS_DIR / "training_curves.png")
    print(f"\nBest val AUC {best_auc:.4f} at epoch {best_epoch} -> {args.out}")


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = [r for r in history if r["epoch"] > 0]
    ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(ep, [r["loss"] for r in h]); ax[0].set_title("Train ArcFace loss")
    ax[1].plot(ep, [r["train_acc"] for r in h]); ax[1].set_title("Train identity accuracy")
    ax[2].plot([r["epoch"] for r in history], [r["val_auc"] for r in history], label="val AUC")
    ax[2].plot([r["epoch"] for r in history], [r["val_rank1"] for r in history], label="val rank-1 (LOO)")
    ax[2].set_title("Validation (held-out identities)"); ax[2].legend()
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
