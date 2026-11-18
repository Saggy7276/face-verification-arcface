# Model checkpoint

The trained model (`best_model.pth`, ~99 MB) is too large to keep in git, so it is published as a
GitHub Release asset.

**Download:** [best_model.pth (v1.0)](https://github.com/Saggy7276/face-verification-arcface/releases/download/v1.0/best_model.pth)
and save it in this folder as `checkpoints/best_model.pth`.

Or from the command line:

```bash
curl -L -o checkpoints/best_model.pth https://github.com/Saggy7276/face-verification-arcface/releases/download/v1.0/best_model.pth
```

| | |
|---|---|
| Architecture | ResNet50 → 512-D L2-normalised embedding |
| Training | ArcFace (m = 0.30, s = 30), 30 epochs, best epoch 28 |
| Validation AUC | 0.9150 |
| Test ROC-AUC / EER | 0.9422 / 0.1307 |
| Cosine threshold | 0.13 |

To train your own instead, run `python train.py`; it writes the file here.
