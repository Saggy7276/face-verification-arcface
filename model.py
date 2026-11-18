"""ResNet face-embedding model and ArcFace training head.

    Face image (3x112x112)
        -> ResNet backbone (ImageNet-pretrained, classifier removed) -> pooled features
        -> embedding layer: Dropout -> Linear(features -> 512) -> BatchNorm1d
        -> L2 normalisation -> 512-D unit-length embedding

The ArcMarginHead is used only during training. At inference only the embedding is used,
so cosine similarity is always computed on embeddings, never on classifier logits.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

ARCHS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


class FaceEmbeddingNet(nn.Module):
    def __init__(self, arch: str = "resnet50", embedding_dim: int = 512, pretrained: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"arch must be one of {list(ARCHS)}")
        build, weights = ARCHS[arch]
        backbone = build(weights=weights if pretrained else None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()           # drop the ImageNet classifier
        self.backbone = backbone
        self.embedding = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )
        self.arch = arch
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        e = self.embedding(self.backbone(x))
        return F.normalize(e, dim=1) if normalize else e


class ArcMarginHead(nn.Module):
    """Additive angular margin softmax (ArcFace, Deng et al. CVPR 2019).

    logits_j = s * cos(theta_j)            for j != y
    logits_y = s * cos(theta_y + m)        for the true class

    With margin = 0 this reduces to a normalised-softmax (cosine) classifier.
    """

    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin                  # may be changed during training (margin warm-up)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        with torch.autocast(device_type=emb.device.type, enabled=False):
            cosine = F.linear(F.normalize(emb.float(), dim=1), F.normalize(self.weight.float(), dim=1))
            if labels is None or self.margin == 0:
                return self.scale * cosine
            m = self.margin
            sine = torch.sqrt((1.0 - cosine ** 2).clamp(1e-7, 1))      # eps keeps the sqrt gradient finite
            phi = cosine * math.cos(m) - sine * math.sin(m)            # cos(theta + m)
            # keep the target logit monotonic in theta when theta + m > pi
            phi = torch.where(cosine > math.cos(math.pi - m), phi, cosine - math.sin(math.pi - m) * m)
            one_hot = F.one_hot(labels, cosine.size(1)).to(cosine.dtype)
            return self.scale * (one_hot * phi + (1.0 - one_hot) * cosine)


if __name__ == "__main__":
    net = FaceEmbeddingNet("resnet18", pretrained=False)
    z = net(torch.randn(4, 3, 112, 112))
    print(z.shape, z.norm(dim=1))
