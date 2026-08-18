"""
Metric Learning Architectures — Accuracy-Maximised
Implements ConvNeXt-small with:
- 3-layer MLP projection head with LayerNorm + GELU (robust across all batch sizes, no single-sample collapse)
- 64-D L2-normalised embedding
Reference: Ma et al. (2025) Section 13.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional


class TigerMetricNet(nn.Module):
    """
    [ACCURACY-MAXIMISED METRIC LEARNING ARCHITECTURE]
    ConvNeXt-small backbone + 3-layer MLP LayerNorm-GELU projection head -> 64-D L2-normalised embedding.
    """
    def __init__(
        self,
        backbone_name: str = "ConvNeXt-small",
        embedding_dim: int = 64,           # [PAPER-SPECIFIED: 64-D]
        mlp_hidden_dim: int = 512,         # 512-D bottleneck
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_p: float = 0.10
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim

        name_clean = backbone_name.lower().replace("-", "").replace("_", "")

        if "convnext" in name_clean:
            weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.convnext_small(weights=weights)
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Identity()
        elif "resnet50" in name_clean:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.convnext_small(weights=weights)
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Identity()

        self.in_features = in_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 3-layer MLP with LayerNorm (robust to batch sizes 1..N):
        # in_features -> 512 -> 256 -> 64
        mid_dim = mlp_hidden_dim // 2
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(mlp_hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, embedding_dim)
        )

        for m in self.projection_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        feat = self.extract_features(x)
        embed = self.projection_head(feat)
        if normalize:
            embed = F.normalize(embed, p=2, dim=1)
        return embed


def get_metric_model(
    name: str = "ConvNeXt-small",
    embedding_dim: int = 64,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> TigerMetricNet:
    """Factory function for paper metric learning models."""
    return TigerMetricNet(
        backbone_name=name,
        embedding_dim=embedding_dim,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
