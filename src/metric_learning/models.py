"""
Metric Learning Architectures
Faithfully implements ConvNeXt-small (Paper-Selected Default) with 64-D MLP projection head.
Reference: Ma et al. (2025) Section 13.
Architecture: Backbone -> remove classifier -> MLP (Linear -> GELU -> Linear -> 64-D embedding).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional


class TigerMetricNet(nn.Module):
    """
    [PAPER-SPECIFIED METRIC LEARNING ARCHITECTURE]
    ConvNeXt-small backbone with MLP projection head producing normalized 64-D embeddings.
    """
    def __init__(
        self,
        backbone_name: str = "ConvNeXt-small",
        embedding_dim: int = 64, # [PAPER-SPECIFIED: 64-D]
        mlp_hidden_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False
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

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # [PAPER-SPECIFIED MLP PROJECTION HEAD: Linear -> GELU -> Linear -> 64-D]
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embedding_dim)
        )

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
