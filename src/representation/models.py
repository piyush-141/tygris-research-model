"""
Representation Learning Architectures
Faithfully implements ConvNeXt-small (Paper Selected Default) and comparative models.
Reference: Ma et al. (2025) Section 12.
Important paper finding: Complete backbone is fine-tuned without freezing lower layers.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class TigerRepresentationNet(nn.Module):
    """
    [PAPER-SPECIFIED REPRESENTATION NETWORK]
    Uses ConvNeXt-small (or comparative backbones) fine-tuned completely for tiger identity classification.
    """
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "ConvNeXt-small",
        pretrained: bool = True,
        freeze_backbone: bool = False # [PAPER SPECIFIED: Default false, entire network fine-tuned]
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # Instantiate backbone
        name_clean = backbone_name.lower().replace("-", "").replace("_", "")
        if "convnext" in name_clean:
            weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.convnext_small(weights=weights)
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Identity()
            self.feature_dim = in_features
        elif "resnet50" in name_clean:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.feature_dim = in_features
        elif "resnext50" in name_clean:
            weights = models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnext50_32x4d(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.feature_dim = in_features
        elif "efficientnet" in name_clean:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Identity()
            self.feature_dim = in_features
        elif "swin" in name_clean:
            weights = models.Swin_S_Weights.DEFAULT if pretrained else None
            self.backbone = models.swin_s(weights=weights)
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
            self.feature_dim = in_features
        else:
            # Default to ConvNeXt-small
            weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.convnext_small(weights=weights)
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Identity()
            self.feature_dim = in_features

        # Freeze/unfreeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Identity classification head
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts pooled feature vector before classification head."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits


def get_representation_model(
    num_classes: int,
    name: str = "ConvNeXt-small",
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> TigerRepresentationNet:
    """Factory function for tiger representation learning models."""
    return TigerRepresentationNet(
        num_classes=num_classes,
        backbone_name=name,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
