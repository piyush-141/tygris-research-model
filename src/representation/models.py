"""
Representation Learning Architectures — Accuracy-Maximised
Implements ConvNeXt-small with:
- ArcFace / Additive Angular Margin classification head (major accuracy booster for Re-ID)
- BN-Neck: batch-normalised feature neck before the classifier head (industry standard for Re-ID)
- Dropout before classifier for regularisation
- Safe single-sample handling for batchnorm
- extract_features() returns BN-neck features for metric branch compatibility
Reference: Ma et al. (2025) Section 12.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional
import math


class ArcFaceHead(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss head.
    Produces normalised feature × normalised weight cosine similarity
    scaled by s and shifted by margin m in the angle domain.
    Reference: Deng et al., CVPR 2019.
    """
    def __init__(self, in_features: int, num_classes: int, s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat_norm = F.normalize(features, p=2, dim=1)
        W_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(feat_norm, W_norm)

        if labels is None or not self.training:
            return cosine * self.s

        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, min=1e-6))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class BNNeck(nn.Module):
    """
    Batch-Normalisation Neck (BN-Neck).
    Handles single-sample forward passes safely without BatchNorm variance error.
    """
    def __init__(self, feat_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(feat_dim)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)
        self.bn.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.size(0) == 1:
            return F.batch_norm(
                x,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                training=False,
                eps=self.bn.eps
            )
        return self.bn(x)


class TigerRepresentationNet(nn.Module):
    """
    [ACCURACY-MAXIMISED REPRESENTATION NETWORK]
    ConvNeXt-small backbone + BN-Neck + ArcFace head.
    """
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "ConvNeXt-small",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_arcface: bool = True,
        arcface_s: float = 30.0,
        arcface_m: float = 0.50,
        dropout_p: float = 0.20
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.use_arcface = use_arcface

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
        elif "resnext50" in name_clean:
            weights = models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnext50_32x4d(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif "efficientnet" in name_clean:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Identity()
        elif "swin" in name_clean:
            weights = models.Swin_S_Weights.DEFAULT if pretrained else None
            self.backbone = models.swin_s(weights=weights)
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.convnext_small(weights=weights)
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Identity()

        self.feature_dim = in_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p=dropout_p)
        self.bn_neck = BNNeck(in_features)

        if use_arcface:
            self.classifier = ArcFaceHead(in_features, num_classes, s=arcface_s, m=arcface_m)
        else:
            self.classifier = nn.Linear(in_features, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return feat

    def extract_bn_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.bn_neck(feat)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.extract_features(x)
        bn_feat = self.bn_neck(feat)
        if self.use_arcface:
            return self.classifier(bn_feat, labels)
        return self.classifier(bn_feat)


def get_representation_model(
    num_classes: int,
    name: str = "ConvNeXt-small",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    use_arcface: bool = True
) -> TigerRepresentationNet:
    """Factory function for tiger representation learning models."""
    return TigerRepresentationNet(
        num_classes=num_classes,
        backbone_name=name,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        use_arcface=use_arcface
    )
