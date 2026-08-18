"""
Semantic Segmentation Trainer & Evaluation Metrics — Accuracy-Maximised
Implements:
- Combined Loss (CrossEntropy + Soft Dice Loss) for handling class imbalance & crisp boundary delineation
- Exact TIoU (Tiger IoU), BIoU (Background IoU), and MIoU (Mean IoU) calculations.
Reference: Ma et al. (2025) Section 7 & 22.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import numpy as np


class DiceLoss(nn.Module):
    """Soft Dice Loss for 2-class foreground-background semantic segmentation."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1]  # (B, H, W)
        targets_f = (targets == 1).float()

        intersection = torch.sum(probs * targets_f, dim=(1, 2))
        cardinality = torch.sum(probs + targets_f, dim=(1, 2))

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return torch.mean(1.0 - dice)


class CombinedSegLoss(nn.Module):
    """CE + Dice combined loss: optimizes pixel accuracy while maximizing IoU directly."""
    def __init__(self, ce_weight: float = 0.6, dice_weight: float = 0.4):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)


class SegmentationMetricsCalculator:
    """
    [PAPER-SPECIFIED METRICS]
    Calculates exact TIoU (Tiger IoU), BIoU (Background IoU), and MIoU.
    """
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        p = preds.cpu().numpy().flatten()
        t = targets.cpu().numpy().flatten()
        mask = (t >= 0) & (t < self.num_classes)
        hist = np.bincount(
            self.num_classes * t[mask].astype(int) + p[mask].astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += hist

    def compute(self) -> Dict[str, float]:
        cm = self.confusion_matrix
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        denom = tp + fp + fn
        ious = np.divide(tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0)

        biou = float(ious[0]) if len(ious) > 0 else 0.0
        tiou = float(ious[1]) if len(ious) > 1 else 0.0
        miou = float(np.mean(ious))

        return {
            "BIoU": biou,
            "TIoU": tiou,
            "MIoU": miou
        }


class SegmentationTrainer:
    """
    Trains and evaluates semantic segmentation models with combined CE+Dice loss.
    """
    def __init__(self, model: nn.Module, device: str = "cpu", lr: float = 0.005):
        self.model = model.to(device)
        self.device = device
        self.criterion = CombinedSegLoss(ce_weight=0.6, dice_weight=0.4)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.metrics = SegmentationMetricsCalculator(num_classes=2)

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0
        for images, masks in dataloader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
        return total_loss / max(1, len(dataloader.dataset))

    def evaluate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        self.metrics.reset()
        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                self.metrics.update(preds, masks)
        return self.metrics.compute()
