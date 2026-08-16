"""
Semantic Segmentation Trainer & Evaluation Metrics
Faithfully calculates TIoU (Tiger IoU), BIoU (Background IoU), and MIoU (Mean IoU).
Reference: Ma et al. (2025) Section 7 & 22.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import numpy as np


class SegmentationMetricsCalculator:
    """
    [PAPER-SPECIFIED METRICS]
    Calculates exact TIoU (Tiger IoU), BIoU (Background IoU), and MIoU.
    """
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        # Confusion matrix: rows = true, cols = pred
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds: (N, H, W) integer class predictions (0=bg, 1=tiger)
        targets: (N, H, W) integer ground truth
        """
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

        biou = float(ious[0]) # Background IoU
        tiou = float(ious[1]) if len(ious) > 1 else 0.0 # Tiger IoU
        miou = float(np.mean(ious)) # Mean IoU

        return {
            "BIoU": biou,
            "TIoU": tiou,
            "MIoU": miou
        }


class SegmentationTrainer:
    """
    Trains and evaluates semantic segmentation models on wild tiger images.
    """
    def __init__(self, model: nn.Module, device: str = "cpu", lr: float = 0.005):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
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
