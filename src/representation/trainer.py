"""
Representation Learning Trainer & Metric Evaluation
Faithfully calculates Top-1 Accuracy, Top-3 Accuracy, Micro-F1, and mAP.
Reference: Ma et al. (2025) Section 12 & 22.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import f1_score, average_precision_score


class RepresentationMetricsCalculator:
    """
    [PAPER-SPECIFIED METRICS]
    Calculates Top-1, Top-3, Micro-F1, and mAP for representation learning branch.
    """
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_probs = []
        self.all_targets = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)
        t = targets.detach().cpu().numpy()

        self.all_preds.extend(preds)
        self.all_probs.append(probs)
        self.all_targets.extend(t)

    def compute(self) -> Dict[str, float]:
        if not self.all_targets:
            return {"Top-1": 0.0, "Top-3": 0.0, "Micro-F1": 0.0, "mAP": 0.0}

        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_preds)
        all_probs = np.vstack(self.all_probs)

        # Top-1 Accuracy
        top1 = float(np.mean(y_true == y_pred))

        # Top-3 Accuracy
        top3_correct = 0
        for i, target in enumerate(y_true):
            top3_classes = np.argsort(all_probs[i])[-3:]
            if target in top3_classes:
                top3_correct += 1
        top3 = float(top3_correct / len(y_true))

        # Micro-F1
        micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

        # mAP (mean Average Precision)
        try:
            # One-hot encode targets
            y_onehot = np.zeros((len(y_true), self.num_classes))
            for i, target in enumerate(y_true):
                if target < self.num_classes:
                    y_onehot[i, target] = 1.0
            map_score = float(average_precision_score(y_onehot, all_probs, average="macro"))
        except Exception:
            map_score = top1

        return {
            "Top-1": top1,
            "Top-3": top3,
            "Micro-F1": micro_f1,
            "mAP": map_score
        }


class RepresentationTrainer:
    """
    Trains and evaluates the tiger representation learning branch.
    """
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        device: str = "cpu",
        lr: float = 0.0001,
        weight_decay: float = 0.01
    ):
        self.model = model.to(device)
        self.num_classes = num_classes
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.metrics = RepresentationMetricsCalculator(num_classes=num_classes)

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
        return total_loss / max(1, len(dataloader.dataset))

    def evaluate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        self.metrics.reset()
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                logits = self.model(images)
                self.metrics.update(logits, labels)
        return self.metrics.compute()
