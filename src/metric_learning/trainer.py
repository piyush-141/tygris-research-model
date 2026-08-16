"""
Metric Learning Trainer & Retrieval Metrics Calculator
Faithfully implements evaluation metrics: Precision@1, R-Precision, MAP@R, MRR, AMI.
Reference: Ma et al. (2025) Section 13 & 22.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.cluster import KMeans

from .losses import MultiSimilarityLoss, HardTripletMarginLoss


class MetricRetrievalEvaluator:
    """
    [PAPER-SPECIFIED METRIC EVALUATION]
    Evaluates embeddings with Euclidean distance retrieval.
    Calculates Precision@1, R-Precision, MAP@R, MRR, and AMI.
    """
    def __init__(self, k: int = 7):
        self.k = k

    def evaluate(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        embeddings: (N, 64) normalized float array
        labels: (N,) integer identity array
        """
        N = len(labels)
        if N < 2:
            return {"Precision@1": 0.0, "R-Precision": 0.0, "MAP@R": 0.0, "MRR": 0.0, "AMI": 0.0}

        # Compute pairwise Euclidean distance matrix
        diff = embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
        np.fill_diagonal(dist_mat, np.inf) # Exclude self-match

        prec_at_1_list = []
        r_prec_list = []
        map_r_list = []
        mrr_list = []

        for i in range(N):
            target_label = labels[i]
            sorted_indices = np.argsort(dist_mat[i])
            sorted_labels = labels[sorted_indices]

            # 1. Precision@1
            p1 = 1.0 if sorted_labels[0] == target_label else 0.0
            prec_at_1_list.append(p1)

            # 2. R-Precision (R = number of true positives in gallery for this identity)
            total_positives = np.sum(labels == target_label) - 1
            if total_positives > 0:
                top_r_labels = sorted_labels[:total_positives]
                r_prec = np.sum(top_r_labels == target_label) / total_positives
                r_prec_list.append(r_prec)

                # 3. MAP@R (Mean Average Precision up to rank R)
                pos_ranks = np.where(sorted_labels == target_label)[0] + 1
                precisions_at_pos = [(idx + 1) / rank for idx, rank in enumerate(pos_ranks[:total_positives])]
                map_r = np.sum(precisions_at_pos) / total_positives if precisions_at_pos else 0.0
                map_r_list.append(map_r)
            else:
                r_prec_list.append(0.0)
                map_r_list.append(0.0)

            # 4. MRR (Mean Reciprocal Rank)
            first_match_rank = np.where(sorted_labels == target_label)[0]
            if len(first_match_rank) > 0:
                mrr_list.append(1.0 / (first_match_rank[0] + 1))
            else:
                mrr_list.append(0.0)

        # 5. AMI (Adjusted Mutual Information via K-Means on embeddings)
        unique_classes = len(np.unique(labels))
        try:
            if unique_classes > 1 and N >= unique_classes:
                kmeans = KMeans(n_clusters=unique_classes, random_state=42, n_init="auto")
                cluster_preds = kmeans.fit_predict(embeddings)
                ami_score = float(adjusted_mutual_info_score(labels, cluster_preds))
            else:
                ami_score = 0.0
        except Exception:
            ami_score = 0.0

        return {
            "Precision@1": float(np.mean(prec_at_1_list)),
            "R-Precision": float(np.mean(r_prec_list)),
            "MAP@R": float(np.mean(map_r_list)),
            "MRR": float(np.mean(mrr_list)),
            "AMI": ami_score
        }


class MetricLearningTrainer:
    """
    Trains the metric learning backbone with projection head on tiger identities.
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 0.0001,
        loss_type: str = "MultiSimilarityLoss"
    ):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = MultiSimilarityLoss() if loss_type == "MultiSimilarityLoss" else HardTripletMarginLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.evaluator = MetricRetrievalEvaluator(k=7)

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            embeds = self.model(images, normalize=True)
            loss = self.loss_fn(embeds, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
        return total_loss / max(1, len(dataloader.dataset))

    def evaluate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        all_embeds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                embeds = self.model(images, normalize=True)
                all_embeds.append(embeds.cpu().numpy())
                all_labels.append(labels.numpy())

        if not all_embeds:
            return {}

        embeddings = np.vstack(all_embeds)
        labels = np.concatenate(all_labels)
        return self.evaluator.evaluate(embeddings, labels)
