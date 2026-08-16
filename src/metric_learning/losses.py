"""
Metric Learning Loss Functions
Implements Multi-Similarity Loss and Triplet Margin Loss for deep metric embedding learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSimilarityLoss(nn.Module):
    """
    Multi-Similarity Loss for deep metric learning.
    Reference: Wang et al., "Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning", CVPR 2019.
    """
    def __init__(self, alpha: float = 2.0, beta: float = 50.0, base: float = 0.5, margin: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Cosine similarity matrix
        sim_mat = torch.matmul(embeddings, embeddings.t())
        
        loss = []
        batch_size = embeddings.size(0)

        for i in range(batch_size):
            pos_mask = (labels == labels[i]) & (torch.arange(batch_size, device=labels.device) != i)
            neg_mask = labels != labels[i]

            pos_sim = sim_mat[i][pos_mask]
            neg_sim = sim_mat[i][neg_mask]

            if len(pos_sim) == 0 or len(neg_sim) == 0:
                continue

            # Mining informative pairs
            neg_mining_thresh = pos_sim.min().item() + self.margin
            pos_mining_thresh = neg_sim.max().item() - self.margin

            hard_pos = pos_sim[pos_sim < pos_mining_thresh] if (pos_sim < pos_mining_thresh).any() else pos_sim
            hard_neg = neg_sim[neg_sim > neg_mining_thresh] if (neg_sim > neg_mining_thresh).any() else neg_sim

            pos_loss = (1.0 / self.alpha) * torch.log(1.0 + torch.sum(torch.exp(-self.alpha * (hard_pos - self.base))))
            neg_loss = (1.0 / self.beta) * torch.log(1.0 + torch.sum(torch.exp(self.beta * (hard_neg - self.base))))

            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.tensor(0.0, requires_grad=True, device=embeddings.device)
        return torch.mean(torch.stack(loss))


class HardTripletMarginLoss(nn.Module):
    """Batch-hard triplet margin loss with Euclidean distance."""
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Pairwise Euclidean distance
        diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)
        dist_mat = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-12)

        batch_size = embeddings.size(0)
        loss = []
        for i in range(batch_size):
            pos_mask = (labels == labels[i]) & (torch.arange(batch_size, device=labels.device) != i)
            neg_mask = labels != labels[i]

            if not pos_mask.any() or not neg_mask.any():
                continue

            hardest_pos = dist_mat[i][pos_mask].max()
            hardest_neg = dist_mat[i][neg_mask].min()

            triplet_loss = F.relu(hardest_pos - hardest_neg + self.margin)
            loss.append(triplet_loss)

        if len(loss) == 0:
            return torch.tensor(0.0, requires_grad=True, device=embeddings.device)
        return torch.mean(torch.stack(loss))
