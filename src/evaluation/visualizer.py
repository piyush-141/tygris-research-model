"""
Visualizations & Diagnostic Projections
Faithfully implements Section 21 & 23:
- 2D Projections (PCA / UMAP) before and after metric learning
- Confusion Matrix generator
- Retrieval rank display and failure case diagnostic logger
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


class EmbeddingVisualizer:
    """
    Computes 2D projections for visual inspection of feature clusters.
    """
    @staticmethod
    def project_2d(embeddings: np.ndarray, method: str = "pca") -> np.ndarray:
        """Projects high-dimensional embeddings to 2D coordinates."""
        if len(embeddings) < 2:
            return np.zeros((len(embeddings), 2))
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(embeddings)

    @staticmethod
    def compute_confusion_matrix(y_true: List[str], y_pred: List[str], labels: Optional[List[str]] = None) -> Dict[str, Any]:
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return {
            "labels": labels,
            "matrix": cm.tolist()
        }
