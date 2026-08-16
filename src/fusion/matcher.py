"""
Euclidean 7-NN Retrieval Engine
Faithfully implements Section 14 of the paper:
- Query 64-D embedding compared against gallery using Euclidean distance
- Retrieves top 7 nearest neighbors (k=7)
- Returns neighbor identity, side, distance, rank, and source path
"""

from typing import List, Dict, Any, Optional
import numpy as np
from .gallery import TigerGallery, GalleryEntry


class MetricKNNMatcher:
    """
    [PAPER-SPECIFIED 7-NN EUCLIDEAN MATCHER]
    """
    def __init__(self, gallery: TigerGallery, k: int = 7):
        self.gallery = gallery
        self.k = k

    def match(
        self,
        query_embedding: np.ndarray,
        filter_side: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs Euclidean distance search against the gallery.
        Returns top-k neighbors with rank, distance, identity, and provenance.
        """
        query_vec = np.asarray(query_embedding, dtype=np.float32).flatten()
        assert len(query_vec) == self.gallery.embedding_dim, "Dimension mismatch"

        gallery_matrix = self.gallery.build_matrix()
        if len(gallery_matrix) == 0:
            return []

        # Euclidean distance
        diff = gallery_matrix - query_vec
        distances = np.sqrt(np.sum(diff ** 2, axis=1) + 1e-12)

        # Apply optional side filtering
        if filter_side and filter_side != "Unknown":
            valid_indices = [
                i for i, entry in enumerate(self.gallery.entries)
                if entry.side == filter_side or entry.side == "Unknown"
            ]
            if not valid_indices:
                valid_indices = list(range(len(self.gallery.entries)))
        else:
            valid_indices = list(range(len(self.gallery.entries)))

        sub_distances = distances[valid_indices]
        sorted_sub_order = np.argsort(sub_distances)

        k_eff = min(self.k, len(valid_indices))
        top_k_results = []

        for rank, sub_idx in enumerate(sorted_sub_order[:k_eff]):
            real_idx = valid_indices[sub_idx]
            entry = self.gallery.entries[real_idx]
            dist = float(sub_distances[sub_idx])

            top_k_results.append({
                "rank": rank + 1,
                "tiger_id": entry.tiger_id,
                "side": entry.side,
                "side_aware_id": entry.side_aware_id,
                "distance": dist,
                "entry_id": entry.entry_id,
                "camera_id": entry.camera_id,
                "video_id": entry.video_id,
                "timestamp": entry.timestamp,
                "source_path": entry.source_path
            })

        return top_k_results
