"""
Persistent Multi-Embedding Tiger Gallery
Faithfully implements Section 14 & 15 of the paper:
- Stores 64-D embeddings with full provenance: tiger_id, side (Left/Right), camera, video, timestamp, source_path
- Maintains multiple reference embeddings per tiger (never collapsed into single vector)
- Supports Left/Right side indexing
"""

import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


class GalleryEntry:
    """Represents a single registered reference crop and its 64-D embedding."""
    def __init__(
        self,
        entry_id: str,
        embedding: np.ndarray,
        tiger_id: str,
        side: str,
        camera_id: str,
        video_id: str,
        timestamp: str,
        source_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None
    ):
        self.entry_id = entry_id
        self.embedding = np.asarray(embedding, dtype=np.float32).flatten()
        self.tiger_id = str(tiger_id)
        self.side = str(side)
        self.camera_id = str(camera_id)
        self.video_id = str(video_id)
        self.timestamp = str(timestamp)
        self.source_path = str(source_path)
        self.extra_metadata = extra_metadata or {}

    @property
    def side_aware_id(self) -> str:
        """[PAPER-SPECIFIED]: TIGER_ID + SIDE representation."""
        return f"{self.tiger_id}_{self.side.upper()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "embedding": self.embedding.tolist(),
            "tiger_id": self.tiger_id,
            "side": self.side,
            "camera_id": self.camera_id,
            "video_id": self.video_id,
            "timestamp": self.timestamp,
            "source_path": self.source_path,
            "side_aware_id": self.side_aware_id,
            "extra_metadata": self.extra_metadata
        }


class TigerGallery:
    """
    [PAPER-SPECIFIED GALLERY ENGINE]
    Maintains multiple reference embeddings per individual with full side/view information.
    """
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.entries: List[GalleryEntry] = []
        self._embeddings_matrix: Optional[np.ndarray] = None

    def add_entry(self, entry: GalleryEntry):
        assert len(entry.embedding) == self.embedding_dim, f"Expected {self.embedding_dim}-D, got {len(entry.embedding)}"
        self.entries.append(entry)
        self._embeddings_matrix = None # Invalidate cache

    def build_matrix(self) -> np.ndarray:
        if self._embeddings_matrix is None or len(self._embeddings_matrix) != len(self.entries):
            if not self.entries:
                self._embeddings_matrix = np.empty((0, self.embedding_dim), dtype=np.float32)
            else:
                self._embeddings_matrix = np.vstack([e.embedding for e in self.entries])
        return self._embeddings_matrix

    def get_identities(self) -> List[str]:
        return sorted(list(set(e.tiger_id for e in self.entries)))

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = [e.to_dict() for e in self.entries]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            return
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.entries = []
        for d in data:
            entry = GalleryEntry(
                entry_id=d["entry_id"],
                embedding=np.array(d["embedding"], dtype=np.float32),
                tiger_id=d["tiger_id"],
                side=d["side"],
                camera_id=d["camera_id"],
                video_id=d["video_id"],
                timestamp=d["timestamp"],
                source_path=d["source_path"],
                extra_metadata=d.get("extra_metadata", {})
            )
            self.entries.append(entry)
        self._embeddings_matrix = None
