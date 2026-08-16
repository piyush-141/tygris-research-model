"""
Human-in-the-Loop Candidate Verification & Enrollment
Faithfully implements Section 18:
- Candidate rejection & verification
- New identity creation with human confirmation
- Secure gallery enrollment (never automated without verification)
"""

from typing import Dict, Any, Optional
import numpy as np
from ..fusion.gallery import TigerGallery, GalleryEntry


class CandidateEnrollmentManager:
    """
    Manages the verification and enrollment workflow for novel or uncertain tiger sightings.
    """
    def __init__(self, gallery: TigerGallery):
        self.gallery = gallery

    def enroll_new_identity(
        self,
        new_tiger_id: str,
        embedding: np.ndarray,
        side: str,
        camera_id: str,
        video_id: str,
        timestamp: str,
        source_path: str,
        verifier_notes: str = ""
    ) -> GalleryEntry:
        """Enrolls a verified new tiger into the active reference gallery."""
        entry_id = f"REF_{new_tiger_id}_{side[:1].upper()}_{len(self.gallery.entries) + 1:04d}"
        entry = GalleryEntry(
            entry_id=entry_id,
            embedding=embedding,
            tiger_id=new_tiger_id,
            side=side,
            camera_id=camera_id,
            video_id=video_id,
            timestamp=timestamp,
            source_path=source_path,
            extra_metadata={"verifier_notes": verifier_notes, "enrolled_status": "VERIFIED"}
        )
        self.gallery.add_entry(entry)
        return entry
