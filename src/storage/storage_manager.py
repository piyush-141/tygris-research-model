"""
Transactional Storage Manager & Cleanup Engine for Camera-Trap Event Processing
Enforces transactional deletion of temporary frame buffers while preserving raw event video and final tiger sightings.
"""

import os
import shutil
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from PIL import Image


@dataclass
class StorageTelemetry:
    raw_video_frames: int # Total frames in original 30 FPS video
    sampled_screening_frames: int # Pass 1 cheap screening frames (e.g. 2-5 FPS)
    candidate_tiger_frames: int # Frames containing candidate tiger detections
    expensive_model_frames: int # Pass 2 frames sent to DDRNet/ConvNeXt
    retained_final_frames: int # Canonical output images kept permanently
    storage_reduction_pct: float # Percent storage saved vs storing all decoded frames
    intermediate_frames_deleted: int
    raw_bytes_estimate_mb: float
    retained_bytes_estimate_mb: float
    processing_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_video_frames": self.raw_video_frames,
            "sampled_screening_frames": self.sampled_screening_frames,
            "candidate_tiger_frames": self.candidate_tiger_frames,
            "expensive_model_frames": self.expensive_model_frames,
            "retained_final_frames": self.retained_final_frames,
            "storage_reduction_pct": round(self.storage_reduction_pct, 1),
            "intermediate_frames_deleted": self.intermediate_frames_deleted,
            "raw_bytes_estimate_mb": round(self.raw_bytes_estimate_mb, 2),
            "retained_bytes_estimate_mb": round(self.retained_bytes_estimate_mb, 2),
            "processing_time_sec": round(self.processing_time_sec, 3),
        }


class StorageManager:
    """
    Manages temporary buffer directories and transactional cleanup.
    """
    def __init__(self, temp_base_dir: str = "outputs/temp_events", output_dir: str = "outputs/final_sightings"):
        self.temp_base_dir = temp_base_dir
        self.output_dir = output_dir
        os.makedirs(self.temp_base_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def create_event_buffer(self, event_id: str) -> str:
        """Creates a dedicated temporary buffer directory for the event."""
        event_dir = os.path.join(self.temp_base_dir, event_id)
        os.makedirs(event_dir, exist_ok=True)
        return event_dir

    def save_canonical_tiger_frame(
        self,
        event_id: str,
        track_id: str,
        image: Image.Image,
        suffix: str = ""
    ) -> str:
        """Saves a permanent canonical representative frame for a verified tiger track."""
        safe_track = track_id.replace(" ", "_").lower()
        filename = f"{event_id}_{safe_track}{suffix}.jpg"
        out_path = os.path.join(self.output_dir, filename)
        image.save(out_path, format="JPEG", quality=95)
        return out_path

    def transactional_cleanup(
        self,
        event_id: str,
        success: bool,
        telemetry: Optional[StorageTelemetry] = None
    ) -> bool:
        """
        Transactional deletion: Deletes temporary buffer only if success == True.
        If processing failed, retains the temporary evidence directory for inspection.
        """
        event_dir = os.path.join(self.temp_base_dir, event_id)
        if not os.path.exists(event_dir):
            return True

        if success:
            try:
                shutil.rmtree(event_dir)
                return True
            except Exception as e:
                print(f"[StorageManager] Warning: could not delete temporary buffer {event_dir}: {e}")
                return False
        else:
            # Mark processing failed and retain
            failed_tag = os.path.join(event_dir, "PROCESSING_FAILED.txt")
            with open(failed_tag, "w", encoding="utf-8") as f:
                f.write(f"Event {event_id} failed during processing at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"[StorageManager] Retained failed event buffer: {event_dir}")
            return False

    @staticmethod
    def calculate_telemetry(
        raw_frames: int,
        sampled_frames: int,
        candidate_frames: int,
        expensive_frames: int,
        retained_frames: int,
        duration_sec: float
    ) -> StorageTelemetry:
        """Calculates storage reduction percentage and compute savings metrics."""
        # Estimate: average JPEG frame ~0.45 MB
        bytes_per_frame_mb = 0.45
        raw_mb = raw_frames * bytes_per_frame_mb
        retained_mb = retained_frames * bytes_per_frame_mb
        deleted_count = max(0, sampled_frames - retained_frames)
        reduction_pct = ((raw_frames - retained_frames) / max(1, raw_frames)) * 100.0
        reduction_pct = min(99.9, max(0.0, reduction_pct))

        return StorageTelemetry(
            raw_video_frames=raw_frames,
            sampled_screening_frames=sampled_frames,
            candidate_tiger_frames=candidate_frames,
            expensive_model_frames=expensive_frames,
            retained_final_frames=retained_frames,
            storage_reduction_pct=reduction_pct,
            intermediate_frames_deleted=deleted_count,
            raw_bytes_estimate_mb=raw_mb,
            retained_bytes_estimate_mb=retained_mb,
            processing_time_sec=duration_sec
        )
