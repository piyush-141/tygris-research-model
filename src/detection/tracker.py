"""
Multi-Tiger Object Tracker for Video Event Screening
Associates temporary instance detections across consecutive frames into continuous tracks.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
from .animal_detector import DetectionResult


@dataclass
class TrackFrame:
    frame_idx: int
    timestamp_offset: float # in seconds
    bbox: List[int] # [x_min, y_min, x_max, y_max]
    confidence: float
    image: Optional[Image.Image] = None # Cached temporary candidate frame
    quality_score: float = 0.0
    pose_info: Optional[Dict[str, Any]] = None


@dataclass
class TigerTrack:
    track_id: str # e.g. "Track 1", "Track 2"
    event_tiger_id: str # e.g. "EVENT_TIGER_001"
    frames: List[TrackFrame] = field(default_factory=list)
    is_active: bool = True
    start_frame: int = 0
    end_frame: int = 0
    best_frame: Optional[TrackFrame] = None
    selected_frames: List[TrackFrame] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def mean_confidence(self) -> float:
        if not self.frames:
            return 0.0
        return float(np.mean([f.confidence for f in self.frames]))


def compute_iou(boxA: List[int], boxB: List[int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return float(iou)


class MultiTigerTracker:
    """
    Multi-Object Tracker to assign continuous track IDs across sampled video frames.
    """
    def __init__(self, iou_threshold: float = 0.25, max_lost_frames: int = 5):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.tracks: List[TigerTrack] = []
        self._next_track_idx = 1

    def reset(self):
        self.tracks = []
        self._next_track_idx = 1

    def update(
        self,
        frame_idx: int,
        detections: List[DetectionResult],
        frame_image: Optional[Image.Image] = None,
        fps: float = 3.0
    ) -> List[TigerTrack]:
        """
        Updates tracks with new detections from the current frame.
        """
        ts_offset = frame_idx / max(1.0, fps)
        unmatched_dets = list(detections)

        # Match existing active tracks with current detections via IoU
        for track in self.tracks:
            if not track.is_active:
                continue

            last_frame = track.frames[-1]
            best_iou = 0.0
            best_det_idx = -1

            for idx, det in enumerate(unmatched_dets):
                iou = compute_iou(last_frame.bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = idx

            if best_iou >= self.iou_threshold and best_det_idx >= 0:
                matched_det = unmatched_dets.pop(best_det_idx)
                track_frame = TrackFrame(
                    frame_idx=frame_idx,
                    timestamp_offset=ts_offset,
                    bbox=matched_det.bbox,
                    confidence=matched_det.confidence,
                    image=frame_image
                )
                track.frames.append(track_frame)
                track.end_frame = frame_idx
            else:
                # Check if lost too long
                if frame_idx - track.end_frame > self.max_lost_frames:
                    track.is_active = False

        # Create new tracks for unmatched detections
        for det in unmatched_dets:
            new_track_id = f"Track {self._next_track_idx}"
            event_tiger_id = f"EVENT_TIGER_{self._next_track_idx:03d}"
            self._next_track_idx += 1

            track_frame = TrackFrame(
                frame_idx=frame_idx,
                timestamp_offset=ts_offset,
                bbox=det.bbox,
                confidence=det.confidence,
                image=frame_image
            )
            new_track = TigerTrack(
                track_id=new_track_id,
                event_tiger_id=event_tiger_id,
                frames=[track_frame],
                is_active=True,
                start_frame=frame_idx,
                end_frame=frame_idx
            )
            self.tracks.append(new_track)

        return self.tracks

    def get_tracks(self, min_frames: int = 1) -> List[TigerTrack]:
        """Returns all completed or persistent tracks exceeding minimum frame length."""
        return [t for t in self.tracks if len(t.frames) >= min_frames]
