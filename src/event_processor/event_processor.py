"""
Unified Two-Pass Video Event Processor
Implements the full camera-trap pipeline according to Ma et al. (2025) with production event screening.
"""

import os
import time
import json
import base64
import io
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
import torch

from src.detection.animal_detector import FastAnimalDetector, TigerInstanceDetector, SpeciesConfirmation
from src.detection.tracker import MultiTigerTracker, TigerTrack, TrackFrame
from src.pose.pose_detector import TigerPoseDetector, PoseResult
from src.quality.frame_quality import FrameQualityScorer, DiversityFrameSelector, QualityMetrics
from src.stripes.stripe_analyzer import TigerStripeAnalyzer, StripeAnalysisResult
from src.storage.storage_manager import StorageManager, StorageTelemetry
from src.segmentation.inference import SegmentationPipeline
from src.representation.augmentations import get_paper_reid_transforms
from src.fusion.late_fusion import WeightedLateFusionEngine
from src.fusion.matcher import MetricKNNMatcher
from src.fusion.gallery import TigerGallery
from src.open_world.unknown_detector import OpenWorldDetector
from src.ecology.sighting_db import SightingDatabase


@dataclass
class TigerSighting:
    track_id: str # e.g. "Track 1"
    event_tiger_id: str # e.g. "EVENT_TIGER_001"
    tiger_id: Optional[str] # e.g. "TIG_007" or "160"
    status: str # "KNOWN", "UNKNOWN", "REVIEW_REQUIRED"
    confidence: float
    best_frame_path: str
    best_frame_idx: int
    pose: str
    pose_confidence: float
    keypoints_data: Dict[str, Any]
    quality_score: float
    segmented_crop_b64: str
    mask_b64: str
    stripe_ridge_b64: str
    stripe_density: float
    stripe_match_score: float
    classifier_prediction: str
    classifier_confidence: float
    nearest_neighbors: List[Dict[str, Any]]
    fusion_score: float
    supporting_frames_count: int
    consensus_breakdown: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "event_tiger_id": self.event_tiger_id,
            "tiger_id": self.tiger_id,
            "status": self.status,
            "confidence": round(self.confidence, 4),
            "best_frame_path": self.best_frame_path,
            "best_frame_idx": self.best_frame_idx,
            "pose": self.pose,
            "pose_confidence": round(self.pose_confidence, 4),
            "keypoints_data": self.keypoints_data,
            "quality_score": round(self.quality_score, 2),
            "segmented_crop_b64": self.segmented_crop_b64,
            "mask_b64": self.mask_b64,
            "stripe_ridge_b64": self.stripe_ridge_b64,
            "stripe_density": round(self.stripe_density, 3),
            "stripe_match_score": round(self.stripe_match_score, 4),
            "classifier_prediction": self.classifier_prediction,
            "classifier_confidence": round(self.classifier_confidence, 4),
            "nearest_neighbors": self.nearest_neighbors,
            "fusion_score": round(self.fusion_score, 2),
            "supporting_frames_count": self.supporting_frames_count,
            "consensus_breakdown": self.consensus_breakdown
        }


@dataclass
class EventResult:
    event_id: str
    camera_id: str
    timestamp: str
    latitude: float
    longitude: float
    mode: str # "production" or "research"
    animal_detected: bool
    tiger_detected: bool
    species_label: str
    tiger_count: int
    review_required: bool
    tigers: List[TigerSighting] = field(default_factory=list)
    telemetry: Optional[StorageTelemetry] = None
    tracks_summary: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "mode": self.mode,
            "animal_detected": self.animal_detected,
            "tiger_detected": self.tiger_detected,
            "species_label": self.species_label,
            "tiger_count": self.tiger_count,
            "review_required": self.review_required,
            "tigers": [t.to_dict() for t in self.tigers],
            "telemetry": self.telemetry.to_dict() if self.telemetry else {},
            "tracks_summary": self.tracks_summary
        }


def img_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format, quality=88)
    return f"data:image/{format.lower()};base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


class EventProcessor:
    """
    Two-Pass Camera-Trap Event Processing Engine.
    Executes screening, tracking, quality scoring, pose estimation, DDRNet segmentation,
    stripe ridge analysis, ConvNeXt Re-ID, late fusion, sighting logging, and storage cleanup.
    """
    def __init__(
        self,
        seg_pipeline: SegmentationPipeline,
        rep_model: torch.nn.Module,
        metric_model: torch.nn.Module,
        gallery: TigerGallery,
        sighting_db: SightingDatabase,
        class_mapping: Optional[List[str]] = None,
        device: str = "cpu",
        mode: str = "production"
    ):
        self.device = device
        self.mode = mode
        self.seg_pipeline = seg_pipeline
        self.rep_model = rep_model
        self.metric_model = metric_model
        self.gallery = gallery
        self.sighting_db = sighting_db
        self.class_mapping = class_mapping or sorted(gallery.get_identities())

        # Sub-engines
        self.animal_detector = FastAnimalDetector(confidence_threshold=0.35, device=device)
        self.instance_detector = TigerInstanceDetector(tiger_conf_threshold=0.45, review_threshold=0.30)
        self.tracker = MultiTigerTracker(iou_threshold=0.25, max_lost_frames=5)
        self.quality_scorer = FrameQualityScorer()
        self.frame_selector = DiversityFrameSelector(default_top_k=3, min_frame_distance=2)
        self.pose_detector = TigerPoseDetector(device=device)
        self.stripe_analyzer = TigerStripeAnalyzer()
        self.matcher = MetricKNNMatcher(self.gallery, k=7)
        self.fusion_engine = WeightedLateFusionEngine(conf_threshold=0.80, distance_threshold=0.40)
        self.open_world = OpenWorldDetector(conf_threshold=0.80, dist_threshold=0.40)
        self.storage_mgr = StorageManager()
        self.transform = get_paper_reid_transforms(input_size=(224, 224), is_training=False)

    def process_event(
        self,
        frames: List[Image.Image],
        metadata: Dict[str, Any],
        sampling_fps: float = 3.0,
        raw_fps: float = 30.0,
        mode: Optional[str] = None
    ) -> EventResult:
        """
        Executes Two-Pass Event Processing across incoming video frames or frame sequence.
        """
        start_time = time.time()
        exec_mode = mode or self.mode
        event_id = metadata.get("event_id", f"EVT_{int(time.time())}")
        cam_id = metadata.get("camera_id", "CAM_DEFAULT")
        ts = metadata.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        lat = float(metadata.get("latitude", 21.655))
        lon = float(metadata.get("longitude", 79.312))

        total_input_frames = len(frames)
        # Approximate raw 30 FPS frame count if sampled sequence provided
        estimated_raw_frames = int(total_input_frames * (raw_fps / max(1.0, sampling_fps)))

        # Create temporary storage buffer for this event
        buf_dir = self.storage_mgr.create_event_buffer(event_id)

        # =========================================================================
        # PASS 1: CHEAP EVENT SCREENING & MULTI-OBJECT TRACKING
        # =========================================================================
        self.instance_detector.reset_counter()
        self.tracker.reset()

        animal_seen = False
        tiger_seen = False
        review_needed = False
        sampled_frame_count = 0
        candidate_frame_count = 0

        for frame_idx, frame_img in enumerate(frames):
            sampled_frame_count += 1
            
            # 1. Fast Animal Detection
            has_animal, animal_conf = self.animal_detector.detect_animal(frame_img)
            if not has_animal:
                continue
            animal_seen = True

            # 2. Species Confirmation & Multi-Tiger Instance Bounding Boxes
            spec_res: SpeciesConfirmation = self.instance_detector.detect_instances(frame_img, frame_idx)
            if spec_res.tiger_detected:
                tiger_seen = True
            if spec_res.review_required:
                review_needed = True

            if spec_res.detections:
                candidate_frame_count += len(spec_res.detections)
                # 3. Multi-Object Tracking association
                self.tracker.update(frame_idx, spec_res.detections, frame_image=frame_img, fps=sampling_fps)

        tracks: List[TigerTrack] = self.tracker.get_tracks(min_frames=1)

        # Early termination if no animal or no tiger detected
        if not animal_seen or not tracks:
            duration = time.time() - start_time
            telemetry = StorageManager.calculate_telemetry(
                raw_frames=estimated_raw_frames,
                sampled_frames=sampled_frame_count,
                candidate_frames=0,
                expensive_frames=0,
                retained_frames=0,
                duration_sec=duration
            )
            self.storage_mgr.transactional_cleanup(event_id, success=True, telemetry=telemetry)
            return EventResult(
                event_id=event_id,
                camera_id=cam_id,
                timestamp=ts,
                latitude=lat,
                longitude=lon,
                mode=exec_mode,
                animal_detected=animal_seen,
                tiger_detected=False,
                species_label="no_animal" if not animal_seen else "non_target_wildlife",
                tiger_count=0,
                review_required=review_needed,
                tigers=[],
                telemetry=telemetry,
                tracks_summary=[]
            )

        # =========================================================================
        # PASS 2: EXPENSIVE DEEP ANALYSIS ON SELECTED FRAMES PER TIGER TRACK
        # =========================================================================
        tiger_sightings: List[TigerSighting] = []
        expensive_frames_processed = 0
        tracks_summary: List[Dict[str, Any]] = []

        for track in tracks:
            # 1. Quality Scoring & Diversity-Aware Top-K Frame Selection
            top_k_frames = self.frame_selector.select_best_frames(track, top_k=3)
            expensive_frames_processed += len(top_k_frames)

            best_track_frame = track.best_frame or (top_k_frames[0] if top_k_frames else None)
            if best_track_frame is None or best_track_frame.image is None:
                continue

            best_img = best_track_frame.image

            # 2. DDRNet-39 + YOLO Segmentation & Tight Tiger Crop (run first — all downstream steps depend on it)
            try:
                mask, tiger_only_img, tiger_segmented_crop, seg_bbox = self.seg_pipeline.segment_and_crop(best_img)
            except Exception:
                # Fallback: use full-image predict_mask path
                mask = self.seg_pipeline.predict_mask(best_img)
                tiger_only_img, tiger_segmented_crop, seg_bbox = self.seg_pipeline.extract_tiger_crop(best_img, mask)

            # QA gate: if mask is nearly empty, use bbox crop from detector as fallback
            mask_coverage = float(np.sum(mask)) / max(1, mask.size)
            if mask_coverage < 0.01:
                x1, y1, x2, y2 = best_track_frame.bbox
                tiger_segmented_crop = best_img.crop((x1, y1, x2, y2)).convert("RGB")

            # 3. Pose Estimation on tight tiger crop (not the full frame)
            pose_res: PoseResult = self.pose_detector.estimate_pose(tiger_segmented_crop, bbox=None)

            # 4. Exact Stripe Pattern Ridge Feature Extraction (on tight crop)
            stripe_res: StripeAnalysisResult = self.stripe_analyzer.extract_stripes(tiger_segmented_crop, mask=None)

            # 5. Dual-Branch Re-ID (Run on all selected Top-K frames of this track for consensus)
            track_predictions: List[str] = []
            track_confidences: List[float] = []
            all_track_neighbors: List[List[Any]] = []

            for sel_frame in top_k_frames:
                if sel_frame.image is None:
                    continue
                # Segment & tight-crop each candidate frame using YOLO+DDRNet path
                try:
                    s_mask, _, s_crop, _ = self.seg_pipeline.segment_and_crop(sel_frame.image)
                    s_cov = float(np.sum(s_mask)) / max(1, s_mask.size)
                    if s_cov < 0.01:
                        sx1, sy1, sx2, sy2 = sel_frame.bbox
                        s_crop = sel_frame.image.crop((sx1, sy1, sx2, sy2)).convert("RGB")
                except Exception:
                    s_mask = self.seg_pipeline.predict_mask(sel_frame.image)
                    _, s_crop, _ = self.seg_pipeline.extract_tiger_crop(sel_frame.image, s_mask)

                # Branch A: Representation classification
                tensor_crop = self.transform(s_crop).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.rep_model(tensor_crop)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    pred_idx = int(np.argmax(probs))
                    pred_conf = float(probs[pred_idx])
                    
                    if pred_idx < len(self.class_mapping):
                        mapped_pred_id = str(self.class_mapping[pred_idx])
                    else:
                        mapped_pred_id = f"TIG_{pred_idx + 1:03d}"
                    
                    track_predictions.append(mapped_pred_id)
                    track_confidences.append(pred_conf)

                # Branch B: 64-D Metric Embedding
                with torch.no_grad():
                    emb = self.metric_model(tensor_crop).squeeze(0).cpu().numpy()
                    emb = emb / (np.linalg.norm(emb) + 1e-6)
                
                # 7-NN Euclidean Retrieval
                neighbors = self.matcher.match(emb)
                all_track_neighbors.append(neighbors)

            # 6. Per-Track Weighted Late Fusion Aggregation
            # Aggregate vote points across all selected frames for this tiger track
            candidate_votes: Dict[str, Dict[str, float]] = {}

            for f_idx, (cls_id, cls_conf, neighbors) in enumerate(zip(track_predictions, track_confidences, all_track_neighbors)):
                # Representation branch vote: wr = 1.0 (if conf >= 0.80)
                if cls_conf >= 0.80:
                    if cls_id not in candidate_votes:
                        candidate_votes[cls_id] = {"rep": 0.0, "metric": 0.0, "frames": 0, "min_dist": 99.0}
                    candidate_votes[cls_id]["rep"] += 1.0
                    candidate_votes[cls_id]["frames"] += 1

                # Metric branch votes: wm = 1 / (0.1 + d)
                for rank, n in enumerate(neighbors):
                    nid = str(n.get("tiger_id", "Unknown"))
                    d_val = float(n.get("distance", 1.0))
                    w_m = 1.0 / (0.1 + d_val)
                    if nid not in candidate_votes:
                        candidate_votes[nid] = {"rep": 0.0, "metric": 0.0, "frames": 0, "min_dist": d_val}
                    candidate_votes[nid]["metric"] += w_m
                    candidate_votes[nid]["min_dist"] = min(candidate_votes[nid]["min_dist"], d_val)
                    if rank == 0:
                        candidate_votes[nid]["frames"] += 1

            # Rank consensus candidates
            consensus_list = []
            for cid, scores in candidate_votes.items():
                total_pts = scores["rep"] + scores["metric"]
                consensus_list.append({
                    "tiger_id": cid,
                    "total_score": round(total_pts, 2),
                    "metric_points": round(scores["metric"], 2),
                    "rep_points": round(scores["rep"], 2),
                    "support_count": max(1, scores["frames"]),
                    "best_distance": round(scores["min_dist"], 4) if scores["min_dist"] < 90 else None
                })
            consensus_list.sort(key=lambda x: x["total_score"], reverse=True)

            # Determine Winning Tiger & Status
            primary_cls_id = track_predictions[0] if track_predictions else "Unknown"
            primary_cls_conf = track_confidences[0] if track_confidences else 0.0
            primary_neighbors = all_track_neighbors[0] if all_track_neighbors else []

            enriched_neighbors = []
            for n in primary_neighbors:
                d_val = float(n.get("distance", 1.0))
                enriched_neighbors.append({
                    "tiger_id": str(n.get("tiger_id", "Unknown")),
                    "distance": round(d_val, 4),
                    "side": n.get("side", "Flank"),
                    "weight": round(1.0 / (0.1 + d_val), 2),
                    "camera_id": n.get("camera_id", cam_id),
                    "entry_id": n.get("entry_id", "REF_001")
                })

            winning_tiger_id = consensus_list[0]["tiger_id"] if consensus_list else primary_cls_id
            winning_score = consensus_list[0]["total_score"] if consensus_list else 10.0
            best_knn_dist = enriched_neighbors[0]["distance"] if enriched_neighbors else 0.50

            # Open-World Verification
            is_known = (primary_cls_conf >= 0.80) or (best_knn_dist <= 0.40) or (winning_score >= 12.0)
            status = "KNOWN" if is_known else ("REVIEW_REQUIRED" if (0.70 <= primary_cls_conf < 0.80 or 0.40 < best_knn_dist <= 0.55) else "UNKNOWN")

            # 7. Save Canonical Representative Tiger Frame
            canonical_path = self.storage_mgr.save_canonical_tiger_frame(
                event_id=event_id,
                track_id=track.track_id,
                image=best_img
            )

            # 8. Record in Sighting Database
            if status == "KNOWN" and winning_tiger_id:
                try:
                    self.sighting_db.record_sighting(
                        event_id=f"{event_id}_{track.track_id.replace(' ', '_')}",
                        tiger_id=winning_tiger_id,
                        camera_id=cam_id,
                        latitude=lat,
                        longitude=lon,
                        timestamp=ts,
                        confidence=primary_cls_conf,
                        side="Left" if pose_res.body_angle_deg > 0 else "Right"
                    )
                except Exception as e:
                    print(f"[EventProcessor] Sighting DB log notice: {e}")

            sighting = TigerSighting(
                track_id=track.track_id,
                event_tiger_id=track.event_tiger_id,
                tiger_id=winning_tiger_id if status == "KNOWN" else None,
                status=status,
                confidence=primary_cls_conf,
                best_frame_path=canonical_path,
                best_frame_idx=best_track_frame.frame_idx,
                pose=pose_res.posture,
                pose_confidence=pose_res.pose_confidence,
                keypoints_data=pose_res.to_dict(),
                quality_score=best_track_frame.quality_score,
                segmented_crop_b64=img_to_base64(tiger_segmented_crop),
                mask_b64=img_to_base64(tiger_only_img),
                stripe_ridge_b64=stripe_res.stripe_ridge_b64,
                stripe_density=stripe_res.stripe_density,
                stripe_match_score=stripe_res.stripe_match_score,
                classifier_prediction=primary_cls_id,
                classifier_confidence=primary_conf if (primary_conf := primary_cls_conf) else 0.0,
                nearest_neighbors=enriched_neighbors,
                fusion_score=winning_score,
                supporting_frames_count=len(top_k_frames),
                consensus_breakdown=consensus_list
            )
            tiger_sightings.append(sighting)

            tracks_summary.append({
                "track_id": track.track_id,
                "event_tiger_id": track.event_tiger_id,
                "frame_count": track.num_frames,
                "mean_confidence": round(track.mean_confidence, 3),
                "best_frame_idx": best_track_frame.frame_idx,
                "selected_frame_indices": [f.frame_idx for f in top_k_frames]
            })

        # =========================================================================
        # STORAGE TELEMETRY & TRANSACTIONAL CLEANUP
        # =========================================================================
        duration = time.time() - start_time
        telemetry = StorageManager.calculate_telemetry(
            raw_frames=estimated_raw_frames,
            sampled_frames=sampled_frame_count,
            candidate_frames=candidate_frame_count,
            expensive_frames=expensive_frames_processed,
            retained_frames=len(tiger_sightings),
            duration_sec=duration
        )

        # Transactional deletion of intermediate temporary frames
        self.storage_mgr.transactional_cleanup(event_id, success=True, telemetry=telemetry)

        return EventResult(
            event_id=event_id,
            camera_id=cam_id,
            timestamp=ts,
            latitude=lat,
            longitude=lon,
            mode=exec_mode,
            animal_detected=animal_seen,
            tiger_detected=True,
            species_label="tiger",
            tiger_count=len(tiger_sightings),
            review_required=review_needed or any(s.status == "REVIEW_REQUIRED" for s in tiger_sightings),
            tigers=tiger_sightings,
            telemetry=telemetry,
            tracks_summary=tracks_summary
        )
