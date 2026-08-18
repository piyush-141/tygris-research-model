"""
Multi-Factor Tiger Frame Quality Scorer & Diversity-Aware Top-K Selection Engine
Evaluates resolution, sharpness, stripe visibility, pose confidence, contrast, and visual diversity.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
from src.detection.tracker import TrackFrame, TigerTrack


@dataclass
class QualityMetrics:
    total_score: float # [0, 100]
    sharpness: float # Laplacian variance
    tiger_size_ratio: float # Bbox area / Image area
    stripe_contrast: float # High frequency stripe contrast
    lighting_score: float # Optimal exposure without clipping
    pose_confidence: float # Anatomical landmark confidence
    motion_blur_penalty: float
    body_completeness: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": round(self.total_score, 2),
            "sharpness": round(self.sharpness, 2),
            "tiger_size_ratio": round(self.tiger_size_ratio, 4),
            "stripe_contrast": round(self.stripe_contrast, 2),
            "lighting_score": round(self.lighting_score, 2),
            "pose_confidence": round(self.pose_confidence, 3),
            "motion_blur_penalty": round(self.motion_blur_penalty, 2),
            "body_completeness": round(self.body_completeness, 2),
        }


class FrameQualityScorer:
    """
    Computes a multi-factor quality score for a candidate tiger crop/frame.
    """
    def __init__(
        self,
        weight_sharpness: float = 0.25,
        weight_size: float = 0.20,
        weight_stripe: float = 0.25,
        weight_pose: float = 0.15,
        weight_lighting: float = 0.15,
    ):
        self.w_sharp = weight_sharpness
        self.w_size = weight_size
        self.w_stripe = weight_stripe
        self.w_pose = weight_pose
        self.w_light = weight_lighting

    def score_frame(
        self,
        image: Image.Image,
        bbox: List[int],
        pose_confidence: float = 0.85
    ) -> QualityMetrics:
        """
        Calculates quality metrics on the given image and tiger bounding box.
        """
        img_w, img_h = image.size
        img_area = max(1, img_w * img_h)

        # Crop tiger bounding box
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        
        crop_w = max(1, x2 - x1)
        crop_h = max(1, y2 - y1)
        crop_area = crop_w * crop_h
        size_ratio = crop_area / float(img_area)

        crop = image.crop((x1, y1, x2, y2)).convert("RGB")
        crop_np = np.array(crop)

        # 1. Sharpness via 2D Laplacian variance
        gray = np.mean(crop_np, axis=2)
        # Laplacian kernel approximation
        if gray.shape[0] > 4 and gray.shape[1] > 4:
            laplacian = (
                -4 * gray[1:-1, 1:-1]
                + gray[:-2, 1:-1]
                + gray[2:, 1:-1]
                + gray[1:-1, :-2]
                + gray[1:-1, 2:]
            )
            lap_var = float(np.var(laplacian))
        else:
            lap_var = 10.0
        
        # Scale sharpness score to [0, 100]
        sharp_score = min(100.0, max(0.0, lap_var / 12.0))

        # 2. Tiger size score
        # Ideal size: between 15% and 80% of total frame area
        size_score = min(100.0, max(10.0, (size_ratio / 0.35) * 100.0))

        # 3. Stripe contrast score
        # Horizontal & vertical gradient within tiger coat
        grad_x = np.abs(np.diff(gray, axis=1))
        stripe_contrast = float(np.percentile(grad_x, 90) if grad_x.size > 0 else 10.0)
        stripe_score = min(100.0, max(0.0, stripe_contrast * 2.2))

        # 4. Lighting & dynamic range score
        mean_lum = float(np.mean(gray))
        std_lum = float(np.std(gray))
        # Penalize overexposure (>220) or underexposure (<30)
        if 60 <= mean_lum <= 180:
            light_score = min(100.0, max(30.0, std_lum * 1.6))
        else:
            light_score = max(10.0, 50.0 - abs(mean_lum - 120) * 0.5)

        # 5. Motion blur penalty
        motion_penalty = max(0.0, (40.0 - lap_var) * 0.5) if lap_var < 40 else 0.0

        # 6. Body completeness
        aspect = crop_w / float(crop_h)
        body_comp = 95.0 if (0.9 <= aspect <= 2.5) else 70.0

        # Weighted aggregate score
        total = (
            self.w_sharp * sharp_score
            + self.w_size * size_score
            + self.w_stripe * stripe_score
            + self.w_pose * (pose_confidence * 100.0)
            + self.w_light * light_score
            - motion_penalty * 0.15
        )
        total = float(np.clip(total, 5.0, 99.5))

        return QualityMetrics(
            total_score=total,
            sharpness=sharp_score,
            tiger_size_ratio=size_ratio,
            stripe_contrast=stripe_score,
            lighting_score=light_score,
            pose_confidence=pose_confidence,
            motion_blur_penalty=motion_penalty,
            body_completeness=body_comp
        )


class DiversityFrameSelector:
    """
    Selects the Top-K highest-quality diverse frames per tiger track.
    Avoids choosing adjacent, near-identical frames.
    """
    def __init__(self, default_top_k: int = 3, min_frame_distance: int = 2):
        self.default_top_k = default_top_k
        self.min_frame_distance = min_frame_distance
        self.scorer = FrameQualityScorer()

    def select_best_frames(
        self,
        track: TigerTrack,
        top_k: Optional[int] = None
    ) -> List[TrackFrame]:
        """
        Ranks candidate frames within the track and selects top-K diverse frames.
        """
        k = top_k or self.default_top_k
        if not track.frames:
            return []

        # If track has fewer or equal frames than K, return all sorted by quality
        for tf in track.frames:
            if tf.quality_score <= 0.0 and tf.image is not None:
                pose_conf = tf.pose_info.get("pose_confidence", 0.85) if tf.pose_info else 0.85
                qm = self.scorer.score_frame(tf.image, tf.bbox, pose_conf)
                tf.quality_score = qm.total_score

        # Sort candidate frames by descending quality score
        ranked_frames = sorted(track.frames, key=lambda f: f.quality_score, reverse=True)

        selected: List[TrackFrame] = []
        for cand in ranked_frames:
            if len(selected) >= k:
                break
            
            # Check diversity: ensure candidate is not too close in frame index to already selected frames
            is_diverse = True
            for sel in selected:
                if abs(cand.frame_idx - sel.frame_idx) < self.min_frame_distance:
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(cand)

        # Fallback if diversity constraint leaves fewer than requested
        if len(selected) < min(k, len(track.frames)):
            for cand in ranked_frames:
                if cand not in selected:
                    selected.append(cand)
                if len(selected) >= k:
                    break

        # Best frame is the single highest scoring
        track.best_frame = ranked_frames[0] if ranked_frames else None
        track.selected_frames = selected
        return selected
