"""
Fast Animal Detector and Species Confirmation Module for Two-Pass Camera-Trap Screening
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import numpy as np
import torch


@dataclass
class DetectionResult:
    bbox: List[int] # [x_min, y_min, x_max, y_max]
    confidence: float
    species: str # "tiger", "deer", "boar", "human", "unknown_animal", etc.
    instance_id: Optional[str] = None # e.g. "EVENT_TIGER_001"
    is_tiger: bool = False


@dataclass
class SpeciesConfirmation:
    animal_detected: bool
    tiger_detected: bool
    species_label: str
    confidence: float
    detections: List[DetectionResult] = field(default_factory=list)
    review_required: bool = False


class FastAnimalDetector:
    """
    Pass 1 Cheap Screening: Determines if frame contains an animal.
    Low-latency heuristic / model screening to filter out wind, leaves, shadows, empty triggers.
    """
    def __init__(self, confidence_threshold: float = 0.35, device: str = "cpu"):
        self.confidence_threshold = confidence_threshold
        self.device = device

    def detect_animal(self, image: Image.Image) -> Tuple[bool, float]:
        """
        Fast animal detection on a single PIL Image.
        Returns: (has_animal: bool, confidence: float)
        """
        # Convert to numpy for fast low-level visual energy / objectness analysis
        img_np = np.array(image.convert("RGB"))
        h, w, _ = img_np.shape

        # Color and gradient energy heuristic combined with central saliency
        gray = np.mean(img_np, axis=2)
        grad_y = np.abs(np.diff(gray, axis=0))
        grad_x = np.abs(np.diff(gray, axis=1))
        energy = np.mean(grad_y) + np.mean(grad_x)

        # Foreground contrast ratio
        center_crop = gray[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        std_contrast = np.std(center_crop)

        # Baseline confidence estimate
        conf = min(0.99, max(0.1, (energy / 35.0) * 0.5 + (std_contrast / 50.0) * 0.5))
        has_animal = conf >= self.confidence_threshold
        return has_animal, float(conf)


class TigerInstanceDetector:
    """
    Species Confirmation and Multi-Tiger Instance Bounding Box Extractor.
    Detects multiple tiger instances per frame, assigning temporary instance IDs.
    """
    def __init__(self, tiger_conf_threshold: float = 0.45, review_threshold: float = 0.30):
        self.tiger_conf_threshold = tiger_conf_threshold
        self.review_threshold = review_threshold
        self._instance_counter = 0

    def reset_counter(self):
        self._instance_counter = 0

    def detect_instances(self, image: Image.Image, frame_idx: int = 0) -> SpeciesConfirmation:
        """
        Runs species confirmation and multi-tiger instance localization on a frame.
        """
        img_np = np.array(image.convert("RGB"))
        h, w, _ = img_np.shape

        # Compute orange/amber tiger coat saturation & stripe dark frequency
        r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
        # Tiger color signature: high red/amber relative to blue/green
        tiger_color_mask = (r > 90) & (r > g + 15) & (g > b) & (b < 140)
        tiger_pixels = np.sum(tiger_color_mask)
        total_pixels = h * w
        tiger_ratio = tiger_pixels / max(1, total_pixels)

        # Dark stripe frequency within amber regions
        gray = np.mean(img_np, axis=2)
        stripe_mask = (gray < 75) & (tiger_color_mask | (r > 70))
        stripe_pixels = np.sum(stripe_mask)

        # Multi-region cluster analysis to detect potential multiple tigers
        detections: List[DetectionResult] = []
        
        # Saliency bounding box generation
        # Find active regions
        active_y, active_x = np.where(tiger_color_mask | stripe_mask)
        
        if len(active_y) > 100:
            # Check if there are multiple spatially separated tiger clusters
            x_min, x_max = int(np.percentile(active_x, 2)), int(np.percentile(active_x, 98))
            y_min, y_max = int(np.percentile(active_y, 2)), int(np.percentile(active_y, 98))
            
            # Add padding
            pad_x = int((x_max - x_min) * 0.08)
            pad_y = int((y_max - y_min) * 0.08)
            bbox = [
                max(0, x_min - pad_x),
                max(0, y_min - pad_y),
                min(w, x_max + pad_x),
                min(h, y_max + pad_y)
            ]
            
            # Confidence score calculation
            conf = min(0.98, max(0.20, (tiger_ratio * 25.0) + (stripe_pixels / max(1, tiger_pixels + 1)) * 0.45 + 0.35))
            
            # Check if large width suggests two tigers standing side-by-side or behind
            box_w = bbox[2] - bbox[0]
            box_h = bbox[3] - bbox[1]
            aspect = box_w / max(1, box_h)

            if aspect > 2.2 and (x_max - x_min) > w * 0.5:
                # Split into two candidate tiger instances
                mid_x = (bbox[0] + bbox[2]) // 2
                self._instance_counter += 1
                det1 = DetectionResult(
                    bbox=[bbox[0], bbox[1], mid_x, bbox[3]],
                    confidence=float(conf * 0.95),
                    species="tiger",
                    instance_id=f"EVENT_TIGER_{self._instance_counter:03d}",
                    is_tiger=True
                )
                self._instance_counter += 1
                det2 = DetectionResult(
                    bbox=[mid_x, bbox[1], bbox[2], bbox[3]],
                    confidence=float(conf * 0.92),
                    species="tiger",
                    instance_id=f"EVENT_TIGER_{self._instance_counter:03d}",
                    is_tiger=True
                )
                detections.extend([det1, det2])
            else:
                self._instance_counter += 1
                det = DetectionResult(
                    bbox=bbox,
                    confidence=float(conf),
                    species="tiger",
                    instance_id=f"EVENT_TIGER_{self._instance_counter:03d}",
                    is_tiger=True
                )
                detections.append(det)

        if not detections:
            # Fallback: whole frame bbox if general animal characteristics found
            gray_std = np.std(gray)
            if gray_std > 30:
                conf = 0.42
                det = DetectionResult(
                    bbox=[int(w*0.05), int(h*0.05), int(w*0.95), int(h*0.95)],
                    confidence=conf,
                    species="tiger",
                    instance_id=f"EVENT_TIGER_{self._instance_counter+1:03d}",
                    is_tiger=True
                )
                self._instance_counter += 1
                detections.append(det)
            else:
                return SpeciesConfirmation(
                    animal_detected=False,
                    tiger_detected=False,
                    species_label="no_animal",
                    confidence=0.1,
                    detections=[],
                    review_required=False
                )

        max_conf = max([d.confidence for d in detections]) if detections else 0.0
        tiger_detected = max_conf >= self.tiger_conf_threshold
        review_required = (self.review_threshold <= max_conf < self.tiger_conf_threshold)

        return SpeciesConfirmation(
            animal_detected=True,
            tiger_detected=tiger_detected,
            species_label="tiger" if tiger_detected else ("uncertain" if review_required else "other_animal"),
            confidence=float(max_conf),
            detections=detections,
            review_required=review_required
        )
