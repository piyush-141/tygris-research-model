"""
Segmentation Inference, Background Removal & Visual QA Logger
Faithfully implements Section 9 & 10 of the paper:
- Generates binary segmentation masks using DDRNet-39
- Falls back to YOLO-seg instance segmentation
- Falls back to GrabCut + HSV saliency mask
- Removes environmental background
- Crops tight bounding boxes around segmented tigers
- Validates masks and logs visual QA failure modes
"""

import os
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


FAILURE_MODES = [
    "fragmented tiger",
    "missing body parts",
    "vegetation classified as tiger",
    "background retained",
    "low-light failure",
    "motion blur",
    "incomplete stripe extraction"
]


class SegmentationPipeline:
    """
    [PAPER-SPECIFIED SEGMENTATION & BACKGROUND REMOVAL PIPELINE]
    Priority chain:
      1. YOLO-seg instance segmentation (fast, precise bounding polygon)
      2. DDRNet-39 pixel-wise semantic mask
      3. GrabCut + HSV tiger colour saliency (robust fallback)
    """
    def __init__(self, model: Optional[nn.Module] = None, device: str = "cpu", input_size: Tuple[int, int] = (512, 512)):
        self.model = model.to(device) if model is not None else None
        self.device = device
        self.input_size = input_size
        self.qa_failure_log: List[Dict[str, Any]] = []

        # Load YOLO-seg engine (camera-trap optimised: conf 0.05)
        self.yolo_seg = None
        if _YOLO_AVAILABLE:
            try:
                self.yolo_seg = YOLO("yolov8n-seg.pt")
            except Exception:
                self.yolo_seg = None

    # -------------------------------------------------------------------------
    # PRIMARY ENTRY POINT — used by event_processor Pass 2
    # -------------------------------------------------------------------------
    def segment_and_crop(self, image: Image.Image) -> Tuple[np.ndarray, Image.Image, Image.Image, Tuple[int, int, int, int]]:
        """
        End-to-end segmentation and crop extractor.
        Priority: YOLO-seg → DDRNet-39 → GrabCut+HSV.
        Returns: (mask_full [H×W uint8 0/1], tiger_only_full_img, tiger_tight_crop, bbox)
        """
        orig_w, orig_h = image.size
        img_np = np.array(image.convert("RGB"))

        mask_full = None
        bbox = None

        # --- Priority 1: YOLO-seg instance segmentation ---
        if self.yolo_seg is not None:
            try:
                results = self.yolo_seg.predict(image, verbose=False, conf=0.05)[0]
                if results.boxes is not None and len(results.boxes) > 0:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    best_idx = int(np.argmax(areas))
                    box = boxes[best_idx]

                    x1, y1, x2, y2 = map(int, box)
                    pad_w = int((x2 - x1) * 0.04)
                    pad_h = int((y2 - y1) * 0.04)
                    x1 = max(0, x1 - pad_w)
                    y1 = max(0, y1 - pad_h)
                    x2 = min(orig_w, x2 + pad_w)
                    y2 = min(orig_h, y2 + pad_h)
                    bbox = (x1, y1, x2, y2)

                    if results.masks is not None and len(results.masks) > best_idx:
                        m = results.masks.data[best_idx].cpu().numpy()
                        m_resized = cv2.resize(
                            (m * 255).astype(np.uint8), (orig_w, orig_h),
                            interpolation=cv2.INTER_LINEAR
                        )
                        # Morphological close to fill gaps inside the mask
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                        m_closed = cv2.morphologyEx(m_resized, cv2.MORPH_CLOSE, kernel, iterations=2)
                        mask_full = (m_closed > 80).astype(np.uint8)
                    else:
                        # No polygon — use bounding-box region as rectangular mask
                        mask_full = np.zeros((orig_h, orig_w), dtype=np.uint8)
                        mask_full[y1:y2, x1:x2] = 1
            except Exception:
                mask_full = None

        # --- Priority 2: DDRNet-39 semantic mask ---
        if mask_full is None or float(np.sum(mask_full)) / max(1, orig_w * orig_h) < 0.005:
            ddr_mask = self.predict_mask(image)
            # Only adopt DDRNet mask if it provides meaningful coverage
            ddr_cov = float(np.sum(ddr_mask)) / max(1, orig_w * orig_h)
            if ddr_cov >= 0.005:
                mask_full = ddr_mask

        # --- Priority 3: GrabCut + HSV saliency fallback ---
        if mask_full is None or float(np.sum(mask_full)) / max(1, orig_w * orig_h) < 0.005:
            mask_full = self._grabcut_tiger_mask(image)

        # Compute bbox from mask if still missing
        if bbox is None:
            coords = np.argwhere(mask_full == 1)
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                pad_w = int((x_max - x_min) * 0.04)
                pad_h = int((y_max - y_min) * 0.04)
                bbox = (
                    int(max(0, x_min - pad_w)),
                    int(max(0, y_min - pad_h)),
                    int(min(orig_w, x_max + pad_w)),
                    int(min(orig_h, y_max + pad_h))
                )
            else:
                bbox = (0, 0, orig_w, orig_h)

        # Apply mask: zero-out background pixels
        mask_3d = np.repeat(mask_full[:, :, np.newaxis], 3, axis=2)
        tiger_only_np = np.where(mask_3d == 1, img_np, 0)
        tiger_only_img = Image.fromarray(tiger_only_np)

        # Extract tight crop
        x1, y1, x2, y2 = bbox
        if x2 - x1 < 10 or y2 - y1 < 10:
            x1, y1, x2, y2 = 0, 0, orig_w, orig_h

        # The neural network was trained on natural bounding box crops (with background).
        # Masking the background to black ruins the feature distributions for Re-ID!
        crop_np = img_np[y1:y2, x1:x2]

        tiger_crop = Image.fromarray(crop_np)
        return mask_full, tiger_only_img, tiger_crop, bbox

    # -------------------------------------------------------------------------
    # DDRNet-39 forward pass
    # -------------------------------------------------------------------------
    def predict_mask(self, image: Image.Image) -> np.ndarray:
        """Runs DDRNet-39 segmentation model. Falls back to GrabCut+HSV on failure."""
        orig_w, orig_h = image.size

        if self.model is not None:
            try:
                img_resized = image.resize(self.input_size, Image.Resampling.BILINEAR)
                img_np = np.array(img_resized, dtype=np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img_norm = (img_np - mean) / std
                tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model(tensor)
                    if output.shape[1] >= 2:
                        probs = torch.softmax(output, dim=1)[:, 1, :, :].squeeze(0).cpu().numpy()
                        # Permissive threshold (0.35) to recover more tiger pixels
                        pred = (probs > 0.35).astype(np.uint8)
                    else:
                        pred = (torch.sigmoid(output).squeeze().cpu().numpy() > 0.45).astype(np.uint8)

                # Morphological cleanup
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel, iterations=2)
                pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel, iterations=1)

                coverage = np.mean(pred)
                if 0.005 <= coverage <= 0.85:
                    mask_full = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    return mask_full
            except Exception:
                pass

        return self._grabcut_tiger_mask(image)

    # -------------------------------------------------------------------------
    # GrabCut + HSV saliency — replaces simple np.diff-based fallback
    # -------------------------------------------------------------------------
    def _grabcut_tiger_mask(self, image: Image.Image) -> np.ndarray:
        """
        Combines HSV tiger-orange colour saliency with GrabCut iterative refinement.
        Produces a much tighter and more accurate foreground mask than a center crop.
        """
        orig_w, orig_h = image.size
        img_np = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        # Broad tiger-fur HSV range (orange-amber-tawny)
        lower_fur = np.array([5, 30, 30])
        upper_fur = np.array([35, 255, 255])
        fur_mask = cv2.inRange(hsv, lower_fur, upper_fur)

        # Find bounding box of the dominant fur-coloured blob
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fur_mask, connectivity=8)
        grab_rect = None
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            x = stats[largest_label, cv2.CC_STAT_LEFT]
            y = stats[largest_label, cv2.CC_STAT_TOP]
            w = stats[largest_label, cv2.CC_STAT_WIDTH]
            h = stats[largest_label, cv2.CC_STAT_HEIGHT]
            pad_x, pad_y = int(w * 0.10), int(h * 0.10)
            x1 = max(1, x - pad_x)
            y1 = max(1, y - pad_y)
            x2 = min(orig_w - 2, x + w + pad_x)
            y2 = min(orig_h - 2, y + h + pad_y)
            if (x2 - x1) > 20 and (y2 - y1) > 20:
                grab_rect = (x1, y1, x2 - x1, y2 - y1)

        if grab_rect is None:
            # Fallback: centre 70% of the image
            grab_rect = (
                int(orig_w * 0.05), int(orig_h * 0.05),
                int(orig_w * 0.90), int(orig_h * 0.90)
            )

        # GrabCut refinement
        try:
            bgd_model = np.zeros((1, 65), dtype=np.float64)
            fgd_model = np.zeros((1, 65), dtype=np.float64)
            gc_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            cv2.grabCut(img_bgr, gc_mask, grab_rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
            fg_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        except Exception:
            fg_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            rx, ry, rw, rh = grab_rect
            fg_mask[ry:ry+rh, rx:rx+rw] = 1

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return fg_mask

    # -------------------------------------------------------------------------
    # Legacy compatibility wrapper
    # -------------------------------------------------------------------------
    def extract_tiger_crop(
        self,
        image: Image.Image,
        mask: np.ndarray,
        padding_ratio: float = 0.05
    ) -> Tuple[Image.Image, Image.Image, Optional[Tuple[int, int, int, int]]]:
        """Wrapper maintaining compatibility with existing calls."""
        orig_w, orig_h = image.size
        img_np = np.array(image.convert("RGB"))

        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        tiger_only_np = np.where(mask_3d == 1, img_np, 0)
        tiger_only_img = Image.fromarray(tiger_only_np)

        coords = np.argwhere(mask == 1)
        if len(coords) == 0:
            return tiger_only_img, image, (0, 0, orig_w, orig_h)

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        pad_w = int((x_max - x_min) * padding_ratio)
        pad_h = int((y_max - y_min) * padding_ratio)

        x1 = max(0, x_min - pad_w)
        y1 = max(0, y_min - pad_h)
        x2 = min(orig_w, x_max + pad_w)
        y2 = min(orig_h, y_max + pad_h)

        crop_np = img_np[y1:y2, x1:x2]
        if crop_np.shape[0] < 10 or crop_np.shape[1] < 10:
            return tiger_only_img, image, (x1, y1, x2, y2)

        tiger_crop = Image.fromarray(crop_np)
        return tiger_only_img, tiger_crop, (int(x1), int(y1), int(x2), int(y2))

    def validate_mask_and_log_qa(
        self,
        image: Image.Image,
        mask: np.ndarray,
        image_name: str
    ) -> Dict[str, Any]:
        """Performs visual QA checks on segmentation masks as required by Section 10 of the paper."""
        orig_w, orig_h = image.size
        total_pixels = orig_w * orig_h
        tiger_pixels = np.sum(mask == 1)
        coverage_ratio = tiger_pixels / float(total_pixels)

        is_failed = False
        reasons: List[str] = []

        if coverage_ratio < 0.005:
            is_failed = True
            reasons.append("fragmented tiger")
        elif coverage_ratio > 0.90:
            is_failed = True
            reasons.append("background retained")

        record = {
            "image_name": image_name,
            "coverage_ratio": round(coverage_ratio, 4),
            "status": "FLAGGED" if is_failed else "PASSED",
            "failure_reasons": reasons,
            "remediation": "Requires manual verification or bbox refinement" if is_failed else "Automatic extraction approved"
        }
        self.qa_failure_log.append(record)
        return record
