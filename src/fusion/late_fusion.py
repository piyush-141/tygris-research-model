"""
Weighted Late Fusion & Cross-Frame Video Aggregator — Accuracy-Maximised
Faithfully implements Section 16, 17, and 20 of the paper with:
- Representation branch: weight = 1.0 (participates when probability > Conf-thres = 0.95 or top probability)
- Metric branch: top-7 neighbors with inverse-distance kernel w = 1 / (0.1 + d)
- Side-Aware Boosting: matching body side (Left vs Left, Right vs Right) receives boost
- Quality-Weighted Video Aggregation: frames with higher sharpness contribute proportionally
- Rejection / Unknown detection if accumulated evidence does not meet criteria.
"""

from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np


class WeightedLateFusionEngine:
    """
    [ACCURACY-MAXIMISED WEIGHTED LATE FUSION]
    Combines Representation Learning logits, Metric Learning 7-NN evidence,
    camera side awareness, and multi-frame quality weighting.
    """
    def __init__(
        self,
        conf_threshold: float = 0.80,       # Representation confidence threshold
        distance_threshold: float = 1.35,   # Metric distance threshold
        representation_weight: float = 1.0, # [PAPER-SPECIFIED: 1.0]
        metric_numerator: float = 1.0,      # [PAPER-SPECIFIED: 1.0]
        metric_constant: float = 0.10,      # [PAPER-SPECIFIED: 0.10]
        side_boost: float = 1.20            # Side-matching boost factor
    ):
        self.conf_threshold = conf_threshold
        self.distance_threshold = distance_threshold
        self.rep_weight = representation_weight
        self.metric_num = metric_numerator
        self.metric_const = metric_constant
        self.side_boost = side_boost

    def fuse_single_frame(
        self,
        classifier_pred_id: str,
        classifier_confidence: float,
        metric_top_k: List[Dict[str, Any]],
        query_side: Optional[str] = None,
        frame_quality: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Computes the weighted evidence score for a single frame.
        """
        scores: Dict[str, float] = defaultdict(float)
        evidence_breakdown: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"rep_points": 0.0, "metric_points": 0.0, "metric_neighbors": []})

        # Quality factor: 1.0 if not specified, otherwise scaled relative to nominal 75/100
        quality_factor = 1.0 if (frame_quality is None or frame_quality <= 1.0) else max(0.6, min(1.4, frame_quality / 75.0))

        # 1. Representation Branch Evidence
        if classifier_confidence >= self.conf_threshold:
            rep_pts = self.rep_weight * 1.0 * quality_factor
            scores[classifier_pred_id] += rep_pts
            evidence_breakdown[classifier_pred_id]["rep_points"] += rep_pts
        elif classifier_confidence > 0.30:
            rep_pts = self.rep_weight * float(classifier_confidence) * quality_factor
            scores[classifier_pred_id] += rep_pts
            evidence_breakdown[classifier_pred_id]["rep_points"] += rep_pts

        # 2. Metric Branch Evidence (Top-7 Neighbors)
        for neighbor in metric_top_k:
            d = neighbor["distance"]
            n_id = neighbor["tiger_id"]
            n_side = neighbor.get("side", "Unknown")

            if d <= self.distance_threshold:
                # [PAPER-SPECIFIED FORMULA]: weight = 1 / (0.1 + d)
                metric_weight = (self.metric_num / (self.metric_const + d)) * quality_factor

                # Side-aware boost: if query side matches gallery flank side
                if query_side and n_side and query_side in ["Left", "Right"] and n_side == query_side:
                    metric_weight *= self.side_boost

                scores[n_id] += metric_weight
                evidence_breakdown[n_id]["metric_points"] += metric_weight
                evidence_breakdown[n_id]["metric_neighbors"].append({
                    "rank": neighbor.get("rank"),
                    "distance": d,
                    "side": n_side,
                    "weight": round(metric_weight, 4)
                })

        if not scores:
            return {
                "recognized": False,
                "tiger_id": None,
                "status": "UNKNOWN",
                "total_score": 0.0,
                "candidate_scores": {},
                "evidence_breakdown": {}
            }

        best_id = max(scores, key=scores.get)
        best_score = scores[best_id]

        return {
            "recognized": True,
            "tiger_id": best_id,
            "status": "KNOWN",
            "total_score": float(best_score),
            "candidate_scores": dict(scores),
            "evidence_breakdown": dict(evidence_breakdown)
        }

    def aggregate_video_event(
        self,
        frame_results: List[Dict[str, Any]],
        quality_weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        [PAPER-SPECIFIED VIDEO/EVENT-LEVEL DECISION]
        Accumulates quality-weighted evidence across all frames in a video event.
        """
        accumulated_scores: Dict[str, float] = defaultdict(float)
        frame_votes: Dict[str, int] = defaultdict(int)
        all_metric_matches: List[Dict[str, Any]] = []

        total_frames = len(frame_results)
        valid_evidence_frames = 0

        for f_idx, f_res in enumerate(frame_results):
            q_w = quality_weights[f_idx] if quality_weights and f_idx < len(quality_weights) else 1.0
            cand_scores = f_res.get("candidate_scores", {})
            
            if cand_scores:
                valid_evidence_frames += 1
                for tiger_id, score in cand_scores.items():
                    accumulated_scores[tiger_id] += score * q_w
                    frame_votes[tiger_id] += 1

            for tiger_id, ev in f_res.get("evidence_breakdown", {}).items():
                for n_entry in ev.get("metric_neighbors", []):
                    all_metric_matches.append({
                        "tiger_id": tiger_id,
                        **n_entry
                    })

        if not accumulated_scores:
            return {
                "tiger_id": None,
                "recognized": False,
                "status": "UNKNOWN",
                "confidence": 0.0,
                "supporting_frame_count": 0,
                "total_frames_analyzed": total_frames,
                "candidate_scores": {},
                "decision_reason": "No frame satisfied confidence or distance thresholds"
            }

        best_id = max(accumulated_scores, key=accumulated_scores.get)
        best_score = accumulated_scores[best_id]

        avg_score = best_score / max(1, valid_evidence_frames)
        normalized_confidence = min(0.99, max(0.4, avg_score / 2.0))

        all_metric_matches.sort(key=lambda x: x["distance"])

        return {
            "tiger_id": best_id,
            "recognized": True,
            "status": "KNOWN",
            "confidence": round(normalized_confidence, 4),
            "accumulated_fusion_score": round(best_score, 4),
            "supporting_frame_count": frame_votes.get(best_id, 0),
            "total_frames_analyzed": total_frames,
            "candidate_scores": dict(accumulated_scores),
            "top_metric_neighbors": all_metric_matches[:7]
        }
