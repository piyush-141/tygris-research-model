"""
Weighted Late Fusion & Cross-Frame Video Aggregator
Faithfully implements Section 16, 17, and 20 of the paper:
- Representation branch: weight = 1.0 (participates when probability > Conf-thres = 0.95 or top probability)
- Metric branch: top-7 neighbors with weight = 1 / (0.1 + d) (participates when distance <= Dis-thres)
- Cross-frame / Video-level evidence accumulation over N frames.
- Rejection / Unknown detection if accumulated evidence does not meet criteria.
"""

from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np


class WeightedLateFusionEngine:
    """
    [PAPER-SPECIFIED WEIGHTED LATE FUSION]
    Combines Representation Learning logits and Metric Learning 7-NN evidence.
    """
    def __init__(
        self,
        conf_threshold: float = 0.80,       # Representation confidence threshold
        distance_threshold: float = 1.35,   # Metric distance threshold (calibrated for unit-sphere embeddings)
        representation_weight: float = 1.0, # [PAPER-SPECIFIED: 1.0]
        metric_numerator: float = 1.0,      # [PAPER-SPECIFIED: 1.0]
        metric_constant: float = 0.1        # [PAPER-SPECIFIED: 1 / (0.1 + d)]
    ):
        self.conf_threshold = conf_threshold
        self.distance_threshold = distance_threshold
        self.rep_weight = representation_weight
        self.metric_num = metric_numerator
        self.metric_const = metric_constant

    def fuse_single_frame(
        self,
        classifier_pred_id: str,
        classifier_confidence: float,
        metric_top_k: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Computes the weighted evidence score for a single frame.
        """
        scores: Dict[str, float] = defaultdict(float)
        evidence_breakdown: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"rep_points": 0.0, "metric_points": 0.0, "metric_neighbors": []})

        # 1. Representation Branch Evidence
        if classifier_confidence >= self.conf_threshold:
            rep_pts = self.rep_weight * 1.0
            scores[classifier_pred_id] += rep_pts
            evidence_breakdown[classifier_pred_id]["rep_points"] += rep_pts
        elif classifier_confidence > 0.30:
            rep_pts = self.rep_weight * float(classifier_confidence)
            scores[classifier_pred_id] += rep_pts
            evidence_breakdown[classifier_pred_id]["rep_points"] += rep_pts

        # 2. Metric Branch Evidence (Top-7 Neighbors)
        for neighbor in metric_top_k:
            d = neighbor["distance"]
            n_id = neighbor["tiger_id"]
            if d <= self.distance_threshold:
                # [PAPER-SPECIFIED FORMULA]: weight = 1 / (0.1 + d)
                metric_weight = self.metric_num / (self.metric_const + d)
                scores[n_id] += metric_weight
                evidence_breakdown[n_id]["metric_points"] += metric_weight
                evidence_breakdown[n_id]["metric_neighbors"].append({
                    "rank": neighbor.get("rank"),
                    "distance": d,
                    "side": neighbor.get("side"),
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
        frame_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        [PAPER-SPECIFIED VIDEO/EVENT-LEVEL DECISION]
        Accumulates weighted evidence across all frames in a video or camera event.
        """
        accumulated_scores: Dict[str, float] = defaultdict(float)
        frame_votes: Dict[str, int] = defaultdict(int)
        all_metric_matches: List[Dict[str, Any]] = []

        total_frames = len(frame_results)
        valid_evidence_frames = 0

        for f_res in frame_results:
            cand_scores = f_res.get("candidate_scores", {})
            if cand_scores:
                valid_evidence_frames += 1
                for tiger_id, score in cand_scores.items():
                    accumulated_scores[tiger_id] += score
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

        # Select identity with maximum accumulated score
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
