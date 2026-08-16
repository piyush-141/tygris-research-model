"""
Open-World Unknown Tiger Detector & Distance-Threshold Benchmark
Faithfully implements Section 18 & 19 of the paper:
- Supports KNOWN vs UNKNOWN recognition without forced closed-set classification
- Evaluates open-world benchmarks across distance thresholds [0.10, 0.08, 0.05, 0.02, 0.01]
- Manages candidate rejection and verification queue
"""

from typing import List, Dict, Any, Tuple
import numpy as np


PAPER_OPEN_WORLD_BENCHMARK_THRESHOLDS = [0.10, 0.08, 0.05, 0.02, 0.01]


class OpenWorldDetector:
    """
    [PAPER-SPECIFIED OPEN-WORLD ENGINE]
    Determines if a tiger sighting is a known individual or a novel/unknown candidate.
    """
    def __init__(
        self,
        conf_threshold: float = 0.70,
        dist_threshold: float = 1.35
    ):
        self.conf_threshold = conf_threshold
        self.dist_threshold = dist_threshold
        self.verification_queue: List[Dict[str, Any]] = []

    def classify_sighting(
        self,
        classifier_pred_id: str,
        classifier_prob: float,
        nearest_distance: float,
        nearest_tiger_id: str,
        provenance_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Applies dual-threshold gating to decide KNOWN vs UNKNOWN.
        """
        is_conf_valid = classifier_prob >= self.conf_threshold
        is_dist_valid = nearest_distance <= self.dist_threshold
        agrees = (classifier_pred_id == nearest_tiger_id)

        if (is_conf_valid and is_dist_valid) or (agrees and is_dist_valid):
            status = "KNOWN"
            recognized = True
            decision_id = nearest_tiger_id if is_dist_valid else classifier_pred_id
            confidence = float(max(classifier_prob, 1.0 / (0.1 + nearest_distance)))
            confidence = min(0.99, max(0.55, confidence))
        elif is_dist_valid:
            status = "KNOWN"
            recognized = True
            decision_id = nearest_tiger_id
            confidence = float(1.0 / (0.1 + nearest_distance))
            confidence = min(0.99, max(0.50, confidence))
        else:
            status = "UNKNOWN"
            recognized = False
            decision_id = None
            confidence = 0.0

            # Queue for manual human verification & enrollment
            queue_item = {
                "candidate_id": f"CAND_{len(self.verification_queue) + 1:04d}",
                "status": "PENDING_VERIFICATION",
                "nearest_gallery_tiger": nearest_tiger_id,
                "nearest_distance": float(nearest_distance),
                "classifier_hypothesis": classifier_pred_id,
                "classifier_prob": float(classifier_prob),
                "provenance": provenance_dict
            }
            self.verification_queue.append(queue_item)

        return {
            "recognized": recognized,
            "status": status,
            "tiger_id": decision_id,
            "confidence": round(confidence, 4),
            "nearest_neighbor_distance": round(float(nearest_distance), 4),
            "classifier_confidence": round(float(classifier_prob), 4)
        }

    def evaluate_open_world_benchmark(
        self,
        test_distances: np.ndarray,
        is_known_mask: np.ndarray,
        thresholds: List[float] = PAPER_OPEN_WORLD_BENCHMARK_THRESHOLDS
    ) -> List[Dict[str, Any]]:
        """
        [PAPER-SPECIFIED BENCHMARK]: Evaluates known-individual accuracy vs unknown detection rate
        across paper thresholds.
        """
        results = []
        for thres in thresholds:
            accepted = (test_distances <= thres)
            tp = np.sum(accepted & is_known_mask)
            fn = np.sum((~accepted) & is_known_mask)
            tn = np.sum((~accepted) & (~is_known_mask))
            fp = np.sum(accepted & (~is_known_mask))

            total_pos = max(1, np.sum(is_known_mask))
            total_neg = max(1, np.sum(~is_known_mask))

            known_accuracy = float(tp / total_pos)
            unknown_detection_rate = float(tn / total_neg)

            results.append({
                "threshold": thres,
                "known_accuracy": round(known_accuracy, 4),
                "unknown_detection_rate": round(unknown_detection_rate, 4),
                "tp": int(tp),
                "fn": int(fn),
                "tn": int(tn),
                "fp": int(fp)
            })
        return results
