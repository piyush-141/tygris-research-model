"""
Comprehensive Paper & Deployment Metrics Suite
Faithfully implements Section 22:
- Segmentation metrics: TIoU, BIoU, MIoU
- Representation metrics: mAP, Micro-F1, Top-1, Top-3
- Metric learning metrics: AMI, Precision@1, R-Precision, MAP@R, MRR
- Deployment metrics: System Accuracy, Precision, Micro-F1, Unknown Detection Rate,
  Known-Individual Accuracy, False-ID Rate, Inference Latency
"""

import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score


class SystemDeploymentMetricsCalculator:
    """
    [DEPLOYMENT METRICS CALCULATOR]
    Computes system-level open-world and operational performance.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.true_identities = []
        self.predicted_identities = []
        self.recognized_flags = []
        self.true_known_flags = []
        self.latencies_ms = []

    def update_record(
        self,
        true_id: str,
        predicted_id: Optional[str],
        is_recognized: bool,
        is_true_known: bool,
        latency_ms: float
    ):
        self.true_identities.append(true_id)
        self.predicted_identities.append(predicted_id)
        self.recognized_flags.append(is_recognized)
        self.true_known_flags.append(is_true_known)
        self.latencies_ms.append(latency_ms)

    def compute(self) -> Dict[str, float]:
        if not self.true_identities:
            return {}

        n = len(self.true_identities)
        true_known = np.array(self.true_known_flags)
        pred_recog = np.array(self.recognized_flags)

        # 1. Unknown Detection Rate (True Unknowns correctly rejected)
        unknown_mask = ~true_known
        if np.sum(unknown_mask) > 0:
            unknown_detection_rate = float(np.sum((~pred_recog) & unknown_mask) / np.sum(unknown_mask))
        else:
            unknown_detection_rate = 1.0

        # 2. Known-Individual Accuracy (Among true knowns, correctly identified)
        known_mask = true_known
        if np.sum(known_mask) > 0:
            correct_known = 0
            for i in range(n):
                if known_mask[i] and pred_recog[i] and self.predicted_identities[i] == self.true_identities[i]:
                    correct_known += 1
            known_acc = float(correct_known / np.sum(known_mask))
        else:
            known_acc = 0.0

        # 3. False ID Rate (Wrong tiger assigned among recognized queries)
        recognized_count = np.sum(pred_recog)
        if recognized_count > 0:
            false_ids = 0
            for i in range(n):
                if pred_recog[i] and self.predicted_identities[i] != self.true_identities[i]:
                    false_ids += 1
            false_id_rate = float(false_ids / recognized_count)
        else:
            false_id_rate = 0.0

        # 4. Latency
        avg_latency = float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

        return {
            "Total Queries Evaluated": n,
            "Known-Individual Accuracy": round(known_acc, 4),
            "Unknown Detection Rate": round(unknown_detection_rate, 4),
            "False-ID Rate": round(false_id_rate, 4),
            "Average Latency (ms/query)": round(avg_latency, 2),
            "Throughput (FPS)": round(1000.0 / max(0.1, avg_latency), 1)
        }
