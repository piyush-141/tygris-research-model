"""
Automated Ablation Experiments Suite
Faithfully implements Section 10 & 21 of the paper:
- Background Ablation: Background Retained vs Object Detection Crop vs Semantic Segmentation Crop
- Architecture Ablation: Representation Only vs Metric Only vs Weighted Late Fusion
"""

from typing import Dict, Any, List
import pandas as pd


# [PAPER-REPORTED REFERENCE BENCHMARKS ON CONVNEXT-SMALL]
PAPER_BACKGROUND_BENCHMARKS = {
    "Background Retained": {
        "Accuracy": 0.5664,
        "Precision": 0.5678,
        "Micro-F1": 0.7232
    },
    "Object Detection Crop": {
        "Accuracy": 0.9549,
        "Precision": 0.9573,
        "Micro-F1": 0.9769
    },
    "Semantic Segmentation Crop": {
        "Accuracy": 0.9549,
        "Precision": 0.9549,
        "Micro-F1": 0.9769
    }
}

PAPER_ARCHITECTURE_BENCHMARKS = {
    "Representation Only": {
        "Accuracy": 0.9373,
        "Precision": 0.9373,
        "Micro-F1": 0.9677
    },
    "Metric Only": {
        "Accuracy": 0.9148,
        "Precision": 0.9148,
        "Micro-F1": 0.9555
    },
    "Weighted Late Fusion": {
        "Accuracy": 0.9549,
        "Precision": 0.9549,
        "Micro-F1": 0.9769
    }
}


class AblationExperimentRunner:
    """
    Manages and compares ablation trials against the paper's reported baselines.
    """
    def __init__(self):
        self.dataset_results: Dict[str, Dict[str, Any]] = {}

    def record_experiment_result(self, category: str, variant: str, metrics: Dict[str, float]):
        if category not in self.dataset_results:
            self.dataset_results[category] = {}
        self.dataset_results[category][variant] = metrics

    def generate_ablation_comparison_table(self, category: str) -> pd.DataFrame:
        """
        Creates side-by-side comparison between Paper Reported Benchmarks and Dataset Experimental Results.
        """
        paper_ref = PAPER_BACKGROUND_BENCHMARKS if category == "background" else PAPER_ARCHITECTURE_BENCHMARKS
        my_results = self.dataset_results.get(category, {})

        rows = []
        for variant, p_metrics in paper_ref.items():
            my_m = my_results.get(variant, {})
            rows.append({
                "Variant / Setting": variant,
                "Paper Reported Accuracy": p_metrics.get("Accuracy"),
                "Dataset Accuracy": my_m.get("Accuracy", "Pending Trial"),
                "Paper Reported Precision": p_metrics.get("Precision"),
                "Dataset Precision": my_m.get("Precision", "Pending Trial"),
                "Paper Reported Micro-F1": p_metrics.get("Micro-F1"),
                "Dataset Micro-F1": my_m.get("Micro-F1", "Pending Trial")
            })
        return pd.DataFrame(rows)
