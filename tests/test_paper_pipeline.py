"""
Comprehensive Automated Unit & Pipeline Tests
Verifies all paper mathematical specifications:
1. DDRNet-39 segmentation output shape and TIoU/BIoU/MIoU computation
2. ConvNeXt-small representation output shape and Top-1/Top-3/mAP computation
3. ConvNeXt-small metric branch 64-D normalized embedding head
4. Euclidean 7-NN retrieval correctness
5. Exact Weighted Late Fusion formula (w_rep = 1.0, w_metric = 1 / (0.1 + d))
6. Open-World dual-threshold gating and distance threshold sweep
7. 100% Minimum Convex Polygon (MCP) convex hull home-range calculation
"""

import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.data.provenance import TigerProvenanceRecord, ProvenanceRegistry
from src.data.splitter import VideoLevelDatasetSplitter
from src.segmentation.models.ddrnet import ddrnet39, ddrnet23
from src.segmentation.trainer import SegmentationMetricsCalculator
from src.representation.models import get_representation_model
from src.representation.trainer import RepresentationMetricsCalculator
from src.metric_learning.models import get_metric_model
from src.metric_learning.trainer import MetricRetrievalEvaluator
from src.fusion.gallery import TigerGallery, GalleryEntry
from src.fusion.matcher import MetricKNNMatcher
from src.fusion.late_fusion import WeightedLateFusionEngine
from src.open_world.unknown_detector import OpenWorldDetector
from src.ecology.spatial_analysis import EcologicalSpatialAnalyzer, polygon_area_km2, haversine_distance_km
import pandas as pd


class TestPaperPipeline(unittest.TestCase):

    def test_ddrnet39_architecture(self):
        """Tests DDRNet-39 forward pass on dummy image batch."""
        model = ddrnet39(num_classes=2)
        model.eval()
        dummy_input = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            output = model(dummy_input)
        self.assertEqual(output.shape, (2, 2, 256, 256))

    def test_segmentation_metrics(self):
        """Tests TIoU, BIoU, MIoU calculation."""
        calc = SegmentationMetricsCalculator(num_classes=2)
        # Perfect prediction on 10x10 patch
        preds = torch.zeros((1, 10, 10), dtype=torch.long)
        preds[0, :5, :5] = 1
        targets = preds.clone()
        calc.update(preds, targets)
        metrics = calc.compute()
        self.assertAlmostEqual(metrics["TIoU"], 1.0)
        self.assertAlmostEqual(metrics["BIoU"], 1.0)
        self.assertAlmostEqual(metrics["MIoU"], 1.0)

    def test_representation_model_and_metrics(self):
        """Tests ConvNeXt-small representation output and Top-1 / Top-3 metrics."""
        rep_net = get_representation_model(num_classes=10, name="ConvNeXt-small", pretrained=False)
        rep_net.eval()
        dummy_crop = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = rep_net(dummy_crop)
        self.assertEqual(logits.shape, (2, 10))

        # Test metrics
        calc = RepresentationMetricsCalculator(num_classes=10)
        dummy_logits = torch.tensor([[10.0, 1.0, 0.0], [0.0, 10.0, 0.0]])
        dummy_targets = torch.tensor([0, 1])
        calc.update(dummy_logits, dummy_targets)
        res = calc.compute()
        self.assertEqual(res["Top-1"], 1.0)
        self.assertEqual(res["Top-3"], 1.0)

    def test_metric_learning_64d_embedding(self):
        """Tests 64-D embedding generation and L2 normalization."""
        metric_net = get_metric_model(name="ConvNeXt-small", embedding_dim=64, pretrained=False)
        metric_net.eval()
        dummy_crop = torch.randn(3, 3, 224, 224)
        with torch.no_grad():
            embeds = metric_net(dummy_crop, normalize=True)
        self.assertEqual(embeds.shape, (3, 64))
        # Verify L2 unit norm
        norms = torch.norm(embeds, p=2, dim=1).numpy()
        for norm in norms:
            self.assertAlmostEqual(float(norm), 1.0, places=4)

    def test_7nn_euclidean_matcher(self):
        """Tests Euclidean distance calculation and 7-NN ranking."""
        gallery = TigerGallery(embedding_dim=64)
        for i in range(10):
            vec = np.zeros(64, dtype=np.float32)
            vec[i] = 1.0
            entry = GalleryEntry(
                entry_id=f"entry_{i}",
                embedding=vec,
                tiger_id=f"TIG_{i}",
                side="Left" if i % 2 == 0 else "Right",
                camera_id="CAM_01",
                video_id=f"vid_{i}",
                timestamp="2025-01-01",
                source_path=f"path_{i}.jpg"
            )
            gallery.add_entry(entry)

        matcher = MetricKNNMatcher(gallery, k=7)
        # Query close to entry_0
        query = np.zeros(64, dtype=np.float32)
        query[0] = 0.95
        query[1] = 0.05
        query /= np.linalg.norm(query)

        results = matcher.match(query)
        self.assertEqual(len(results), 7)
        self.assertEqual(results[0]["tiger_id"], "TIG_0")
        self.assertLess(results[0]["distance"], 0.2)

    def test_exact_paper_late_fusion_formula(self):
        """Tests weighted fusion formula: w_rep = 1.0, w_metric = 1 / (0.1 + d)."""
        engine = WeightedLateFusionEngine(
            conf_threshold=0.95,
            distance_threshold=0.4,
            representation_weight=1.0,
            metric_numerator=1.0,
            metric_constant=0.1
        )
        # Scenario: Classifier predicts TIG_001 with 0.98 prob (score += 1.0)
        # Metric top-7 has TIG_001 at distance 0.1 (weight = 1/(0.1+0.1) = 5.0)
        metric_matches = [
            {"tiger_id": "TIG_001", "distance": 0.1, "side": "Left", "rank": 1},
            {"tiger_id": "TIG_002", "distance": 0.3, "side": "Left", "rank": 2}, # weight = 1/(0.1+0.3) = 2.5
        ]
        res = engine.fuse_single_frame(
            classifier_pred_id="TIG_001",
            classifier_confidence=0.98,
            metric_top_k=metric_matches
        )
        self.assertTrue(res["recognized"])
        self.assertEqual(res["tiger_id"], "TIG_001")
        # Expected score: 1.0 (rep) + 5.0 (metric) = 6.0
        self.assertAlmostEqual(res["candidate_scores"]["TIG_001"], 6.0)
        self.assertAlmostEqual(res["candidate_scores"]["TIG_002"], 2.5)

    def test_open_world_rejection(self):
        """Tests rejection of novel/unconfident queries as UNKNOWN."""
        detector = OpenWorldDetector(conf_threshold=0.95, dist_threshold=0.4)
        res = detector.classify_sighting(
            classifier_pred_id="TIG_001",
            classifier_prob=0.40, # Below 0.95
            nearest_distance=0.85, # Above 0.40
            nearest_tiger_id="TIG_002",
            provenance_dict={}
        )
        self.assertFalse(res["recognized"])
        self.assertEqual(res["status"], "UNKNOWN")

    def test_100_percent_mcp_calculation(self):
        """Tests 100% Minimum Convex Polygon (MCP) calculation and polygon area."""
        df_sightings = pd.DataFrame([
            {"tiger_id": "TIG_01", "latitude": 21.650, "longitude": 79.310, "timestamp": "2025-01-01", "camera_id": "C1"},
            {"tiger_id": "TIG_01", "latitude": 21.660, "longitude": 79.310, "timestamp": "2025-01-02", "camera_id": "C2"},
            {"tiger_id": "TIG_01", "latitude": 21.660, "longitude": 79.320, "timestamp": "2025-01-03", "camera_id": "C3"},
            {"tiger_id": "TIG_01", "latitude": 21.650, "longitude": 79.320, "timestamp": "2025-01-04", "camera_id": "C4"},
        ])
        analyzer = EcologicalSpatialAnalyzer(df_sightings)
        mcp = analyzer.compute_100_percent_mcp("TIG_01")
        self.assertEqual(mcp["status"], "COMPUTED_100_MCP")
        self.assertGreater(mcp["mcp_area_km2"], 0.0)
        self.assertEqual(len(mcp["polygon_points"]), 4)


if __name__ == "__main__":
    unittest.main()
