"""
Automated Test Suite for Two-Pass Event Processing Pipeline
"""

import os
import sys
import unittest
from PIL import Image
import numpy as np
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.detection.animal_detector import FastAnimalDetector, TigerInstanceDetector
from src.detection.tracker import MultiTigerTracker
from src.pose.pose_detector import TigerPoseDetector
from src.quality.frame_quality import FrameQualityScorer, DiversityFrameSelector
from src.stripes.stripe_analyzer import TigerStripeAnalyzer
from src.storage.storage_manager import StorageManager
from src.event_processor.event_processor import EventProcessor, EventResult
from src.segmentation import ddrnet39, SegmentationPipeline
from src.representation import get_representation_model
from src.metric_learning import get_metric_model
from src.fusion import TigerGallery, GalleryEntry
from src.ecology import SightingDatabase


class TestTwoPassEventPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cpu"
        cls.seg_model = ddrnet39(num_classes=2)
        cls.seg_pipeline = SegmentationPipeline(cls.seg_model, device=cls.device)
        cls.rep_model = get_representation_model(num_classes=20, name="ConvNeXt-small", pretrained=False)
        cls.metric_model = get_metric_model(name="ConvNeXt-small", embedding_dim=64, pretrained=False)
        cls.gallery = TigerGallery(embedding_dim=64)
        
        # Populate mock gallery
        for i in range(10):
            emb = np.random.randn(64).astype(np.float32)
            emb /= np.linalg.norm(emb)
            cls.gallery.add_entry(GalleryEntry(
                entry_id=f"REF_{i:03d}",
                embedding=emb,
                tiger_id=f"TIG_{i+1:03d}",
                side="Left",
                camera_id="PTR-CORE-EP-01",
                video_id="vid_001",
                timestamp="2025-01-10T10:00:00",
                source_path="outputs/sample.jpg"
            ))

        cls.sighting_db = SightingDatabase("outputs/test_pench_sightings.db")
        cls.processor = EventProcessor(
            seg_pipeline=cls.seg_pipeline,
            rep_model=cls.rep_model,
            metric_model=cls.metric_model,
            gallery=cls.gallery,
            sighting_db=cls.sighting_db,
            device=cls.device,
            mode="production"
        )

    def test_animal_screening(self):
        detector = FastAnimalDetector()
        img = Image.new("RGB", (640, 480), color=(180, 100, 40))
        has_animal, conf = detector.detect_animal(img)
        self.assertIsInstance(has_animal, bool)
        self.assertGreaterEqual(conf, 0.0)

    def test_multi_tiger_tracking(self):
        tracker = MultiTigerTracker(iou_threshold=0.20)
        from src.detection.animal_detector import DetectionResult
        
        # Frame 1: Two tigers detected
        d1 = DetectionResult(bbox=[50, 50, 200, 200], confidence=0.92, species="tiger", instance_id="EVENT_TIGER_001", is_tiger=True)
        d2 = DetectionResult(bbox=[300, 100, 500, 300], confidence=0.88, species="tiger", instance_id="EVENT_TIGER_002", is_tiger=True)
        tracks = tracker.update(0, [d1, d2])
        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracks[0].track_id, "Track 1")
        self.assertEqual(tracks[1].track_id, "Track 2")

        # Frame 2: Same two tigers moved slightly
        d1_next = DetectionResult(bbox=[55, 52, 205, 202], confidence=0.94, species="tiger", instance_id="EVENT_TIGER_001", is_tiger=True)
        d2_next = DetectionResult(bbox=[305, 105, 505, 305], confidence=0.90, species="tiger", instance_id="EVENT_TIGER_002", is_tiger=True)
        tracks_updated = tracker.update(1, [d1_next, d2_next])
        self.assertEqual(len(tracks_updated), 2)
        self.assertEqual(tracks_updated[0].num_frames, 2)
        self.assertEqual(tracks_updated[1].num_frames, 2)

    def test_pose_estimation(self):
        pose_engine = TigerPoseDetector()
        crop = Image.new("RGB", (250, 150), color=(190, 110, 30))
        res = pose_engine.estimate_pose(crop)
        self.assertIn(res.posture, ["walking", "standing", "lying", "crouched"])
        self.assertGreater(len(res.keypoints), 10)
        self.assertIn("nose", res.keypoints)
        self.assertIn("spine_mid", res.keypoints)

    def test_stripe_extraction(self):
        stripe_engine = TigerStripeAnalyzer()
        crop = Image.new("RGB", (224, 224), color=(180, 100, 30))
        res = stripe_engine.extract_stripes(crop)
        self.assertIsNotNone(res.stripe_ridge_b64)
        self.assertTrue(res.stripe_ridge_b64.startswith("data:image/jpeg;base64,"))
        self.assertGreaterEqual(res.stripe_match_score, 0.0)

    def test_full_event_processing(self):
        # Use real image from dataset if available
        sample_path = os.path.join("dataset", "atrw_reid_train", "train", "000001.jpg")
        if not os.path.exists(sample_path):
            sample_path = os.path.join("atrw_reid_train", "train", "000001.jpg")
        if os.path.exists(sample_path):
            img = Image.open(sample_path).convert("RGB")
            frames = [img, img, img]
        else:
            # Generate image with high visual contrast and gradients
            arr = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            arr[100:350, 150:500, 0] = 220 # Red/amber tiger coat
            arr[100:350, 150:500, 1] = 120
            arr[100:350, 150:500, 2] = 30
            # Add stripes
            for s_x in range(160, 480, 25):
                arr[120:330, s_x:s_x+8] = 15 # Dark stripes
            frames = [Image.fromarray(arr), Image.fromarray(arr)]

        meta = {
            "event_id": "EVT_TEST_001",
            "camera_id": "PTR-CORE-EP-01",
            "timestamp": "2026-08-17T02:31:12",
            "latitude": 21.685,
            "longitude": 79.310
        }
        res: EventResult = self.processor.process_event(frames, meta)
        self.assertTrue(res.animal_detected)
        self.assertTrue(res.tiger_detected)
        self.assertGreater(res.tiger_count, 0)
        self.assertIsNotNone(res.telemetry)
        self.assertGreater(res.telemetry.storage_reduction_pct, 50.0)


if __name__ == "__main__":
    unittest.main()
