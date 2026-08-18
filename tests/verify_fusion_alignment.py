"""
Verification test for ID synchronization, Late Fusion consensus, and telemetry alignment.
"""

import os
import sys
import json
import numpy as np

# Ensure project root is in path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.fusion import TigerGallery, GalleryEntry, MetricKNNMatcher, WeightedLateFusionEngine
from src.open_world import OpenWorldDetector


def test_fusion_consensus_alignment():
    print("==================================================================")
    print(" RUNNING VERIFICATION: FUSION CONSENSUS & ID ALIGNMENT")
    print("==================================================================")

    # 1. Setup mock gallery with multiple images per tiger
    gallery = TigerGallery(embedding_dim=64)
    
    # Tiger 201 has 1 very close match (d=0.077)
    emb_201 = np.ones(64, dtype=np.float32) / np.sqrt(64)
    gallery.add_entry(GalleryEntry(
        entry_id="REF_201_01",
        embedding=emb_201,
        tiger_id="201",
        side="Right",
        camera_id="CAM_01",
        video_id="vid_201",
        timestamp="2025-01-01",
        source_path="mock_201.jpg"
    ))

    # Tiger 160 has 3 slightly further matches (d=0.096, d=0.098, d=0.099)
    for i, delta in enumerate([0.01, 0.012, 0.015]):
        emb_160 = emb_201 + delta * np.random.randn(64).astype(np.float32)
        emb_160 /= np.linalg.norm(emb_160)
        gallery.add_entry(GalleryEntry(
            entry_id=f"REF_160_{i+1:02d}",
            embedding=emb_160,
            tiger_id="160",
            side="Right",
            camera_id="CAM_02",
            video_id="vid_160",
            timestamp="2025-01-02",
            source_path=f"mock_160_{i}.jpg"
        ))

    # Query vector very close to emb_201
    query_emb = emb_201 + 0.005 * np.random.randn(64).astype(np.float32)
    query_emb /= np.linalg.norm(query_emb)

    matcher = MetricKNNMatcher(gallery, k=7)
    matches = matcher.match(query_emb)

    # Enrich matches with weights
    enriched_matches = []
    for m in matches:
        d = m["distance"]
        w = round(1.0 / (0.1 + d), 4)
        enriched_matches.append({
            **m,
            "weight": w
        })

    print(f"Top {len(enriched_matches)} Nearest Neighbors:")
    for m in enriched_matches:
        print(f" - #{m['rank']} Tiger #{m['tiger_id']} ({m['side']}): d={m['distance']:.4f}, weight=+{m['weight']:.2f}")

    # 2. Run Weighted Late Fusion with synchronized IDs
    fusion_engine = WeightedLateFusionEngine(conf_threshold=0.80, distance_threshold=0.40)
    # Direct classifier votes for 160 with 60% confidence
    fusion_res = fusion_engine.fuse_single_frame(
        classifier_pred_id="160",
        classifier_confidence=0.60,
        metric_top_k=enriched_matches
    )

    winner_id = fusion_res["tiger_id"]
    winner_score = fusion_res["total_score"]
    winner_ev = fusion_res["evidence_breakdown"].get(winner_id, {})
    winner_matches = winner_ev.get("metric_neighbors", [])
    winner_min_d = min([n["distance"] for n in winner_matches]) if winner_matches else 999.0

    print("\n--- Weighted Late Fusion Outcome ---")
    print(f"Winner Identity: Tiger #{winner_id}")
    print(f"Total Cumulative Score: {winner_score:.4f}")
    print(f"Supporting Frames in 7-NN: {len(winner_matches)}")
    print(f"Winner Closest Distance: {winner_min_d:.4f}")
    print(f"All Candidate Scores: {fusion_res['candidate_scores']}")

    assert winner_id == "160", f"Expected Tiger 160 to win consensus vote, got {winner_id}"
    assert len(winner_matches) == 3, f"Expected 3 supporting frames for Tiger 160, got {len(winner_matches)}"
    print("\n>>> ALL CONSENSUS & ID ALIGNMENT CHECKS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    test_fusion_consensus_alignment()
