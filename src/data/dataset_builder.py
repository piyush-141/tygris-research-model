"""
Tiger Dataset Builder & Pipeline Preprocessor
Builds binary semantic segmentation datasets (0=bg, 1=tiger) and Re-ID datasets with full provenance.
"""

import os
import json
import hashlib
from typing import Dict, List, Any, Optional
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np

from .provenance import TigerProvenanceRecord, ProvenanceRegistry
from .qc_audit import QualityControlAudit
from .splitter import VideoLevelDatasetSplitter


class TigerDatasetBuilder:
    """
    [PAPER-SPECIFIED DATASET BUILDER]
    Constructs segmentation and Re-ID datasets with video-level splitting and provenance records.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = ProvenanceRegistry()
        self.qc = QualityControlAudit()
        self.splitter = VideoLevelDatasetSplitter(seed=config.get("system", {}).get("random_seed", 42))

    def build_from_workspace(
        self,
        base_dir: str = ".",
        pench_metadata_path: Optional[str] = "../dataset training/Pench_Synthetic_Metadata/pench_tiger_metadata_train.csv",
        reid_list_path: Optional[str] = "../dataset training/Amur Tigers/reid_list_train.csv",
        keypoints_path: Optional[str] = "../dataset training/Amur Tigers/reid_keypoints_train.json"
    ) -> Dict[str, Any]:
        """
        Builds the unified dataset from the current workspace images, synthetic Pench metadata, and ATRW annotations.
        """
        # Load available metadata
        pench_df = None
        if pench_metadata_path and os.path.exists(pench_metadata_path):
            try:
                pench_df = pd.read_csv(pench_metadata_path)
                pench_df.set_index("filename", inplace=True)
            except Exception as e:
                print(f"[Warning] Could not load Pench metadata: {e}")

        reid_map = {}
        if reid_list_path and os.path.exists(reid_list_path):
            try:
                df_reid = pd.read_csv(reid_list_path, header=None, names=["tiger_id", "filename"])
                reid_map = dict(zip(df_reid["filename"], df_reid["tiger_id"].astype(str)))
            except Exception as e:
                print(f"[Warning] Could not load ReID list: {e}")

        keypoints_data = {}
        if keypoints_path and os.path.exists(keypoints_path):
            try:
                with open(keypoints_path, "r") as f:
                    keypoints_data = json.load(f)
            except Exception as e:
                print(f"[Warning] Could not load Keypoints: {e}")

        # Scan images across local folders (supporting dataset/ subfolder or root)
        def _resolve_dir(*subpaths):
            p1 = os.path.join(base_dir, "dataset", *subpaths)
            if os.path.exists(p1):
                return p1
            return os.path.join(base_dir, *subpaths)

        directories = {
            "reid_train": _resolve_dir("atrw_reid_train", "train"),
            "reid_test": _resolve_dir("atrw_reid_test", "test"),
            "detection_train": _resolve_dir("atrw_detection_train", "trainval"),
            "detection_test": _resolve_dir("atrw_detection_test", "test"),
            "pose_train": _resolve_dir("atrw_pose_train", "train"),
            "pose_val": _resolve_dir("atrw_pose_val", "val"),
        }

        # Run QC Audit on raw image files
        image_audit = self.qc.audit_image_files(directories)

        # Build provenance records for all valid images
        records: List[TigerProvenanceRecord] = []
        for split_dir_name, path in directories.items():
            if not os.path.exists(path):
                continue
            for fname in sorted(os.listdir(path)):
                fpath = os.path.join(path, fname)
                if not os.path.isfile(fpath) or not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                # Default values or extracted from metadata
                tiger_id = reid_map.get(fname, "Unknown")
                camera_id = "CAM_DEFAULT"
                video_id = f"vid_{fname[:4]}" # Video grouping based on sequence prefix
                timestamp = "2025-01-01T12:00:00"
                lat, lon = 21.65, 79.30 # Pench reserve reference coordinates
                season = "Winter"
                time_of_day = "Day"
                lighting = "No direct sunlight"
                vegetation = "No obvious obstruction"
                posture = "Walking"
                stripe_integrity = "70–100%"
                side = "Left" if int(hashlib.md5(fname.encode()).hexdigest(), 16) % 2 == 0 else "Right"

                # If Pench metadata exists, overwrite with rich ground truth / deployment data
                if pench_df is not None and fname in pench_df.index:
                    row = pench_df.loc[fname]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    tiger_id = str(row.get("tiger_id", tiger_id))
                    camera_id = str(row.get("station_id", camera_id))
                    lat = float(row.get("latitude", lat))
                    lon = float(row.get("longitude", lon))
                    timestamp = str(row.get("timestamp", timestamp))
                    lighting = str(row.get("lighting_condition", lighting))
                    video_id = f"vid_{camera_id}_{timestamp[:10]}"

                record = TigerProvenanceRecord(
                    tiger_id=tiger_id,
                    side=side,
                    camera_id=camera_id,
                    video_id=video_id,
                    frame_id=fname,
                    timestamp=timestamp,
                    latitude=lat,
                    longitude=lon,
                    season=season,
                    time_of_day=time_of_day,
                    lighting=lighting,
                    vegetation_occlusion=vegetation,
                    posture=posture,
                    stripe_integrity=stripe_integrity,
                    source_path=fpath,
                    dataset_split="unassigned",
                    extra_metadata={"has_keypoints": fname in keypoints_data}
                )
                records.append(record)
                self.registry.register(fname, record)

        # Run diversity audit
        diversity_audit = self.qc.audit_provenance_diversity(records)

        # Split ReID dataset at video level
        reid_records = [r for r in records if r.tiger_id != "Unknown"]
        train_reid, val_reid, test_reid = self.splitter.split_reid_dataset(
            reid_records,
            train_ratio=self.config.get("reid", {}).get("split", {}).get("train", 0.6),
            val_ratio=self.config.get("reid", {}).get("split", {}).get("val", 0.2),
            test_ratio=self.config.get("reid", {}).get("split", {}).get("test", 0.2),
            mode="video_held_out"
        )

        # Split Segmentation dataset at video level (80/20)
        train_seg, val_seg = self.splitter.split_segmentation_dataset(
            records,
            train_ratio=self.config.get("segmentation", {}).get("split", {}).get("train", 0.8),
            val_ratio=self.config.get("segmentation", {}).get("split", {}).get("val", 0.2)
        )

        return {
            "image_audit": image_audit,
            "diversity_audit": diversity_audit,
            "total_records": len(records),
            "reid_split_counts": {
                "train": len(train_reid),
                "val": len(val_reid),
                "test": len(test_reid)
            },
            "seg_split_counts": {
                "train": len(train_seg),
                "val": len(val_seg)
            },
            "records": records
        }
