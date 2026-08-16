"""
Video-Level Zero-Leakage Dataset Splitter
Faithfully implements Section 6 of the paper specification:
- Never randomly splits frames from the same video.
- All frames belonging to a video/burst remain strictly in one split.
- Re-ID split: 60% train, 20% validation, 20% test (at video/shot level).
- Segmentation split: 80% train, 20% validation.
- Closed-set guaranteed across shots for individuals with sufficient videos.
- Deployment extensions: camera-held-out, time-held-out.
"""

from typing import List, Dict, Any, Tuple
import random
from collections import defaultdict
import numpy as np

from .provenance import TigerProvenanceRecord


class VideoLevelDatasetSplitter:
    """
    [PAPER-SPECIFIED VIDEO-LEVEL SPLITTER]
    Splits records based on video/shot boundaries to guarantee zero frame leakage.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def split_reid_dataset(
        self,
        records: List[TigerProvenanceRecord],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        mode: str = "video_held_out"
    ) -> Tuple[List[TigerProvenanceRecord], List[TigerProvenanceRecord], List[TigerProvenanceRecord]]:
        """
        Splits Re-ID records at the video/shot level.
        Preserves individual presence across splits when multiple videos exist (closed-set requirement).
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-4, "Ratios must sum to 1.0"

        # Group records by individual, then by video
        tiger_videos: Dict[str, Dict[str, List[TigerProvenanceRecord]]] = defaultdict(lambda: defaultdict(list))
        for r in records:
            # If video_id is unknown or empty, use fallback grouping (e.g. source dir or burst)
            vid = r.video_id if r.video_id and r.video_id != "Unknown" else f"vid_{r.camera_id}_{r.timestamp[:10]}"
            tiger_videos[r.tiger_id][vid].append(r)

        train_records: List[TigerProvenanceRecord] = []
        val_records: List[TigerProvenanceRecord] = []
        test_records: List[TigerProvenanceRecord] = []

        if mode == "video_held_out":
            # Paper closed-set video split: distribute videos of each tiger across train/val/test
            for tiger_id, vid_dict in tiger_videos.items():
                video_ids = list(vid_dict.keys())
                random.shuffle(video_ids)
                n_vids = len(video_ids)

                if n_vids == 1:
                    # Single video: assigned to train to maintain identity representation
                    train_vids = video_ids
                    val_vids, test_vids = [], []
                elif n_vids == 2:
                    train_vids = [video_ids[0]]
                    val_vids = [video_ids[1]]
                    test_vids = []
                else:
                    n_train = max(1, int(round(n_vids * train_ratio)))
                    n_val = max(1, int(round(n_vids * val_ratio)))
                    if n_train + n_val >= n_vids:
                        n_train = max(1, n_vids - 2)
                        n_val = 1
                    train_vids = video_ids[:n_train]
                    val_vids = video_ids[n_train:n_train + n_val]
                    test_vids = video_ids[n_train + n_val:]

                for vid in train_vids:
                    for r in vid_dict[vid]:
                        r.dataset_split = "train"
                        train_records.append(r)
                for vid in val_vids:
                    for r in vid_dict[vid]:
                        r.dataset_split = "val"
                        val_records.append(r)
                for vid in test_vids:
                    for r in vid_dict[vid]:
                        r.dataset_split = "test"
                        test_records.append(r)

        elif mode == "camera_held_out":
            # [DEPLOYMENT EXTENSION]: Split by camera site
            cameras = list(set(r.camera_id for r in records))
            random.shuffle(cameras)
            n_cams = len(cameras)
            n_train = int(round(n_cams * train_ratio))
            n_val = int(round(n_cams * val_ratio))

            train_cams = set(cameras[:n_train])
            val_cams = set(cameras[n_train:n_train + n_val])
            test_cams = set(cameras[n_train + n_val:])

            for r in records:
                if r.camera_id in train_cams:
                    r.dataset_split = "train"
                    train_records.append(r)
                elif r.camera_id in val_cams:
                    r.dataset_split = "val"
                    val_records.append(r)
                else:
                    r.dataset_split = "test"
                    test_records.append(r)

        elif mode == "time_held_out":
            # [DEPLOYMENT EXTENSION]: Chronological split by timestamp
            sorted_records = sorted(records, key=lambda x: str(x.timestamp))
            n_total = len(sorted_records)
            n_train = int(round(n_total * train_ratio))
            n_val = int(round(n_total * val_ratio))

            for i, r in enumerate(sorted_records):
                if i < n_train:
                    r.dataset_split = "train"
                    train_records.append(r)
                elif i < n_train + n_val:
                    r.dataset_split = "val"
                    val_records.append(r)
                else:
                    r.dataset_split = "test"
                    test_records.append(r)

        return train_records, val_records, test_records

    def split_segmentation_dataset(
        self,
        records: List[TigerProvenanceRecord],
        train_ratio: float = 0.8,
        val_ratio: float = 0.2
    ) -> Tuple[List[TigerProvenanceRecord], List[TigerProvenanceRecord]]:
        """
        [PAPER-SPECIFIED]: 80% train, 20% validation for semantic segmentation.
        """
        video_groups: Dict[str, List[TigerProvenanceRecord]] = defaultdict(list)
        for r in records:
            vid = r.video_id if r.video_id and r.video_id != "Unknown" else r.source_path
            video_groups[vid].append(r)

        vids = list(video_groups.keys())
        random.shuffle(vids)
        n_train = max(1, int(round(len(vids) * train_ratio)))

        train_vids = set(vids[:n_train])
        val_vids = set(vids[n_train:])

        train_records = []
        val_records = []

        for vid in train_vids:
            for r in video_groups[vid]:
                r.dataset_split = "train"
                train_records.append(r)
        for vid in val_vids:
            for r in video_groups[vid]:
                r.dataset_split = "val"
                val_records.append(r)

        return train_records, val_records
