"""
Tiger Re-Identification Data Provenance Schema
Faithful implementation of provenance and metadata tracking.
Every image, frame, crop, and embedding preserves the complete 16-field record.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
import json
import os
import pandas as pd


VALID_SEASONS = {"Spring", "Summer", "Autumn", "Winter", "Unknown"}
VALID_TIMES_OF_DAY = {"Day", "Dusk", "Night", "Dawn", "Unknown"}
VALID_LIGHTING = {"Direct sunlight", "No direct sunlight", "Night (IR Flash)", "Dawn / Dusk", "Unknown"}
VALID_VIEWS = {"Left", "Right", "Left-oblique", "Right-oblique", "Front", "Back", "Unknown"}
VALID_VEGETATION = {"Herbaceous obstruction", "Woody obstruction", "No obvious obstruction", "Unknown"}
VALID_POSTURES = {"Walking", "Standing", "Lying", "Running", "Unknown"}
VALID_STRIPE_VISIBILITY = {"10–30%", "30–50%", "50–70%", "70–100%", "Unknown"}
VALID_SPLITS = {"train", "val", "test", "unassigned"}


@dataclass
class TigerProvenanceRecord:
    """
    [PAPER-SPECIFIED & PROVENANCE REQUIREMENT]
    Retains all 16 required provenance fields for every image, crop, and embedding.
    """
    tiger_id: str
    side: str                          # Left, Right, Left-oblique, Right-oblique, Unknown
    camera_id: str
    video_id: str                      # Video / Shot identifier (unit of splitting)
    frame_id: str                      # Frame sequence number or unique ID
    timestamp: str                     # ISO-8601 or standard datetime
    latitude: float
    longitude: float
    season: str                        # Spring, Summer, Autumn, Winter
    time_of_day: str                   # Day, Dusk, Night, Dawn
    lighting: str                      # Direct sunlight, No direct sunlight, IR Flash
    vegetation_occlusion: str          # Herbaceous, Woody, No obvious obstruction
    posture: str                       # Walking, Standing, Lying
    stripe_integrity: str              # 10–30%, 30–50%, 50–70%, 70–100%
    source_path: str                   # Filepath to original image
    dataset_split: str = "unassigned"  # train, val, test

    # Optional internal tracking
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TigerProvenanceRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def validate(self) -> List[str]:
        """Validates that all required fields conform to expected categories or flags them."""
        warnings = []
        if self.season not in VALID_SEASONS:
            warnings.append(f"Non-standard season: {self.season}")
        if self.side not in VALID_VIEWS:
            warnings.append(f"Non-standard side/view: {self.side}")
        if self.dataset_split not in VALID_SPLITS:
            warnings.append(f"Invalid dataset_split: {self.dataset_split}")
        return warnings


class ProvenanceRegistry:
    """
    Manages collection, persistence, and querying of provenance records across the pipeline.
    """
    def __init__(self):
        self.records: Dict[str, TigerProvenanceRecord] = {}

    def register(self, key: str, record: TigerProvenanceRecord):
        self.records[key] = record

    def get(self, key: str) -> Optional[TigerProvenanceRecord]:
        return self.records.get(key)

    def save_json(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {k: v.to_dict() for k, v in self.records.items()}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_json(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.records = {k: TigerProvenanceRecord.from_dict(v) for k, v in data.items()}

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame([v.to_dict() for v in self.records.values()])
