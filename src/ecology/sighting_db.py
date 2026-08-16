"""
Tiger Sighting & Occurrence Database
Faithfully implements Section 26 of the paper:
- Records occurrence logs with all 13 required fields:
  event_id, tiger_id, camera_id, latitude, longitude, timestamp, confidence,
  source_image/video, side, embedding_distance, supporting_frame_count, model_version, threshold_version
- Powers last-seen querying, sighting history, and ecological spatial analysis
"""

import sqlite3
import os
import json
from typing import List, Dict, Any, Optional
import pandas as pd


class SightingDatabase:
    """
    [PAPER-SPECIFIED SIGHTING DATABASE]
    Stores verified tiger occurrences with spatial, temporal, and model provenance.
    """
    def __init__(self, db_path: str = "outputs/pench_sightings.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sightings (
                    event_id TEXT PRIMARY KEY,
                    tiger_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_image_or_video TEXT,
                    side TEXT,
                    embedding_distance REAL,
                    supporting_frame_count INTEGER,
                    model_version TEXT,
                    threshold_version TEXT,
                    extra_json TEXT
                )
            """)
            conn.commit()

    def record_sighting(
        self,
        event_id: str,
        tiger_id: str,
        camera_id: str,
        latitude: float,
        longitude: float,
        timestamp: str,
        confidence: float,
        source_image_or_video: str = "",
        side: str = "Unknown",
        embedding_distance: float = 0.0,
        supporting_frame_count: int = 1,
        model_version: str = "DDRNet39+ConvNeXt-S-v1",
        threshold_version: str = "Conf0.95_Dis0.4",
        extra: Optional[Dict[str, Any]] = None
    ) -> bool:
        extra_str = json.dumps(extra or {})
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO sightings (
                    event_id, tiger_id, camera_id, latitude, longitude, timestamp,
                    confidence, source_image_or_video, side, embedding_distance,
                    supporting_frame_count, model_version, threshold_version, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id, tiger_id, camera_id, latitude, longitude, timestamp,
                confidence, source_image_or_video, side, embedding_distance,
                supporting_frame_count, model_version, threshold_version, extra_str
            ))
            conn.commit()
        return True

    def get_all_sightings(self) -> pd.DataFrame:
        with self._get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM sightings ORDER BY timestamp ASC", conn)

    def get_tiger_sightings(self, tiger_id: str) -> pd.DataFrame:
        with self._get_connection() as conn:
            return pd.read_sql_query(
                "SELECT * FROM sightings WHERE tiger_id = ? ORDER BY timestamp ASC",
                conn,
                params=(tiger_id,)
            )

    def get_last_seen_report(self) -> pd.DataFrame:
        """Computes the latest sighting timestamp and camera for every tracked individual."""
        query = """
            SELECT 
                tiger_id,
                MAX(timestamp) as last_seen_time,
                camera_id as last_seen_camera,
                latitude as last_latitude,
                longitude as last_longitude,
                COUNT(*) as total_sightings,
                AVG(confidence) as avg_confidence
            FROM sightings
            GROUP BY tiger_id
            ORDER BY last_seen_time DESC
        """
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn)
