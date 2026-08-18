"""
Detection & Tracking Module for Two-Pass Event Processing
"""

from .animal_detector import FastAnimalDetector, TigerInstanceDetector, DetectionResult, SpeciesConfirmation
from .tracker import MultiTigerTracker, TigerTrack, TrackFrame

__all__ = [
    "FastAnimalDetector",
    "TigerInstanceDetector",
    "DetectionResult",
    "SpeciesConfirmation",
    "MultiTigerTracker",
    "TigerTrack",
    "TrackFrame",
]
