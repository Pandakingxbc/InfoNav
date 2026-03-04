"""
Object Detection Module for ApexNav.

Provides multiple detector backends:
- YOLOv7: Fast real-time detection (original)
- D-FINE: State-of-the-art DETR-based detection (recommended)
- GroundingDINO: Open-vocabulary grounding detection
"""

from .detections import ObjectDetections

# Import detectors (handle import errors gracefully)
try:
    from .yolov7 import YOLOv7, YOLOv7Client
except ImportError:
    YOLOv7 = None
    YOLOv7Client = None

try:
    from .dfine import DFine, DFineClient, ObjectDetector, ObjectDetectorClient
except ImportError:
    DFine = None
    DFineClient = None
    ObjectDetector = None
    ObjectDetectorClient = None

try:
    from .grounding_dino import GroundingDINO, GroundingDINOClient
except ImportError:
    GroundingDINO = None
    GroundingDINOClient = None

__all__ = [
    "ObjectDetections",
    # YOLOv7
    "YOLOv7",
    "YOLOv7Client",
    # D-FINE
    "DFine",
    "DFineClient",
    "ObjectDetector",
    "ObjectDetectorClient",
    # GroundingDINO
    "GroundingDINO",
    "GroundingDINOClient",
]
