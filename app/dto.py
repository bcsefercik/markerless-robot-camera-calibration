from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class PointCloudDTO:
    points: np.array
    rgb: np.array
    timestamp: datetime
    joint_angles: np.array = None
    id: str = None


@dataclass
class ResultDTO:
    segmentation: np.array
    ee_pose: np.array = np.zeros(7, dtype=np.float)
    timestamp: datetime = None
    confidence: float = None
    base_pose: np.array = None
    id: str = None
