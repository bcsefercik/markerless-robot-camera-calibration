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
    base_pose: np.array = None
    key_points: list((int, np.array)) = None
    key_points_pose: np.array = np.zeros(7, dtype=np.float)
    is_confident: bool = False
    timestamp: datetime = None
    confidence: float = None
    id: str = None


@dataclass
class RawDTO:
    points: np.array
    rgb: np.array
    pose: np.array
    segmentation: np.array
    other: dict = None

    def to_point_cloud_dto(self) -> PointCloudDTO:
        return PointCloudDTO(self.points, self.rgb, datetime.utcnow())
