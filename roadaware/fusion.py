"""Road-mask and bounding-box semantic fusion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass
class Detection:
    """A detector output represented in xyxy image coordinates."""

    bbox: Sequence[float]
    score: float
    label: str = "pothole"



def compute_iom(box: Sequence[float], road_mask: np.ndarray) -> float:
    """Compute the fraction of a box covered by drivable-road pixels.

    IoM is defined here as ``Area(box ∩ road_mask) / Area(box)``. The mask may
    be boolean, 0/1, or any numeric array where nonzero means drivable road.
    """
    if road_mask.ndim != 2:
        raise ValueError("road_mask must have shape [height, width]")

    x1, y1, x2, y2 = (int(round(value)) for value in box)
    height, width = road_mask.shape
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = road_mask[y1:y2, x1:x2]
    return float(np.count_nonzero(roi) / roi.size)


class SemanticFusion:
    """Accept or reject detector outputs using a drivable-road mask."""

    def __init__(self, threshold: float = 0.8) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold

    def apply(
        self,
        detections: Iterable[Detection | dict[str, Any]],
        road_mask: np.ndarray,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for item in detections:
            result = item.__dict__.copy() if isinstance(item, Detection) else dict(item)
            result["iom"] = compute_iom(result["bbox"], road_mask)
            result["status"] = (
                "accepted" if result["iom"] >= self.threshold else "rejected"
            )
            (accepted if result["status"] == "accepted" else rejected).append(result)

        return accepted, rejected
