"""End-to-end orchestration for detector, segmenter, and semantic fusion."""

from __future__ import annotations

from typing import Any

import numpy as np

from .fusion import SemanticFusion
from .models import PIDNetSegmenter, RTDETRv3Detector


class RoadAwarePipeline:
    def __init__(
        self,
        detector: RTDETRv3Detector,
        segmenter: PIDNetSegmenter,
        iom_threshold: float = 0.8,
    ) -> None:
        self.detector = detector
        self.segmenter = segmenter
        self.fusion = SemanticFusion(iom_threshold)

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        detections = self.detector.predict(image)
        road_mask = self.segmenter.predict_road_mask(image)
        accepted, rejected = self.fusion.apply(detections, road_mask)
        return {
            "detections": detections,
            "road_mask": road_mask,
            "accepted": accepted,
            "rejected": rejected,
        }
