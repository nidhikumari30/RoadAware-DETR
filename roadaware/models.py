"""Model adapters used by the example pipeline.

Replace these adapters with the actual RT-DETRv3 and PIDNet inference calls
once the external model repositories and checkpoints are installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class RTDETRv3Detector:
    """Adapter interface for an RT-DETRv3 detector.

    The starter implementation intentionally raises NotImplementedError rather
    than returning fabricated detections.
    """

    def __init__(self, config: str | Path | None = None, weights: str | Path | None = None):
        self.config = config
        self.weights = weights

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "Connect this adapter to the RT-DETRv3/PaddleDetection inference API."
        )


class PIDNetSegmenter:
    """Adapter interface for PIDNet drivable-road segmentation."""

    def __init__(self, config: str | Path | None = None, weights: str | Path | None = None):
        self.config = config
        self.weights = weights

    def predict_road_mask(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Connect this adapter to the PIDNet inference API."
        )
