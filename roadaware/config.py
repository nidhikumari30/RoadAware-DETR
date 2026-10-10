"""Configuration objects for the RoadAware-DETR prototype."""

from dataclasses import dataclass


@dataclass
class FusionConfig:
    """Semantic-fusion settings."""

    iom_threshold: float = 0.8
    detector_confidence: float = 0.25


@dataclass
class ProjectConfig:
    """Paths and runtime settings used by the example pipeline."""

    fusion: FusionConfig = FusionConfig()
    device: str = "cpu"
    image_size: int = 640
