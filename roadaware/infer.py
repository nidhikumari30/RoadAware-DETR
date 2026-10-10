"""Run semantic fusion on JSON detections and a binary road-mask image.

Example:
    python -m roadaware.infer \
      --image data/frame.jpg \
      --road-mask data/frame_mask.png \
      --detections data/detections.json \
      --output outputs/frame_result.json

The detector and segmenter adapters in ``roadaware.models`` are deliberately
interfaces. Use this utility to test the fusion layer independently first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .fusion import SemanticFusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Optional image, used for size validation")
    parser.add_argument("--road-mask", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/result.json"))
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    road_mask = cv2.imread(str(args.road_mask), cv2.IMREAD_GRAYSCALE)
    if road_mask is None:
        raise FileNotFoundError(f"Could not read road mask: {args.road_mask}")

    if args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")
        if image.shape[:2] != road_mask.shape[:2]:
            raise ValueError("Image and road mask must have identical height and width")

    with args.detections.open(encoding="utf-8") as handle:
        detections = json.load(handle)

    accepted, rejected = SemanticFusion(args.threshold).apply(detections, road_mask > 0)
    result = {"threshold": args.threshold, "accepted": accepted, "rejected": rejected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
