"""Training entry point placeholder.

The actual RT-DETRv3 and PIDNet training commands depend on the selected
upstream repositories, configs, datasets, and checkpoints. This script keeps
those commands explicit instead of silently training a different model.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="RoadAware-DETR training plan")
    parser.add_argument("--detector-command", help="Optional RT-DETRv3 training command")
    parser.add_argument("--segmenter-command", help="Optional PIDNet training command")
    args = parser.parse_args()

    print("RoadAware-DETR training plan")
    print("1. Prepare pothole/crack detection annotations.")
    print("2. Prepare pixel-level drivable-road masks.")
    print("3. Train or fine-tune RT-DETRv3.")
    print("4. Train or fine-tune PIDNet.")
    print("5. Validate semantic fusion and IoM threshold on a held-out split.")
    if args.detector_command:
        print(f"Detector command supplied: {args.detector_command}")
    if args.segmenter_command:
        print(f"Segmenter command supplied: {args.segmenter_command}")


if __name__ == "__main__":
    main()
