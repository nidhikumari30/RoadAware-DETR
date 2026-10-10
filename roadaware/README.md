# RoadAware-DETR Python starter

This folder contains a runnable starter for the semantic-fusion portion of the project.

## Install

```bash
pip install -r requirements-roadaware.txt
```

## Test the fusion layer

Prepare:

- a grayscale road-mask image, where nonzero pixels represent drivable road;
- a JSON list of detections, for example:

```json
[
  {"bbox": [10, 20, 110, 140], "score": 0.91, "label": "pothole"}
]
```

Run:

```bash
python -m roadaware.infer \
  --road-mask data/frame_mask.png \
  --detections data/detections.json \
  --output outputs/frame_result.json
```

The default proposed IoM threshold is `0.8`; change it with `--threshold`.

## Connecting the real models

`roadaware/models.py` contains adapter interfaces for RT-DETRv3 and PIDNet. The adapters intentionally raise `NotImplementedError` until you connect them to the official model APIs and checkpoints. This avoids claiming fabricated inference results.

The project references:

- RT-DETRv3: https://github.com/clxia12/RT-DETRv3
- PIDNet: https://github.com/XuJiacong/PIDNet

## Important metric definition

This implementation uses:

```text
IoM = road pixels inside detection box / total pixels in detection box
```

That corresponds to `Area(B ∩ M) / Area(B)` and measures how much of a detected hazard box lies on the drivable road.
