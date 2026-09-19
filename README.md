# Assembly Monitor V2.1: AI-Supported Assembly Monitoring with YOLO11

> **Project family:** [V1 Classical CV](https://github.com/MehulTU/Assembly-Monitoring-System) · [V2 YOLO11 pipeline](https://github.com/MehulTU/Assembly-Monitoring-System-V2) · **V2.1 (this repo)** · Author: [@MehulTU](https://github.com/MehulTU)

AI-supported modular assembly monitoring, developed for my M.Sc. thesis in Aerospace Engineering at TU Darmstadt (Germany). The software monitors manual assembly operations with computer vision and YOLO object detection. Its modular architecture lets new detection methods be added without changing the overall software structure.

## What is new in V2.1

V2.1 is a drop-in upgrade for V2. Four files under `Assembly_monitor/` replace their V2 counterparts and two scripts are added.

| File | Change |
| --- | --- |
| `Assembly_monitor/main.py` | Settings-driven YOLO, throttled terminal output |
| `Assembly_monitor/config/settings.py` | New YOLO section |
| `Assembly_monitor/processing/object_detector.py` | Per-class confidence statistics and rolling average |
| `Assembly_monitor/visualization/display.py` | Confidence-coded boxes and a DETECTIONS panel |
| `scripts/train_yolo.py` | New: training, evaluation and automatic deployment |
| `scripts/record_dataset_v2_2.py` | New: fast trial recorder (session workflow) |

## Project structure

```
Assembly-Monitoring-System-V2.1/
├── Assembly_monitor/
├── datasets/
├── scripts/
├── weights/
├── recordings/
├── experiments/
└── README.md
```

## Technologies

Python, OpenCV, YOLO (Ultralytics), NumPy, PyTorch, CVAT (annotation)

## Workflow: record, train, run

1. **Record** trials: `python scripts/record_dataset_v2_2.py`
2. **Annotate**: extract frames, review and finalise them, annotate all images in CVAT and download the YOLO export.
3. **Prepare** the dataset (automatic split by trial, no code edits): `python scripts/prepare_yolo_dataset.py`
4. **Train, evaluate and deploy** in one command: `python scripts/train_yolo.py`
   - trains on `datasets/yolo_dataset_v1`
   - evaluates on the test split (per-class precision, recall and mAP)
   - backs up the old `weights/best.pt` to `weights/backups/` and copies the new `best.pt` to `weights/best.pt`
5. **Run** the live system: `python Assembly_monitor/main.py`

`main.py` loads `weights/best.pt` via `YOLO_MODEL_PATH` in `config/settings.py`, so newly trained objects are detected with no software changes. This is the modular integration point.

## Troubleshooting: low detection confidence (0.25 to 0.45)

Most likely causes, in order of impact:

1. **Unannotated images.** The last dataset had 105 labels for 304 images. Every unannotated image that contains a marker or adapter teaches the model that it is background, which suppresses confidence. Annotate everything before retraining.
2. **Small dataset.** 15 trials of one scene is little data. Record more trials with varied positions, distances, rotations and lighting.
3. **No hard negatives.** The model once labelled a face as "marker" because it had never seen faces, hands or the background labelled as empty. Include `empty_workspace` and `hand_interaction` trials with correct (empty or partial) labels.
4. **Train/inference mismatch.** Live inference uses the same image size (640) as training (`YOLO_IMGSZ` in settings).

After retraining with full annotations, the DETECTIONS panel should move from red and yellow into green (above 60%) for clearly visible objects.

## Future improvements

- Additional object classes
- ROS integration
- Multi-camera support
- Real-time performance optimisation

## Author

**Mehul Patil**, M.Sc. Aerospace Engineering, Technical University of Darmstadt (TU Darmstadt), Germany. Master's thesis project.
