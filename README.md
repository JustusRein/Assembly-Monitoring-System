# Assembly Monitor V2.1 — Vision-Based Monitoring of Manual Assembly

> **Project family:** [V1 Classical CV](https://github.com/MehulTU/Assembly-Monitoring-System) · [V2 YOLO pipeline](https://github.com/MehulTU/Assembly-Monitoring-System-V2) · **V2.1 (this repo)** · Author: [@MehulTU](https://github.com/MehulTU)

Final software from my M.Sc. thesis *AI-Supported Ergonomic and Productivity Analysis in Manual Assembly*, Institute of Product Life Cycle Management (PLCM), Technische Universität Darmstadt.

The system watches a manual assembly workstation with a single RGB camera and turns what it sees into structured process data: which components are present, which assembly stage the operator has reached, how long each stage took, and whether the observed sequence deviates from the expected one.

---

## What the system does

```
camera frame
   → component detection          (YOLO, 7 classes)
   → composition per frame        (which parts + how many)
   → candidate assembly stage     (matched against the 7-stage reference)
   → temporal confirmation        (3 of the last 5 frames must agree)
   → confirmed stage + process log (timestamp, duration, WRONG_ORDER warnings)
```

The key design decision is that **perception and interpretation are separate layers**. The detector only reports what it sees. The state machine decides what that means, driven by a sequence table rather than by code. Either side can be replaced without touching the other.

### Use case

An industrial lever-lock subassembly supplied by a partner company: **8 physical components, 7 detector classes** (the two bushings are one class), assembled in **7 defined stages**.

| Class | Stage sequence |
| --- | --- |
| `body_a`, `body_b`, `block_pin`, `bushing`, `nut`, `knob`, `threaded_pin` | S01 base body + 1 bushing → S02 + 2 bushings → S03 + body_b → S04 + block_pin → S05 + nut → S06 locking subassembly (threaded pin + knob) → S07 complete |

---

## Repository structure

```
Assembly-Monitoring-System-V2.1/
├── Assembly_monitor/            the monitoring software (modular package)
│   ├── acquisition/             camera interface, frame validation
│   ├── processing/              motion detection, YOLO object detection
│   ├── decision/                assembly state machine, event + position logic
│   ├── analytics/               session metrics
│   ├── storage/                 process log, session logger
│   ├── visualization/           live display, stage preview
│   ├── config/                  settings.py, position config
│   ├── tools/                   calibration and measurement utilities
│   ├── thesis_results/          detector metrics, resolution study, trial results
│   ├── validation_output/       V2.1 validation analysis output
│   └── main.py                  entry point for the live system
├── scripts/                     dataset preparation, annotation, training, offline inference
├── V21_validation_evidence_pack/  scripts that rebuild the thesis figures and tables
├── datasets/                    dataset metadata and annotation exports
├── experiments/                 experiment plans
├── logs/                        V1 baseline session CSVs and V2.1 process logs (JSONL)
├── models/ · runs/              training and inference outputs
└── archive/                     superseded code kept for traceability
```

Large binaries are deliberately **not** in the repository — see [What is not here](#what-is-not-here).

---

## Running it

Requires Python 3.12, a webcam, and the trained weights at `weights/best.pt`.

```bash
pip install -r requirements.txt        # ultralytics, opencv-python, numpy, torch
python Assembly_monitor/check_setup.py # verifies camera, weights, class names, paths
python Assembly_monitor/main.py        # live monitoring
```

`main.py` loads the detector from `YOLO_MODEL_PATH` in `Assembly_monitor/config/settings.py`, so a retrained checkpoint is picked up with no code change. That is the modular integration point.

Useful utilities:

| Command | Purpose |
| --- | --- |
| `python Assembly_monitor/check_setup.py` | Pre-flight check of camera, weights and class names |
| `python Assembly_monitor/tools/set_work_area.py` | Define the workspace region of interest |
| `python Assembly_monitor/analyze_v21_validation.py` | Analyse recorded process logs |
| `python scripts/train_yolo.py` | Train, evaluate and deploy a detector from an annotated real-image dataset |
| `python scripts/run_offline_inference.py` | Run the detector over stored frames |

---

## Results

**Detector**, evaluated offline on held-out synthetic images:

| Split | Images | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
| --- | --- | --- | --- | --- | --- |
| Validation | 1,316 | 91.64 % | 83.14 % | 91.93 % | 69.38 % |
| Test (held out) | 1,319 | 89.15 % | 86.24 % | 92.54 % | 69.90 % |

Weakest classes: `threaded_pin` (recall 70.56 %, small and partly inside the body once inserted) and `body_b` (mAP@0.50:0.95 61.61 %, occluded once the halves are joined). Both are occlusion and scale effects rather than class confusion.

**Training data:** 6,551 images from five sources, 8,714 annotated instances, split 3,916 / 1,316 / 1,319. The dataset was generated from CAD geometry in Blender — bounding boxes are computed from the known 3-D geometry and camera, so no label was drawn by hand. The generator lives in a separate project; this repository consumes its output.

**Integrated runtime:** four live sessions produced 60 state records and 11 `WRONG_ORDER` warnings, with session durations of 64.52–184.23 s and stage times of 0.17–13.12 s.

### What the evidence does and does not support

This is stated plainly because the distinction matters:

- **Quantified** — detector performance on held-out *synthetic* images, and the camera-viewpoint study.
- **Demonstrated but not scored** — the integrated runtime. The live sessions have no independently annotated reference of the physical stage transitions or of when deviations were introduced, so the 11 warning records establish that the warning function was active, **not** its sensitivity, specificity or response delay.
- **Not implemented** — human pose estimation and RULA/REBA ergonomic scoring. These are defined in the requirements and the architecture provides the interface, but they were not built or evaluated.

The thesis targets of workflow accuracy > 90 % and timing error < 10 % therefore remain **unscored**, and no end-to-end "system accuracy" figure is reported. Converting the descriptive counts into rates requires a controlled trial that records stage transitions and error-introduction times independently.

---

## What is not here

Excluded by `.gitignore` to keep the repository usable:

| Excluded | Why |
| --- | --- |
| `weights/*.pt` | Model checkpoints (~20 MB each) |
| `recordings/`, `*.mp4` | Session video |
| `datasets/` image and label folders | Thousands of training images |
| `runs/` | Ultralytics training output |
| `venv/`, `__pycache__/` | Environment and build artefacts |

The trained checkpoint and the synthetic dataset are available on request.

---

## Related

- **V1** — classical computer-vision baseline: greyscale, background subtraction, morphology, contour motion score, temporal filter. Produced ACTIVE/IDLE intervals but no component identity. The limitation that motivated V2.
- **V2 / V2.1** — object detection plus assembly-state interpretation. V2.1 adds the state machine, the structured process log, sequence-deviation warnings and the validation tooling.
- Early classical-CV learning scripts are kept in `Assembly_monitor/src(all basic codes)/`.

## Technologies

Python · OpenCV · Ultralytics YOLO · PyTorch · NumPy · Blender (synthetic data) · CVAT (annotation)

## Author

**Mehul Patil** — M.Sc. Aerospace Engineering, Technische Universität Darmstadt, Germany.
Master's thesis at the Institute of Product Life Cycle Management (PLCM).
