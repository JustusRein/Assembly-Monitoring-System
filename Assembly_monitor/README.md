# `Assembly_monitor/` — the monitoring software

This folder is the application itself. Everything else in the repository supports it: `scripts/` prepares datasets and trains detectors, `logs/` holds what the software recorded, `V21_validation_evidence_pack/` turns those logs into figures.

**What it does:** a fixed RGB camera watches a manual assembly workstation. Each frame is checked, the region of interest is cropped, a YOLO detector reports which components are visible, and a state machine decides which of the seven assembly stages the operator has reached. Confirmed stages, their durations and any sequence deviations are written to a structured log.

For the project background, results and evidence boundary, see the [main README](../README.md).

---

## Quick start

From the **repository root**, not from inside this folder:

```bash
pip install -r requirements.txt
python Assembly_monitor/check_setup.py     # tells you what is missing
python Assembly_monitor/main.py            # run the live system
```

Press **`q`** to quit.

`check_setup.py` is the one to run first. It prints a MUST FIX / WORTH FIXING summary covering settings, the stage preview, position calibration, the body-pose tracker and wrong-part detection. Nothing is silently disabled — if something is not working it says so.

## Before it will run

| Needed | Where it goes | If missing |
| --- | --- | --- |
| Trained weights | `weights/best.pt` (path set by `YOLO_MODEL_PATH`) | Fatal — the detector cannot load |
| A camera | `CAMERA_INDEX = 0` in `config/settings.py` | Fatal — "Could not open camera" |
| Region of interest | `calibration/roi_config.json` | Runs, but processes the whole frame |
| Stage preview frames | `STAGE_PREVIEW_DIR` in `config/settings.py` | Runs, preview panel is skipped |

Define the region of interest with:

```bash
python Assembly_monitor/tools/set_work_area.py
```

### Settings you will have to change

`config/settings.py` contains absolute paths from the development machine. At minimum:

```python
CAMERA_INDEX      = 0                  # which webcam
CAMERA_TYPE       = "logitech"         # "logitech" or "realsense"
YOLO_MODEL_PATH   = .../weights/best.pt
YOLO_CONFIDENCE   = 0.65               # detection threshold
YOLO_IMGSZ        = 640                # must match training
STAGE_PREVIEW_DIR = r"D:\Thesis\..."   # change or the preview is skipped
```

`YOLO_IMGSZ` must match the size the detector was trained at. A mismatch here quietly lowers confidence and is a common cause of poor live performance.

---

## While it is running

The window shows the camera view with detections, the current stage, what the system is still waiting for, and — when the frames are available — a rotating 3-D preview of the stage being built.

| Key | Action |
| --- | --- |
| `q` | Quit |
| `r` | Start / stop recording to `recordings/` |
| `s` | Save a snapshot |
| `v` | Mark the final product as operator-verified in the process log |
| `0` | Reset the state machine to the start |
| `1`–`7` | Jump to a stage (for testing, or to recover after a misread) |
| `e` | Toggle redo mode — jump to a stage without marking it confirmed |
| `b` | Re-learn the empty bench. Use after deliberately changing the scene, otherwise the change is reported as a foreign object |
| `p` | Pause / resume the preview rotation |
| `,` `.` | Step the preview one frame back / forward |

## What it writes

| Output | Contents |
| --- | --- |
| `logs/T-01_*.csv` | Per-session event log — activity intervals and timing |
| `logs/process/process_*.jsonl` | Structured process log — confirmed stages, transitions, durations, `WRONG_ORDER` warnings, session summary |
| `recordings/*.mp4` | Video, only while recording is on |

The JSONL process log is the file the validation tooling reads. `analyze_v21_validation.py` turns it into the session summaries and tables in `validation_output/`.

---

## How the code is organised

Each subsystem is a package with a defined interface, so one can be replaced without touching the others. This is what allowed the classical-vision prototype to become the YOLO-based one without redesigning the pipeline.

| Package | Responsibility |
| --- | --- |
| `acquisition/` | Camera interface and frame validation. `camera_factory.py` picks the backend, so the rest of the code never knows which camera is attached |
| `processing/` | Turns pixels into observations — `motion_detector.py` (classical), `object_detector.py` (YOLO), `foreign_object_detector.py` |
| `decision/` | **The interpretation layer.** `assembly_state_machine.py` maps detections to stages, `position_validator.py` and `seat_finder.py` check placement, `event_detector.py` finds activity intervals |
| `analytics/` | Session metrics from the event stream |
| `storage/` | `process_log.py` (structured JSONL) and `session_logger.py` (CSV) |
| `visualization/` | `display.py` (live window) and `stage_preview.py` (rotating stage view) |
| `config/` | `settings.py` — every threshold and path in one file |
| `tools/` | Calibration and measurement utilities — see below |
| `thesis_results/` | Detector metrics, resolution study and trial results |
| `validation_output/` | Analysis output from the recorded process logs |

### The detection → stage rule

The detector reports which classes it sees and how many. The state machine compares that composition against what each stage should contain:

| Stage | Required composition |
| --- | --- |
| Stage_01 | `body_a` ×1, `bushing` ×1 |
| Stage_02 | `body_a` ×1, `bushing` ×2 |
| Stage_03 | `body_a` ×1, `body_b` ×1, `bushing` ×2 |
| Stage_04 | + `block_pin` ×1 |
| Stage_05 | + `nut` ×1 |
| Stage_06 | `knob` ×1, `threaded_pin` ×1 |
| Stage_07 | all seven |

A candidate stage only becomes the confirmed state when **at least 3 of the last 5 frames agree**, so one noisy frame can never move the process forward. Stages cannot be skipped: while the current stage is unconfirmed, evidence matching the next stage does nothing.

> **The sequence can be overridden by a file.** If `assembly_sequence.json` exists at the path in `settings.py`, it replaces the built-in order entirely. If that file has the wrong order, the running system uses the wrong order regardless of the code. If it is missing, a warning is printed at startup and the built-in order is used. This is deliberate — it is what makes the method portable to a different product without editing code.

## Tools

| Script | Purpose |
| --- | --- |
| `tools/set_work_area.py` | Define the workspace region of interest |
| `tools/calibrate_position_anchors.py` | Calibrate position anchors for placement checking |
| `tools/resolution_study.py` | Reproduce the mAP-vs-input-size study |
| `tools/evaluate_test_split.py` | Score a checkpoint on the held-out test split |
| `tools/score_trial.py` | Score a recorded trial against a reference |
| `tools/test_camera_resolution.py` | Check what resolutions the camera actually supports |
| `analyze_v21_validation.py` | Build session summaries and tables from the process logs |
| `check_setup.py` | Pre-flight check — run this first |

---

## Development notes

Two notes kept from development, because they record decisions that are still load-bearing.

<details>
<summary><b>Stage_01 was being skipped, and the display disagreed with the system</b></summary>

**The bug.** Showing `body_a` with both bushings at the start jumped straight to Stage_02 — Stage_01 was never confirmed. In practice the first thing the system waited for was *two* bushings.

**Cause.** While the current stage was still unconfirmed, evidence that happened to match the *next* stage was allowed to advance the sequence. If both bushings were in view at the start — or the detector double-detected one — Stage_01 was bypassed entirely.

**Fix.** A stage can no longer be skipped. While the current stage is unconfirmed, matching the next stage does nothing. Each stage must be confirmed in turn, in order.

```
both bushings at start  -> stays unconfirmed, still asking for Stage_01
body_a + 1 bushing      -> Stage_01
then 2nd bushing        -> Stage_02
```

**Display and logic now read from one source.** The panel used to print a worded hint while the state machine tested a different dictionary, so any mismatch was invisible on screen. The panel now prints the live requirement straight from `current_requirement`, so the two cannot disagree.

**Regression suite — 10/10:**

| Check | Result |
|---|---|
| Both bushings at start does not skip Stage_01 | PASS |
| Genuine assembly reaches Stage_07 | PASS |
| One bushing held forever stays Stage_01 | PASS |
| Detector glitch does not advance | PASS |
| False Stage_02 gets revoked | PASS |
| Foreign part → WRONG_PART | PASS |
| Skip ahead → WRONG_ORDER | PASS |
| Blank camera, 400 frames, no drift | PASS |
| Slow pace + heavy occlusion reaches Stage_07 | PASS |
| Manual stage control + reset | PASS |

**Still open.** None of this makes the detector find the nut, threaded pin or knob. If the detector never outputs `nut`, the nut stage cannot confirm — the signal never arrives. That is a detection problem, not a logic problem, and it is why the per-class recall figures in the main README matter.

</details>

<details>
<summary><b>Next-stage 360° preview</b></summary>

The preview column shows two rotating views: the stage currently being worked toward, and the one immediately after it. Both are driven by the same `StagePreviewPlayer` and share the `p`, `,` and `.` controls. At Stage_07 there is no next stage, so the single-preview layout is used. Missing preview frames are non-fatal — the panel is simply skipped.

The preview does not touch the detector, the event detector or the state-transition logic.

</details>
