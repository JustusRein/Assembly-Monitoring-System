V2.1 VALIDATION EVIDENCE PACK
============================

PURPOSE
This tool creates plots, CSV tables, and a concise report directly from V2.1 process logs.
It reports observed log behaviour and does NOT invent validation accuracy. If an independent
trials.csv is completed, it also compares recorded transitions/warnings with that ground truth.
An optional dataset inventory graph counts image files under the Blender synthetic_dataset folders.

INSTALL
1. Copy build_v21_validation_evidence.py into:
   D:\Thesis\SOFTWARE V2.1 AI Final\Assembly_monitor

2. Make sure matplotlib is installed in the same virtual environment. Try the run command first.
   If it reports matplotlib missing, run:
   & 'D:\Thesis\SOFTWARE V2.1 AI Final\venv\Scripts\python.exe' -m pip install matplotlib

RUN FROM POWERSHELL
cd 'D:\Thesis\SOFTWARE V2.1 AI Final\Assembly_monitor'
& 'D:\Thesis\SOFTWARE V2.1 AI Final\venv\Scripts\python.exe' .\build_v21_validation_evidence.py --dataset-root 'D:\Thesis\Blender\synthetic_dataset'

OUTPUT
The script writes to a new timestamped folder each run:
validation_output\v21_evidence_YYYYMMDD_HHMMSS\

Main outputs:
- v21_validation_evidence_report.md        summary report
- v21_session_summary.csv                  per-process-log metrics
- v21_transition_events.csv                every recorded stage transition
- v21_warning_events.csv                   every explicitly logged warning
- v21_ground_truth_comparison.csv          only if trial data were entered
- v21_metrics.json                         machine-readable totals
- fig_v21_transition_timeline.png/.pdf     elapsed-time timeline
- fig_v21_session_event_counts.png/.pdf    transitions and warnings by session
- fig_v21_warning_type_counts.png/.pdf     warning-type frequency (if warnings logged)
- fig_v21_session_durations.png/.pdf       duration by session
- fig_v21_ground_truth_stage_counts.png/.pdf only when ground truth has stage counts
- fig_synthetic_dataset_asset_inventory.png/.pdf counts files by top-level folder if --dataset-root used
- LaTeX tables for the descriptive session summary and warning events

GROUND TRUTH (IMPORTANT)
If validation_output\trials.csv does not exist, a blank validation_output\trials_TEMPLATE.csv is
created. Fill trials.csv from independently observed physical assembly/video. Do not infer truth
from the system log. Keep filenames correct. In 'condition' use one of:
nominal | induced_wrong_part | induced_wrong_order | induced_wrong_position
true_stages_completed = number of physical stages actually completed correctly (0-7).
true_stage_route = the physical/observed route, e.g. Stage_01>Stage_02>Stage_03.
error_elapsed_s = elapsed seconds from process session start to the deliberately introduced error.

The script never treats a log's own predicted stage as truth. It does not calculate accuracy if
truth is absent. For induced_wrong_position, the current trials cannot validate position detection
when position validation is disabled. A genuinely unknown object is not one of the seven trained
YOLO classes and may not produce a warning; report such a miss honestly.

SCOPE
By default, only direct *.jsonl files in logs\process are included. Archive older/demo JSONL
files outside that folder before a formal trial batch, but retain the archive for traceability.
The optional dataset inventory is a file count, NOT a training/validation/test split or accuracy
metric. Do not merge counts from all Blender variants and claim it is the 2,800-image YOLO set.

OPTIONAL: EVALUATE THE YOLO MODEL ON THE HELD-OUT TEST SPLIT
-----------------------------------------------------------
The thesis reports YOLO validation-split metrics in Table 5.9. If the root data.yaml under
D:\Thesis\Blender\synthetic_dataset has an explicit test: entry pointing to the held-out test
images (the thesis describes 560 test images), run this additional evaluation to obtain a fresh
set of held-out test metrics and Ultralytics' PR/confusion-matrix plots:

& 'D:\Thesis\SOFTWARE V2.1 AI Final\venv\Scripts\python.exe' .\evaluate_yolo_test_split.py --project-root 'D:\Thesis\SOFTWARE V2.1 AI Final\Assembly_monitor' --data 'D:\Thesis\Blender\synthetic_dataset\data.yaml' --device 0

Run it from the folder containing evaluate_yolo_test_split.py, or provide its full path. The script
checks that data.yaml includes an explicit test: split and refuses to silently evaluate on val/train.
It saves metrics and plots under a new validation_output\yolo_test_evaluation_... folder. If the
YAML belongs to a different dataset variant or does not have a test split, stop and select the
correct data.yaml for the exact 2,800-image dataset described in the thesis. Do not modify the
train/val/test split merely to force the evaluation to run.

This test-split evaluation measures the detector on synthetic images only. Keep it separate from
live camera trial metrics and from end-to-end assembly-state/workflow validation.
