# V2.1 validation evidence report

Generated: 2026-10-09T16:51:08
Project root: `D:\Thesis\SOFTWARE V2.1 AI Final\Assembly_monitor`
Process-log directory: `D:\Thesis\SOFTWARE V2.1 AI Final\Assembly_monitor\logs\process`
Number of JSONL logs analysed: **5**

## Descriptive log evidence

- State records (using session-end counters where present): **95**
- Stage-transition records: **28**
- Warning records (using session-end counters where present): **13**
- Explicit warning-event rows available for plotting: **13**

These are counts from software logs, not accuracy percentages. A transition or warning is not by itself proof that the physical assembly stage or error was correct.

## Ground truth

Ground-truth rows matched to logs: **3**. Rows with a numeric true-stage count: **0**.
Inspect `v21_ground_truth_comparison.csv` before reporting any rate. Route comparison is only scored where the true route was entered.

## Per-session records

| Log | States | Transition records | Warning records | Highest stage logged | Final verified |
|---|---:|---:|---:|---|---|
| process_20261009_160234.jsonl | 14 | 7 | 1 | 7 | True |
| process_20261009_161029.jsonl | 6 | 2 | 0 | 2 | False |
| process_20261009_161525.jsonl | 12 | 3 | 1 | 3 | False |
| process_20261009_163818.jsonl | 28 | 9 | 9 | 7 | False |
| process_20261009_164253.jsonl | 35 | 7 | 2 | 7 | False |

## Generated figures

- `fig_v21_transition_timeline.png`
- `fig_v21_transition_timeline.pdf`
- `fig_v21_session_event_counts.png`
- `fig_v21_session_event_counts.pdf`
- `fig_v21_warning_type_counts.png`
- `fig_v21_warning_type_counts.pdf`
- `fig_v21_session_durations.png`
- `fig_v21_session_durations.pdf`
- `fig_synthetic_dataset_asset_inventory.png`
- `fig_synthetic_dataset_asset_inventory.pdf`

## Interpretation boundaries

1. Sessions in the selected process directory are included even if they were pilots. Move older/demo logs outside `logs/process` before a formal trial series; do not remove raw data, archive them elsewhere.
2. Position validation was disabled in the reported runs. Do not interpret these logs as evidence of wrong-position detection.
3. An unknown object outside the seven YOLO classes may not be detected. Report a missed unknown-object test as a negative observation.
4. The optional dataset folder inventory is only a count of image files by top-level directory. It is not a dataset split, annotation-quality result, or detector-performance score.
5. Keep system logs and the independent physical ground truth separate. Never use the system's own predicted stage as the ground-truth answer.
