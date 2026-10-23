# `weights/` :- trained model checkpoints

**This folder is empty in the repository on purpose.** Model checkpoints are
~20 MB each, and binary files that change every training run do not belong in
git history.

## What goes here

| File | What it is |
| --- | --- |
| `best.pt` | The detector the live system loads. Seven lever-lock component classes. |

`Assembly_monitor/config/settings.py` points at it:

```python
YOLO_MODEL_PATH = str(PROJECT_ROOT / "weights" / "best.pt")
```

## How to get it

Download `best.pt` from the [Releases page](https://github.com/MehulTU/Assembly-Monitoring-System-V2.1/releases) and place it in this
folder. Then:

```bash
python Assembly_monitor/check_setup.py
```

If the weights are missing or in the wrong place, `check_setup.py` says so
before you waste time on it.

## Training your own

`scripts/train_yolo.py` trains a detector from an annotated real-image dataset,
evaluates it on the test split, backs up the previous `best.pt` and copies the
new one into this folder - so the live system picks it up with no code change.

```bash
python scripts/train_yolo.py
```

The detector reported in the thesis was trained separately, on CAD-derived
synthetic images. See [Where the training data came from](../README.md#where-the-training-data-came-from).
