# yolo_dataset_v1 :- early two-class development dataset

**This is not the thesis dataset.** It is kept for history.

This folder holds the first real-image dataset recorded while the detection
pipeline was being built, before the industrial use case was available. It
contains 304 annotated images of two generic objects:

| id | class |
| --- | --- |
| 0 | `marker` |
| 1 | `power_adapter` |

It was used to verify the record → extract → annotate → train → deploy
workflow end to end, and to expose the problems that shaped the final
design: unannotated frames suppressing confidence, missing hard negatives,
and train/inference image-size mismatch.

## The dataset actually used in the thesis

The deployed detector was trained on **6,551 CAD-derived synthetic images**
with **8,714 annotated instances** over the seven lever-lock component
classes (`body_a`, `body_b`, `block_pin`, `bushing`, `nut`, `knob`,
`threaded_pin`), split 3,916 train / 1,316 val / 1,319 test.

That dataset is produced by a separate Blender generation pipeline and is
not stored here because of its size. See the main [README](../../README.md).
