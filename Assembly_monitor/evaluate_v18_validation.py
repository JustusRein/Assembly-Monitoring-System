
from pathlib import Path
from ultralytics import YOLO
import torch

weights = Path(
    r"D:\Thesis\SOFTWARE V2.1 AI Final\Assembly_monitor\weights\best.pt"
)
data_yaml = Path(
    r"D:\Thesis\Blender\synthetic_dataset\data_combined_v18.yaml"
)

if not weights.is_file():
    raise FileNotFoundError(f"Model not found: {weights}")

if not data_yaml.is_file():
    raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Use the same working Python environment "
        "as the completed test evaluation."
    )

model = YOLO(str(weights))

results = model.val(
    data=str(data_yaml),
    split="val",
    imgsz=640,
    batch=4,
    device=0,
    workers=0,
    plots=True,
    project=(
        r"D:\Thesis\SOFTWARE V2.1 AI Final\Assembly_monitor"
        r"\thesis_results\detector_v18"
    ),
    name="validation_same_checkpoint_20261012",
    exist_ok=False,
)

box = results.box

print("\n===== VALIDATION RESULTS: SAME CHECKPOINT =====")
print(f"Images and instances: see the Ultralytics summary above")
print(f"Precision:       {box.mp * 100:.2f}%")
print(f"Recall:          {box.mr * 100:.2f}%")
print(f"mAP@0.50:        {box.map50 * 100:.2f}%")
print(f"mAP@0.50:0.95:   {box.map * 100:.2f}%")

names = model.names
for i, class_id in enumerate(box.ap_class_index):
    class_name = names[int(class_id)]
    print(
        f"{class_name}: "
        f"P={box.p[i] * 100:.2f}%, "
        f"R={box.r[i] * 100:.2f}%, "
        f"mAP50={box.ap50[i] * 100:.2f}%, "
        f"mAP50-95={box.ap[i] * 100:.2f}%"
    )

print(f"\nSaved evaluation plots: {results.save_dir}")
