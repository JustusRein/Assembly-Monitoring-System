"""
===============================================================================
evaluate_v18_test.py - detector metrics on the dataset the DEPLOYED model
                       was actually trained on
===============================================================================

WHY THIS SCRIPT EXISTS

The weights the live system loads are:

    D:\\Thesis\\SOFTWARE V2.1 AI Final\\Assembly_monitor\\weights\\best.pt
    SHA256 A0D81E31F540494BEA39904FCF2A47A5130F7F56247BD345EC86688D48D5A312

Those weights carry their own training arguments inside the checkpoint, and
they read:

    data   = D:\\Thesis\\Blender\\synthetic_dataset\\data_combined_v18.yaml
    epochs = 60
    imgsz  = 640
    batch  = 4
    model  = yolo26s.yaml

That YAML draws from FIVE source folders, not two. Any detector metric
reported in the thesis therefore has to be measured on the test split of
THAT dataset - the five 'test' folders the YAML lists - and not on one
folder taken on its own, which is what produced the earlier numbers.

The test images were never seen during training or validation, so this is a
genuine held-out evaluation. The script does not train anything and does not
modify any weights.

WHAT IT WRITES  (into thesis_results/detector_v18/)

    detector_metrics_overall.csv    P, R, mAP50, mAP50-95 for the whole set
    detector_metrics_perclass.csv   the same four, per class
    table_detector_perclass.tex     paste-ready LaTeX table for section 6.2.2
    dataset_counts.csv              images and label instances per folder
    table_dataset_split.tex         paste-ready LaTeX table for section 5.7
    summary.txt                     everything in one readable page
    val/                            Ultralytics' own plots (PR curve, matrix)

Run it with the same Python you use to run main.py - it needs torch and
ultralytics, and nothing else.
===============================================================================
"""

import csv
import sys
from pathlib import Path

WEIGHTS = r"D:\Thesis\SOFTWARE V2.1 AI Final\Assembly_monitor\weights\best.pt"
DATA_YAML = r"D:\Thesis\Blender\synthetic_dataset\data_combined_v18.yaml"
OUT_DIR = Path(r"D:\Thesis\SOFTWARE V2.1 AI Final\Assembly_monitor\thesis_results\detector_v18")

IMGSZ = 640
BATCH = 4
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def banner(text):
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Dataset accounting - done by counting files, not by trusting any document
# ---------------------------------------------------------------------------

def read_yaml_splits(path):
    """
    Minimal reader for the train/val/test lists in the data YAML.

    Deliberately not using PyYAML: this avoids a dependency that may not be
    installed in the inference environment, and the file's structure is a
    plain list of paths under three keys. Anything unexpected is reported
    rather than guessed at.
    """
    splits = {"train": [], "val": [], "test": []}
    current = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.endswith(":") and stripped[:-1] in splits:
            current = stripped[:-1]
            continue
        if stripped.startswith("- ") and current:
            splits[current].append(stripped[2:].strip())
            continue
        # any other top-level key (nc:, names:) ends the list we were reading
        if not line.startswith((" ", "\t", "-")):
            current = None
    return splits


def count_folder(images_dir):
    """Images, label files and label instances in one images/<split> folder."""
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return None

    images = [p for p in images_dir.iterdir()
              if p.suffix.lower() in IMAGE_SUFFIXES]

    # Ultralytics maps images/<split> -> labels/<split> by replacing the
    # last 'images' component of the path. Mirror that exactly.
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    labels_dir = Path(*parts)

    label_files = 0
    instances = 0
    empty = 0
    if labels_dir.exists():
        for image in images:
            label = labels_dir / f"{image.stem}.txt"
            if not label.exists():
                continue
            label_files += 1
            rows = [r for r in label.read_text(encoding="utf-8").splitlines()
                    if r.strip()]
            instances += len(rows)
            if not rows:
                empty += 1

    return {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "images": len(images),
        "label_files": label_files,
        "instances": instances,
        "empty_labels": empty,
        "missing_labels": len(images) - label_files,
    }


def dataset_report(splits):
    rows = []
    totals = {}
    for split in ("train", "val", "test"):
        split_images = 0
        split_instances = 0
        for folder in splits[split]:
            info = count_folder(folder)
            if info is None:
                print(f"  MISSING: {folder}")
                rows.append({"split": split, "source": folder,
                             "images": 0, "instances": 0,
                             "missing_labels": 0, "note": "FOLDER NOT FOUND"})
                continue
            name = Path(folder).parts[-3] if len(Path(folder).parts) >= 3 else folder
            rows.append({"split": split, "source": name,
                         "images": info["images"],
                         "instances": info["instances"],
                         "missing_labels": info["missing_labels"],
                         "note": ""})
            split_images += info["images"]
            split_instances += info["instances"]
            flag = ""
            if info["missing_labels"]:
                flag = f"   <-- {info['missing_labels']} images have NO label file"
            print(f"  {split:<6} {name:<28} {info['images']:>5} images"
                  f"  {info['instances']:>6} instances{flag}")
        totals[split] = (split_images, split_instances)
        print(f"  {split:<6} {'TOTAL':<28} {split_images:>5} images"
              f"  {split_instances:>6} instances")
        print()
    grand = sum(v[0] for v in totals.values())
    print(f"  dataset total: {grand} images")
    return rows, totals, grand


# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    weights = Path(WEIGHTS)
    if not weights.exists():
        sys.exit(f"Weights not found: {weights}")
    if not Path(DATA_YAML).exists():
        sys.exit(f"Data YAML not found: {DATA_YAML}")

    banner("1. DATASET - counted from the folders the YAML points at")
    splits = read_yaml_splits(DATA_YAML)
    for split in ("train", "val", "test"):
        if not splits[split]:
            sys.exit(f"The YAML has no '{split}:' entries - cannot continue.")
    print(f"  YAML lists {len(splits['train'])} train, {len(splits['val'])} val, "
          f"{len(splits['test'])} test source folders\n")
    count_rows, totals, grand_total = dataset_report(splits)

    with (OUT_DIR / "dataset_counts.csv").open("w", newline="",
                                               encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "source", "images",
                                                "instances", "missing_labels",
                                                "note"])
        writer.writeheader()
        writer.writerows(count_rows)

    banner("2. MODEL")
    import torch
    from ultralytics import YOLO

    cuda = torch.cuda.is_available()
    device = 0 if cuda else "cpu"
    print(f"  torch {torch.__version__}   CUDA available: {cuda}   device: {device}")

    model = YOLO(str(weights))

    # The checkpoint records how it was trained. Print it so the thesis text
    # can be written from the file itself rather than from recollection.
    try:
        checkpoint = torch.load(str(weights), map_location="cpu",
                                weights_only=False)
        args = checkpoint.get("train_args") or checkpoint.get("args") or {}
        if hasattr(args, "__dict__"):
            args = vars(args)
        print("\n  training arguments recorded inside the checkpoint:")
        for key in ("data", "model", "epochs", "patience", "batch", "imgsz",
                    "optimizer", "lr0", "lrf", "momentum", "weight_decay",
                    "warmup_epochs", "seed", "pretrained", "single_cls"):
            if key in args:
                print(f"      {key:<16} {args[key]}")
        recorded_epochs = args.get("epochs")
        recorded_data = args.get("data")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  (could not read embedded training args: {exc})")
        recorded_epochs = recorded_data = None

    print(f"\n  class names in the weights: {model.names}")

    banner("3. EVALUATION on the held-out test split (five folders)")
    print("  Standard detection protocol: IoU 0.50 for mAP50, 0.50:0.95")
    print("  averaged for mAP50-95. Precision and recall are reported at the")
    print("  model's best-F1 operating point, which is how YOLO reports them")
    print("  and is independent of the 0.65 runtime threshold used live.\n")

    results = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        plots=True,
        verbose=True,
        project=str(OUT_DIR),
        name="val",
        exist_ok=True,
    )

    box = results.box
    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
    }

    banner("4. RESULTS")
    print(f"  test images    : {totals['test'][0]}")
    print(f"  test instances : {totals['test'][1]}")
    print()
    print(f"  Precision  (P)     {overall['precision'] * 100:6.2f} %")
    print(f"  Recall     (R)     {overall['recall'] * 100:6.2f} %")
    print(f"  mAP@0.50           {overall['mAP50'] * 100:6.2f} %")
    print(f"  mAP@0.50:0.95      {overall['mAP50_95'] * 100:6.2f} %")
    print()

    names = model.names
    per_class = []
    print(f"  {'class':<14} {'images':>7} {'inst':>7} {'P %':>8} {'R %':>8} "
          f"{'mAP50 %':>9} {'mAP50-95 %':>11}")
    print("  " + "-" * 70)
    for idx, class_id in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(idx)
        label = names[int(class_id)]
        instances = int(box.nt_per_class[int(class_id)]) \
            if hasattr(box, "nt_per_class") else 0
        images = int(box.nt_per_image[int(class_id)]) \
            if hasattr(box, "nt_per_image") else 0
        per_class.append({
            "class": label,
            "images": images,
            "instances": instances,
            "precision": float(p),
            "recall": float(r),
            "mAP50": float(ap50),
            "mAP50_95": float(ap),
        })
        print(f"  {label:<14} {images:>7} {instances:>7} {p * 100:>8.2f} "
              f"{r * 100:>8.2f} {ap50 * 100:>9.2f} {ap * 100:>11.2f}")

    weakest = min(per_class, key=lambda c: c["mAP50_95"]) if per_class else None
    if weakest:
        print(f"\n  weakest class by mAP50-95: {weakest['class']} "
              f"({weakest['mAP50_95'] * 100:.2f} %)")

    # ---------------- files ----------------
    with (OUT_DIR / "detector_metrics_overall.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value_percent"])
        writer.writerow(["precision", f"{overall['precision'] * 100:.2f}"])
        writer.writerow(["recall", f"{overall['recall'] * 100:.2f}"])
        writer.writerow(["mAP50", f"{overall['mAP50'] * 100:.2f}"])
        writer.writerow(["mAP50_95", f"{overall['mAP50_95'] * 100:.2f}"])
        writer.writerow(["test_images", totals["test"][0]])
        writer.writerow(["test_instances", totals["test"][1]])

    with (OUT_DIR / "detector_metrics_perclass.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["class", "images", "instances",
                                                "precision", "recall",
                                                "mAP50", "mAP50_95"])
        writer.writeheader()
        for row in per_class:
            writer.writerow({**row,
                             "precision": f"{row['precision'] * 100:.2f}",
                             "recall": f"{row['recall'] * 100:.2f}",
                             "mAP50": f"{row['mAP50'] * 100:.2f}",
                             "mAP50_95": f"{row['mAP50_95'] * 100:.2f}"})

    # per-class LaTeX table
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-class detection performance of the deployed YOLO26s "
        r"model on the held-out test split of the combined synthetic dataset "
        rf"({totals['test'][0]} images, {totals['test'][1]} labelled "
        r"instances). Precision and recall are reported at the best-F$_1$ "
        r"operating point.}",
        r"\label{tab:detector_perclass}",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Class & Images & Instances & P (\%) & R (\%) & "
        r"mAP@0.5 (\%) & mAP@0.5:0.95 (\%) \\",
        r"\hline",
    ]
    for row in per_class:
        tex.append(
            f"{row['class'].replace('_', chr(92) + '_')} & "
            f"{row['images']} & {row['instances']} & "
            f"{row['precision'] * 100:.2f} & {row['recall'] * 100:.2f} & "
            f"{row['mAP50'] * 100:.2f} & {row['mAP50_95'] * 100:.2f} \\\\")
    tex += [
        r"\hline",
        f"\\textbf{{All classes}} & {totals['test'][0]} & "
        f"{totals['test'][1]} & "
        f"\\textbf{{{overall['precision'] * 100:.2f}}} & "
        f"\\textbf{{{overall['recall'] * 100:.2f}}} & "
        f"\\textbf{{{overall['mAP50'] * 100:.2f}}} & "
        f"\\textbf{{{overall['mAP50_95'] * 100:.2f}}} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUT_DIR / "table_detector_perclass.tex").write_text(
        "\n".join(tex), encoding="utf-8")

    # dataset-split LaTeX table
    sources = []
    for row in count_rows:
        if row["source"] not in sources and row["note"] != "FOLDER NOT FOUND":
            sources.append(row["source"])
    by_source = {s: {"train": 0, "val": 0, "test": 0} for s in sources}
    for row in count_rows:
        if row["source"] in by_source:
            by_source[row["source"]][row["split"]] += row["images"]

    tex2 = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Composition of the combined synthetic training dataset "
        r"used to train the deployed detector. Each source folder was split "
        r"60/20/20 into training, validation and test partitions before "
        r"combination; the test partition was held out from both training "
        r"and validation.}",
        r"\label{tab:dataset_split}",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Source set & Train & Validation & Test & Total \\",
        r"\hline",
    ]
    for source in sources:
        counts = by_source[source]
        total = counts["train"] + counts["val"] + counts["test"]
        tex2.append(f"{source.replace('_', chr(92) + '_')} & "
                    f"{counts['train']} & {counts['val']} & "
                    f"{counts['test']} & {total} \\\\")
    tex2 += [
        r"\hline",
        f"\\textbf{{Total}} & \\textbf{{{totals['train'][0]}}} & "
        f"\\textbf{{{totals['val'][0]}}} & \\textbf{{{totals['test'][0]}}} & "
        f"\\textbf{{{grand_total}}} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUT_DIR / "table_dataset_split.tex").write_text(
        "\n".join(tex2), encoding="utf-8")

    # summary
    lines = [
        "DETECTOR EVALUATION - deployed model, correct test split",
        "=" * 70,
        f"weights            {weights}",
        "sha256             A0D81E31F540494BEA39904FCF2A47A5130F7F5"
        "6247BD345EC86688D48D5A312",
        f"data yaml          {DATA_YAML}",
        f"recorded in ckpt   data={recorded_data}  epochs={recorded_epochs}",
        f"inference imgsz    {IMGSZ}",
        "",
        "DATASET",
        f"  train            {totals['train'][0]} images, "
        f"{totals['train'][1]} instances",
        f"  validation       {totals['val'][0]} images, "
        f"{totals['val'][1]} instances",
        f"  test (held out)  {totals['test'][0]} images, "
        f"{totals['test'][1]} instances",
        f"  total            {grand_total} images",
        "",
        "TEST-SPLIT PERFORMANCE",
        f"  Precision        {overall['precision'] * 100:.2f} %",
        f"  Recall           {overall['recall'] * 100:.2f} %",
        f"  mAP@0.50         {overall['mAP50'] * 100:.2f} %",
        f"  mAP@0.50:0.95    {overall['mAP50_95'] * 100:.2f} %",
        "",
        "PER CLASS",
    ]
    for row in per_class:
        lines.append(f"  {row['class']:<14} inst {row['instances']:>5}   "
                     f"P {row['precision'] * 100:6.2f}   "
                     f"R {row['recall'] * 100:6.2f}   "
                     f"mAP50 {row['mAP50'] * 100:6.2f}   "
                     f"mAP50-95 {row['mAP50_95'] * 100:6.2f}")
    (OUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    banner("5. WRITTEN")
    for name in sorted(p.name for p in OUT_DIR.iterdir()):
        print(f"  {name}")
    print(f"\n  folder: {OUT_DIR}")
    print("\n  Paste the contents of summary.txt back into the chat.")


if __name__ == "__main__":
    main()
