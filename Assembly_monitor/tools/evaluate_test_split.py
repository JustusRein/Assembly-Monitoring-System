"""
evaluate_test_split.py
========================================================================
RUN THE HELD-OUT 560-IMAGE TEST SPLIT AND WRITE THESIS-READY RESULTS.

WHY THIS MATTERS FOR THE THESIS
-------------------------------
Table 6.10 currently says, for the "Independent detector test" objective:

    "A separate 560-image test subset is defined in the dataset, but its
     independent metrics are not reported in the current results."
     Status: Prepared; test result not reported.

The data exists. The model exists. This script closes that row. It is the
only quantitative result in the whole thesis you can add without doing any
physical work at all.

WHAT IT PRODUCES
----------------
Everything lands in  thesis_results/detector/  :

    test_metrics.tex          Table: precision, recall, mAP50, mAP50-95
    test_per_class.tex        Table: the same, per component class
    val_vs_test.tex           Table: validation vs test, side by side
    test_metrics.png          Bar chart of the four headline metrics
    confusion_matrix.png      Which classes get confused with which
    test_results.json         Raw numbers, so nothing is retyped by hand
    REPORT.txt                Plain-language summary of what was measured

HOW TO RUN
----------
    python tools\\evaluate_test_split.py

  It looks for the dataset YAML automatically. If it cannot find it:

    python tools\\evaluate_test_split.py --data "D:\\path\\to\\data.yaml"

WHAT THIS RESULT IS, AND IS NOT
-------------------------------
It is an INDEPENDENT measurement of the detector inside the synthetic data
distribution. The test images were never used for training or for choosing
the training configuration.

It is NOT evidence of performance on the physical workstation, because both
splits are synthetic. Section 6.5 of the thesis already makes this argument;
this script prints the same caveat into REPORT.txt so the wording you paste
into the thesis stays consistent with it.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "thesis_results" / "detector"

# Values already reported in the thesis (Table 6.3), used only to build the
# side-by-side validation/test comparison table.
THESIS_VALIDATION = {
    "precision": 0.9420,
    "recall": 0.8918,
    "mAP50": 0.9578,
    "mAP50_95": 0.7655,
}


def find_data_yaml(explicit=None):
    """Locate the dataset YAML that defines the train/val/test splits."""
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        raise SystemExit(f"--data given but not found: {p}")

    candidates = []
    try:
        from config.settings import ASSEMBLY_SEQUENCE_JSON_PATH
        seq = Path(ASSEMBLY_SEQUENCE_JSON_PATH)
        # .../synthetic_dataset/subassembly/assembly_sequence.json
        for base in (seq.parent, seq.parent.parent, seq.parent.parent.parent):
            candidates.append(base)
    except Exception:                                       # noqa: BLE001
        pass

    candidates += [PROJECT_ROOT, PROJECT_ROOT / "weights",
                   Path(r"D:\Thesis\Blender\synthetic_dataset")]

    seen = set()
    for base in candidates:
        try:
            base = Path(base)
        except Exception:                                   # noqa: BLE001
            continue
        if not base.is_dir() or base in seen:
            continue
        seen.add(base)
        for name in ("data.yaml", "dataset.yaml", "data.yml"):
            hit = base / name
            if hit.is_file():
                return hit
        for hit in list(base.rglob("data.yaml"))[:5]:
            return hit

    raise SystemExit(
        "\nCould not find the dataset YAML automatically.\n"
        "Re-run and point at it, for example:\n"
        '    python tools\\evaluate_test_split.py --data '
        '"D:\\Thesis\\Blender\\synthetic_dataset\\data.yaml"\n'
    )


def tex_escape(s):
    return str(s).replace("_", r"\_")


def pct(x):
    return f"{100.0 * float(x):.2f}"


# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="path to the dataset YAML")
    ap.add_argument("--weights", default=None, help="override model weights")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    args = ap.parse_args()

    from config.settings import YOLO_MODEL_PATH, YOLO_IMGSZ
    from ultralytics import YOLO

    weights = args.weights or YOLO_MODEL_PATH
    data_yaml = find_data_yaml(args.data)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  HELD-OUT TEST SPLIT EVALUATION")
    print("=" * 70)
    print(f"  weights : {weights}")
    print(f"  data    : {data_yaml}")
    print(f"  split   : {args.split}")
    print(f"  output  : {OUT_DIR}")
    print()
    print("  Running. This takes a few minutes on the RTX 4060.")
    print()

    model = YOLO(str(weights))
    results = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=YOLO_IMGSZ,
        plots=True,
        project=str(OUT_DIR),
        name="run",
        exist_ok=True,
        verbose=True,
    )

    box = results.box
    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
    }

    names = getattr(results, "names", None) or getattr(model, "names", {})
    per_class = []
    try:
        for i, class_idx in enumerate(box.ap_class_index):
            p, r, ap50, ap = box.class_result(i)
            per_class.append({
                "class": str(names.get(int(class_idx), int(class_idx))),
                "precision": float(p),
                "recall": float(r),
                "mAP50": float(ap50),
                "mAP50_95": float(ap),
            })
    except Exception as exc:                                # noqa: BLE001
        print(f"  [warn] per-class breakdown unavailable: {exc}")

    # ---------------- raw numbers ----------------
    payload = {
        "split": args.split,
        "weights": str(weights),
        "data_yaml": str(data_yaml),
        "imgsz": YOLO_IMGSZ,
        "overall": overall,
        "per_class": per_class,
        "thesis_validation_reference": THESIS_VALIDATION,
    }
    (OUT_DIR / "test_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    # ---------------- Table: headline metrics ----------------
    rows = [("Precision", overall["precision"]),
            ("Recall", overall["recall"]),
            ("mAP@0.50", overall["mAP50"]),
            ("mAP@0.50:0.95", overall["mAP50_95"])]

    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Detector results on the held-out synthetic test split "
        r"(560 images). These images were not used for training or for "
        r"selecting the training configuration.}",
        r"\label{tab:test-split-results}",
        r"\begin{tabular}{|l|r|}",
        r"\hline",
        r"\textbf{Metric} & \textbf{Test result} \\",
        r"\hline",
    ]
    for label, value in rows:
        tex.append(f"{label} & {pct(value)}\\,\\% \\\\")
        tex.append(r"\hline")
    tex += [r"\end{tabular}", r"\end{table}", ""]
    (OUT_DIR / "test_metrics.tex").write_text("\n".join(tex), encoding="utf-8")

    # ---------------- Table: per class ----------------
    if per_class:
        tex = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Per-class detector results on the held-out synthetic "
            r"test split. Per-class values are reported because an overall "
            r"average can hide differences between individual components.}",
            r"\label{tab:test-split-per-class}",
            r"\begin{tabular}{|l|r|r|r|r|}",
            r"\hline",
            r"\textbf{Class} & \textbf{Precision} & \textbf{Recall} & "
            r"\textbf{mAP@0.50} & \textbf{mAP@0.50:0.95} \\",
            r"\hline",
        ]
        for c in per_class:
            tex.append(
                f"\\texttt{{{tex_escape(c['class'])}}} & {pct(c['precision'])} & "
                f"{pct(c['recall'])} & {pct(c['mAP50'])} & "
                f"{pct(c['mAP50_95'])} \\\\")
            tex.append(r"\hline")
        tex += [r"\end{tabular}", r"\end{table}", ""]
        (OUT_DIR / "test_per_class.tex").write_text("\n".join(tex),
                                                    encoding="utf-8")

    # ---------------- Table: validation vs test ----------------
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparison of the synthetic validation split with the "
        r"independent synthetic test split. A difference between the two "
        r"describes generalisation within the synthetic data distribution; "
        r"neither value establishes performance on real camera images.}",
        r"\label{tab:val-vs-test}",
        r"\begin{tabular}{|l|r|r|r|}",
        r"\hline",
        r"\textbf{Metric} & \textbf{Validation} & \textbf{Test} & "
        r"\textbf{Difference} \\",
        r"\hline",
    ]
    for label, key in (("Precision", "precision"), ("Recall", "recall"),
                       ("mAP@0.50", "mAP50"), ("mAP@0.50:0.95", "mAP50_95")):
        v = THESIS_VALIDATION[key]
        t = overall[key]
        tex.append(f"{label} & {pct(v)}\\,\\% & {pct(t)}\\,\\% & "
                   f"{100.0 * (t - v):+.2f}\\,pp \\\\")
        tex.append(r"\hline")
    tex += [r"\end{tabular}", r"\end{table}", ""]
    (OUT_DIR / "val_vs_test.tex").write_text("\n".join(tex), encoding="utf-8")

    # ---------------- Figures ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        bars = ax.bar(labels, vals, color="#8f93e0", edgecolor="#4a4d99")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", fontsize=10, color="#2a2d6b")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Metric value")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "test_metrics.png", dpi=200)
        plt.close(fig)

        # validation vs test, grouped
        import numpy as np
        fig, ax = plt.subplots(figsize=(8, 4))
        keys = ["precision", "recall", "mAP50", "mAP50_95"]
        x = np.arange(len(keys))
        w = 0.38
        ax.bar(x - w / 2, [THESIS_VALIDATION[k] for k in keys], w,
               label="Validation split", color="#8f93e0", edgecolor="#4a4d99")
        ax.bar(x + w / 2, [overall[k] for k in keys], w,
               label="Test split (held out)", color="#e8a0a0",
               edgecolor="#9c4a4a")
        ax.set_xticks(x)
        ax.set_xticklabels(["Precision", "Recall", "mAP@0.50",
                            "mAP@0.50:0.95"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Metric value")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "val_vs_test.png", dpi=200)
        plt.close(fig)
    except Exception as exc:                                # noqa: BLE001
        print(f"  [warn] could not draw figures: {exc}")

    # Ultralytics writes its own confusion matrix / PR curves into run/
    run_dir = OUT_DIR / "run"
    for src_name, dst_name in (
        ("confusion_matrix_normalized.png", "confusion_matrix.png"),
        ("confusion_matrix.png", "confusion_matrix_counts.png"),
        ("BoxPR_curve.png", "pr_curve.png"),
        ("PR_curve.png", "pr_curve.png"),
    ):
        src = run_dir / src_name
        if src.is_file():
            shutil.copy2(src, OUT_DIR / dst_name)

    # ---------------- Plain-language report ----------------
    lines = [
        "HELD-OUT TEST SPLIT EVALUATION",
        "=" * 70,
        "",
        f"  weights   {weights}",
        f"  data      {data_yaml}",
        f"  split     {args.split}",
        "",
        "RESULT",
        "-" * 70,
        f"  Precision        {pct(overall['precision'])} %",
        f"  Recall           {pct(overall['recall'])} %",
        f"  mAP@0.50         {pct(overall['mAP50'])} %",
        f"  mAP@0.50:0.95    {pct(overall['mAP50_95'])} %",
        "",
        "COMPARED WITH THE VALIDATION SPLIT ALREADY IN THE THESIS",
        "-" * 70,
    ]
    for label, key in (("Precision", "precision"), ("Recall", "recall"),
                       ("mAP@0.50", "mAP50"), ("mAP@0.50:0.95", "mAP50_95")):
        lines.append(f"  {label:<16} val {pct(THESIS_VALIDATION[key])} %   "
                     f"test {pct(overall[key])} %   "
                     f"({100.0 * (overall[key] - THESIS_VALIDATION[key]):+.2f} pp)")

    if per_class:
        lines += ["", "PER CLASS", "-" * 70,
                  f"  {'class':<14}{'P':>8}{'R':>8}{'mAP50':>9}{'mAP50-95':>10}"]
        for c in per_class:
            lines.append(f"  {c['class']:<14}{pct(c['precision']):>8}"
                         f"{pct(c['recall']):>8}{pct(c['mAP50']):>9}"
                         f"{pct(c['mAP50_95']):>10}")
        worst = min(per_class, key=lambda c: c["mAP50_95"])
        best = max(per_class, key=lambda c: c["mAP50_95"])
        lines += ["",
                  f"  Best localised class:  {best['class']} "
                  f"(mAP@0.50:0.95 {pct(best['mAP50_95'])} %)",
                  f"  Weakest class:         {worst['class']} "
                  f"(mAP@0.50:0.95 {pct(worst['mAP50_95'])} %)"]

    lines += [
        "",
        "WHAT THIS RESULT IS",
        "-" * 70,
        "  An INDEPENDENT measurement of the detector within the synthetic",
        "  data distribution. These 560 images were not used for training",
        "  and not used to choose the training configuration, so this is a",
        "  genuine held-out result and not a second reading of the",
        "  validation split.",
        "",
        "WHAT THIS RESULT IS NOT",
        "-" * 70,
        "  It is NOT evidence of performance on the physical workstation.",
        "  Both splits are rendered images. Surface finish, illumination,",
        "  sensor response, motion blur and hand occlusion differ on the",
        "  real camera. This is the same argument already made in Section",
        "  6.5 of the thesis, and this result does not change it.",
        "",
        "WHERE IT GOES IN THE THESIS",
        "-" * 70,
        "  Section 6.2.2 (Object-detection results), after Table 6.3.",
        "  Update Table 6.10, row 'Independent detector test', from",
        "    'Prepared; test result not reported'",
        "  to",
        "    'Evaluated on the held-out 560-image synthetic test split'.",
        "",
        "FILES WRITTEN",
        "-" * 70,
    ]
    for f in sorted(OUT_DIR.glob("*")):
        if f.is_file():
            lines.append(f"  {f.name}")

    report = "\n".join(lines)
    (OUT_DIR / "REPORT.txt").write_text(report, encoding="utf-8")
    print()
    print(report)


if __name__ == "__main__":
    main()
