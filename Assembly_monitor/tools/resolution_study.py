"""
resolution_study.py
========================================================================
ONE-FACTOR EXPERIMENT: INFERENCE RESOLUTION vs DETECTION PERFORMANCE.

WHY THIS EXISTS
---------------
On the bench, a bushing seated in body_a measures roughly 25-30 px in the
1280x720 camera frame. The detector runs at an input size of 640, so the
frame is halved before inference and that bushing becomes about 15 px.
YOLO's finest detection stride is 8 px, so a 15 px object spans roughly
two grid cells, which is the edge of what the network can resolve. Moving
the part towards the camera restores the detection, which is consistent
with a size limit rather than a model limit: on the synthetic test split
the same class reaches 87.85 % recall.

This script treats inference resolution as a single experimental factor
and measures its effect on detection performance per class. It is a
one-factor study, not a full factorial design, and the output says so.

WHAT IT PRODUCES
----------------
  thesis_results\\resolution_study\\
      resolution_study.tex        Table: overall metrics at each level
      resolution_per_class.tex    Table: per-class mAP@0.50:0.95 by level
      resolution_study.png        Figure: metric vs input size
      resolution_small_parts.png  Figure: the small classes specifically
      resolution_results.json     Raw numbers
      REPORT.txt                  Plain-language summary

HOW TO RUN
----------
  In PowerShell, from the Assembly_monitor folder:

    & "D:\\Thesis\\SOFTWARE V2.1 AI Final\\venv\\Scripts\\python.exe" tools\\resolution_study.py

  Takes about one minute. Default levels are 640, 960 and 1280.
  To change them:

    ... tools\\resolution_study.py --sizes 640 1280

WHAT THE RESULT MEANS, AND WHAT IT DOES NOT
-------------------------------------------
The measurement is made on SYNTHETIC test images. Those renders show the
components at a generous apparent size, so the headroom visible here is
smaller than the improvement expected on the live camera, where the parts
are much smaller in frame. The study establishes whether raising the
inference resolution costs accuracy and what it costs in runtime; the
live benefit has to be confirmed separately at the workstation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "thesis_results" / "resolution_study"

# Parts whose apparent size on the bench is small enough that inference
# resolution is expected to matter. Used only for the focused figure.
SMALL_PARTS = ("bushing", "nut", "threaded_pin", "block_pin")


def pct(x):
    return f"{100.0 * float(x):.2f}"


def esc(s):
    return str(s).replace("_", r"\_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[640, 960, 1280])
    ap.add_argument("--data", default=None)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    from config.settings import YOLO_MODEL_PATH
    from ultralytics import YOLO

    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    from evaluate_test_split import find_data_yaml

    data_yaml = find_data_yaml(args.data)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  ONE-FACTOR STUDY: INFERENCE RESOLUTION")
    print("=" * 70)
    print(f"  weights : {YOLO_MODEL_PATH}")
    print(f"  data    : {data_yaml}")
    print(f"  split   : {args.split}")
    print(f"  levels  : {', '.join(str(s) for s in args.sizes)} px")
    print()

    model = YOLO(str(YOLO_MODEL_PATH))
    runs = []

    for size in args.sizes:
        print(f"  --- inference at {size} px ---")
        t0 = time.time()
        res = model.val(data=str(data_yaml), split=args.split, imgsz=size,
                        plots=False, project=str(OUT_DIR),
                        name=f"imgsz{size}", exist_ok=True, verbose=False)
        wall = time.time() - t0
        box = res.box
        names = getattr(res, "names", None) or getattr(model, "names", {})

        per_class = {}
        try:
            for i, ci in enumerate(box.ap_class_index):
                p, r, ap50, ap = box.class_result(i)
                per_class[str(names.get(int(ci), int(ci)))] = {
                    "precision": float(p), "recall": float(r),
                    "mAP50": float(ap50), "mAP50_95": float(ap)}
        except Exception as exc:                            # noqa: BLE001
            print(f"      [warn] per-class unavailable: {exc}")

        speed = getattr(res, "speed", {}) or {}
        runs.append({
            "imgsz": size,
            "precision": float(box.mp), "recall": float(box.mr),
            "mAP50": float(box.map50), "mAP50_95": float(box.map),
            "inference_ms": float(speed.get("inference", 0.0)),
            "wall_s": wall,
            "per_class": per_class,
        })
        print(f"      P {pct(box.mp)}  R {pct(box.mr)}  "
              f"mAP50 {pct(box.map50)}  mAP50-95 {pct(box.map)}  "
              f"| {speed.get('inference', 0.0):.1f} ms/img")

    (OUT_DIR / "resolution_results.json").write_text(
        json.dumps({"data_yaml": str(data_yaml), "split": args.split,
                    "runs": runs}, indent=2), encoding="utf-8")

    base = runs[0]

    # ---------------- overall table ----------------
    tex = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Effect of inference input size on detector performance, "
        r"measured on the synthetic test split with the deployed model "
        r"weights. Inference resolution is varied as a single experimental "
        r"factor; all other conditions are held constant.}",
        r"\label{tab:resolution-study}",
        r"\begin{tabular}{|r|r|r|r|r|r|}", r"\hline",
        r"\textbf{Input size} & \textbf{Precision} & \textbf{Recall} & "
        r"\textbf{mAP@0.50} & \textbf{mAP@0.50:0.95} & "
        r"\textbf{Inference} \\", r"\hline",
    ]
    for r in runs:
        tex.append(f"{r['imgsz']}\\,px & {pct(r['precision'])}\\,\\% & "
                   f"{pct(r['recall'])}\\,\\% & {pct(r['mAP50'])}\\,\\% & "
                   f"{pct(r['mAP50_95'])}\\,\\% & "
                   f"{r['inference_ms']:.1f}\\,ms \\\\")
        tex.append(r"\hline")
    tex += [r"\end{tabular}", r"\end{table}", ""]
    (OUT_DIR / "resolution_study.tex").write_text("\n".join(tex),
                                                  encoding="utf-8")

    # ---------------- per-class table ----------------
    classes = sorted(base["per_class"].keys())
    if classes:
        tex = [
            r"\begin{table}[htbp]", r"\centering",
            r"\caption{Per-class mAP@0.50:0.95 at each inference input "
            r"size. The components that appear small in the camera frame "
            r"are the ones for which input size is expected to matter.}",
            r"\label{tab:resolution-per-class}",
            r"\begin{tabular}{|l|" + "r|" * len(runs) + "}", r"\hline",
            r"\textbf{Class} & " +
            " & ".join(rf"\textbf{{{r['imgsz']}\,px}}" for r in runs) +
            r" \\", r"\hline",
        ]
        for c in classes:
            cells = []
            for r in runs:
                v = r["per_class"].get(c)
                cells.append("---" if v is None else f"{pct(v['mAP50_95'])}")
            tex.append(f"\\texttt{{{esc(c)}}} & " + " & ".join(cells) + r" \\")
            tex.append(r"\hline")
        tex += [r"\end{tabular}", r"\end{table}", ""]
        (OUT_DIR / "resolution_per_class.tex").write_text(
            "\n".join(tex), encoding="utf-8")

    # ---------------- figures ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [r["imgsz"] for r in runs]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for key, label, colour in (
            ("mAP50", "mAP@0.50", "#2a2d6b"),
            ("mAP50_95", "mAP@0.50:0.95", "#9c4a4a"),
            ("recall", "Recall", "#3f7f3f"),
        ):
            ax.plot(xs, [r[key] for r in runs], "o-", label=label,
                    color=colour)
        ax.set_xlabel("Inference input size (px)")
        ax.set_ylabel("Metric value")
        ax.set_xticks(xs)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "resolution_study.png", dpi=200)
        plt.close(fig)

        small = [c for c in classes if c in SMALL_PARTS]
        if small:
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            for c in small:
                ax.plot(xs, [r["per_class"].get(c, {}).get("mAP50_95", 0)
                             for r in runs], "o-", label=c)
            ax.set_xlabel("Inference input size (px)")
            ax.set_ylabel("mAP@0.50:0.95")
            ax.set_xticks(xs)
            ax.grid(alpha=0.3)
            ax.set_axisbelow(True)
            ax.legend(title="small components")
            fig.tight_layout()
            fig.savefig(OUT_DIR / "resolution_small_parts.png", dpi=200)
            plt.close(fig)
    except Exception as exc:                                # noqa: BLE001
        print(f"  [warn] could not draw figures: {exc}")

    # ---------------- report ----------------
    best = max(runs, key=lambda r: r["mAP50_95"])
    L = ["ONE-FACTOR STUDY: INFERENCE RESOLUTION", "=" * 70, "",
         f"  split   {args.split}",
         f"  levels  {', '.join(str(s) for s in args.sizes)} px", "",
         "OVERALL", "-" * 70,
         f"  {'size':>8}{'P':>9}{'R':>9}{'mAP50':>9}{'mAP50-95':>10}"
         f"{'ms/img':>9}"]
    for r in runs:
        L.append(f"  {r['imgsz']:>8}{pct(r['precision']):>9}"
                 f"{pct(r['recall']):>9}{pct(r['mAP50']):>9}"
                 f"{pct(r['mAP50_95']):>10}{r['inference_ms']:>9.1f}")

    L += ["", "CHANGE RELATIVE TO THE 640 px BASELINE", "-" * 70]
    for r in runs[1:]:
        L.append(f"  {r['imgsz']:>5} px   "
                 f"mAP50 {100*(r['mAP50']-base['mAP50']):+6.2f} pp   "
                 f"mAP50-95 {100*(r['mAP50_95']-base['mAP50_95']):+6.2f} pp   "
                 f"recall {100*(r['recall']-base['recall']):+6.2f} pp   "
                 f"cost {r['inference_ms']-base['inference_ms']:+.1f} ms")

    if classes:
        L += ["", "SMALL COMPONENTS, mAP@0.50:0.95", "-" * 70,
              "  " + "class".ljust(14) +
              "".join(f"{r['imgsz']:>10}" for r in runs)]
        for c in classes:
            if c not in SMALL_PARTS:
                continue
            L.append("  " + c.ljust(14) +
                     "".join(f"{pct(r['per_class'].get(c, {}).get('mAP50_95', 0)):>10}"
                             for r in runs))

    L += ["", "READING THIS RESULT", "-" * 70,
          f"  Best overall mAP@0.50:0.95 at {best['imgsz']} px "
          f"({pct(best['mAP50_95'])} %).",
          "",
          "  These images are RENDERS, and the renders show the components",
          "  at a generous apparent size. The live camera does not: a seated",
          "  bushing is roughly 25-30 px in the 1280x720 frame, which at an",
          "  input size of 640 is halved to about 15 px. The improvement",
          "  available on the bench is therefore larger than the improvement",
          "  visible here, and it has to be confirmed at the workstation",
          "  rather than inferred from this table.",
          "",
          "  What this table DOES establish is the cost side: whether raising",
          "  the input size reduces accuracy, and how much inference time it",
          "  adds. Both are needed before changing the deployed setting.",
          "",
          "NEXT STEP", "-" * 70,
          f"  Set YOLO_IMGSZ = {best['imgsz']} in config\\settings.py, run",
          "  main.py, and check whether a seated bushing is detected at",
          "  normal working distance. That live check is the result that",
          "  matters; this table is the evidence that the change is safe.",
          "", "FILES WRITTEN", "-" * 70]
    for f in sorted(OUT_DIR.glob("*")):
        if f.is_file():
            L.append(f"  {f.name}")

    rep = "\n".join(L)
    (OUT_DIR / "REPORT.txt").write_text(rep, encoding="utf-8")
    print()
    print(rep)


if __name__ == "__main__":
    main()
