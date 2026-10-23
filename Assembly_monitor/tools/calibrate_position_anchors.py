"""
tools/calibrate_position_anchors.py
===============================================================================
Marks the two bushing seats in a reference camera view and writes
config/position_config.json plus clean/annotated reference images. The live
validator tracks body_a relative to that reference so seat markers can move
with the object.

WHY THIS EXISTS
  The CAD knows the two seats are 18.00 mm apart and exactly where they sit
  relative to body_a. The camera does not. One single-step calibration maps
  the two CAD seats onto image coordinates, after which a detected bushing
  can be assigned to the nearer seat with no training, no dataset and no GPU.

  This also makes the position check INTERPRETABLE, which matters for the
  thesis: every decision reduces to "the box centre was N pixels from
  Seat_1 and M pixels from Seat_2", and both numbers are printed.

HOW TO RUN  (Windows)
  1. Put body_a in a clear starting pose with both target seats visible, and
     keep the camera mounted where it will stay. Start the monitor in this
     view so the tracker can initialise, then move the part during assembly.
  2. From the Assembly_monitor folder, with the virtual environment active:

         python tools\\calibrate_position_anchors.py

  3. A live window opens. Press SPACE to freeze a clear frame.
  4. Click the centre of the FIRST seat (where the 1st bushing belongs).
  5. Click the centre of the SECOND seat.
  6. Press ENTER to save, r to redo the clicks, or q to abandon.

WHAT IT WRITES
  config/position_config.json - seat anchors in NORMALISED coordinates
  (0..1 of the frame), plus the per-stage requirements. It also writes
  position_config_reference_raw.png (tracking) and position_config_reference.png
  (annotated record).

WHEN TO RE-RUN IT
  Any time the camera is moved, re-aimed, re-zoomed, or the fixture
  position changes. The anchors are image coordinates and are only valid
  while the assembly sits in a fixed orientation and the camera is still -
  that is the stated scope of this first implementation.

  The seat rings drawn on the live video by the monitor are there so a
  drifted calibration is visible immediately rather than appearing later
  as inexplicable warnings.

OFFLINE MODE
  If no camera is available, calibrate from a saved still instead:

      python tools\\calibrate_position_anchors.py --image shot.png

===============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import cv2

# Make the project root importable when run as tools\...
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# What we are asking the person to click, in order.
# ---------------------------------------------------------------------------

SEAT_PROMPTS = [
    ("Seat_1", "1st bushing seat  (the hole the FIRST bushing goes into)"),
    ("Seat_2", "2nd bushing seat  (the hole the SECOND bushing goes into)"),
]

# Which seats each stage requires. Stage_01 needs only the first; Stage_02
# needs both. Later stages add different parts, so they have no seat
# requirement and the validator reports NOT_APPLICABLE for them.
STAGE_REQUIREMENTS = {
    "Stage_01": [{"seat": "Seat_1", "class": "bushing", "ordinal": 1}],
    "Stage_02": [{"seat": "Seat_1", "class": "bushing", "ordinal": 1},
                 {"seat": "Seat_2", "class": "bushing", "ordinal": 2}],
}

WINDOW = "Seat calibration  -  SPACE freeze | click seats | ENTER save | r redo | q quit"


# ---------------------------------------------------------------------------

def default_output_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "config", "position_config.json")


def grab_frame_from_camera():
    """Open the project's configured camera and let the user freeze a frame."""
    try:
        from acquisition.camera_factory import create_camera
        camera = create_camera()
        opened = camera.open()
    except Exception as exc:                                    # noqa: BLE001
        print(f"[calibrate] Project camera unavailable ({exc}); "
              "falling back to cv2.VideoCapture(0).")
        camera, opened = None, False

    if camera is not None and opened:
        print("[calibrate] Camera open. Press SPACE to freeze a clear frame, "
              "q to abandon.")
        frozen = None
        try:
            while True:
                ret, frame = camera.read_frame()
                if not ret:
                    print("[calibrate] Camera read failed.")
                    break
                shown = frame.copy()
                cv2.putText(shown, "SPACE = freeze this frame", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow(WINDOW, shown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    frozen = frame.copy()
                    break
                if key == ord('q'):
                    break
        finally:
            camera.release()
        return frozen

    # Fallback: raw OpenCV capture.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[calibrate] No camera could be opened. Use --image instead.")
        return None
    print("[calibrate] Press SPACE to freeze a clear frame, q to abandon.")
    frozen = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        shown = frame.copy()
        cv2.putText(shown, "SPACE = freeze this frame", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(WINDOW, shown)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            frozen = frame.copy()
            break
        if key == ord('q'):
            break
    cap.release()
    return frozen


# ---------------------------------------------------------------------------

def click_seats(frame):
    """Let the user click each seat in turn. Returns [(name, (x, y)), ...]."""
    picks: list = []
    h, w = frame.shape[:2]

    def redraw():
        canvas = frame.copy()
        # Instruction strip
        cv2.rectangle(canvas, (0, 0), (w, 64), (30, 30, 30), -1)
        if len(picks) < len(SEAT_PROMPTS):
            name, prompt = SEAT_PROMPTS[len(picks)]
            cv2.putText(canvas, f"CLICK: {prompt}", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 2)
            cv2.putText(canvas, f"{len(picks)} of {len(SEAT_PROMPTS)} marked"
                                "    r = redo    q = quit", (12, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)
        else:
            cv2.putText(canvas, "Both seats marked.  ENTER = save   "
                                "r = redo   q = quit", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 0), 2)
            dx = picks[1][1][0] - picks[0][1][0]
            dy = picks[1][1][1] - picks[0][1][1]
            sep = (dx * dx + dy * dy) ** 0.5
            cv2.putText(canvas, f"seat separation in the image: {sep:.0f} px",
                        (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        (200, 200, 200), 1)

        for i, (name, (x, y)) in enumerate(picks):
            colour = (0, 220, 0) if i == 0 else (0, 180, 255)
            cv2.circle(canvas, (x, y), 16, colour, 2)
            cv2.drawMarker(canvas, (x, y), colour, cv2.MARKER_CROSS, 12, 1)
            cv2.putText(canvas, name, (x + 20, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        if len(picks) == 2:
            cv2.line(canvas, picks[0][1], picks[1][1], (120, 120, 120), 1)
        cv2.imshow(WINDOW, canvas)

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(picks) < len(SEAT_PROMPTS):
            picks.append((SEAT_PROMPTS[len(picks)][0], (x, y)))
            print(f"[calibrate] {picks[-1][0]} marked at ({x}, {y})")
            redraw()

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            return None
        if key == ord('r'):
            picks.clear()
            print("[calibrate] cleared - click the seats again.")
            redraw()
        if key in (13, 10) and len(picks) == len(SEAT_PROMPTS):
            return picks


# ---------------------------------------------------------------------------

def build_config(frame, picks, args) -> dict:
    h, w = frame.shape[:2]
    seats = {}
    for i, (name, (x, y)) in enumerate(picks):
        _, prompt = SEAT_PROMPTS[i]
        seats[name] = {
            "anchor": [round(x / w, 5), round(y / h, 5)],
            "anchor_px_at_calibration": [int(x), int(y)],
            "class": "bushing",
            "label": "1st hole" if i == 0 else "2nd hole",
            "description": prompt,
        }

    dx = picks[1][1][0] - picks[0][1][0]
    dy = picks[1][1][1] - picks[0][1][1]
    sep_px = (dx * dx + dy * dy) ** 0.5

    # A detection must be nearer than this to count as "in" a seat.
    # Keep the acceptance radius strictly below half the measured seat
    # separation. A fixed 18 px minimum is unsafe for close seats: in the
    # supplied calibration the centres are only 30.1 px apart, so an 18 px
    # radius would make the two acceptance regions overlap.
    if sep_px < 8.0:
        raise ValueError(
            f"Seat centres are only {sep_px:.1f} px apart. "
            "The camera view is too low-resolution/ambiguous to calibrate "
            "these seats reliably; increase resolution or adjust the view."
        )
    max_dist = round(sep_px * 0.40, 1)

    return {
        "assembly": args.assembly,
        "camera_elevation_deg": args.elevation,
        "frame_size": [int(w), int(h)],
        "backend": args.backend,

        "seats": seats,
        "stage_requirements": STAGE_REQUIREMENTS,

        "max_anchor_distance_px": max_dist,
        "seat_memory_sec": 3.0,
        "min_confidence": 0.45,
        "stable_frames_wrong": 8,
        "stable_frames_correct": 8,

        "classifier": {
            "model_path": args.classifier or "",
            "input_size": 96,
            "crop_margin_px": round(max(40.0, sep_px * 0.9), 1),
            "min_confidence": 0.75,
            "correct_label": "correct",
            "wrong_label": "wrong_position",
        },

        "provenance": {
            "calibrated": datetime.now().isoformat(timespec="seconds"),
            "tool": "tools/calibrate_position_anchors.py",
            "seat_separation_px": round(sep_px, 1),
            "cad_seat_separation_mm": 18.0,
            "mm_per_px_estimate": (round(18.0 / sep_px, 4)
                                   if sep_px > 0 else None),
            "feasibility_experiment": "POS-FEAS-01",
            "feasibility_note": (
                "POS-FEAS-01 is a synthetic CAD observability study "
                "(3.38 % seat-region change at 35 deg elevation). It is not "
                "a measurement of real-camera accuracy. No accuracy figure "
                "should be quoted for this feature until a labelled "
                "real-camera test set has been measured."
            ),
            "scope_note": (
                "Seat anchors are image coordinates and are valid only while "
                "the assembly orientation on the fixture and the camera pose "
                "are unchanged. Re-run this tool after any camera or fixture "
                "change. Pose normalisation is a documented extension."
            ),
        },
    }


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Mark the bushing seats in the camera image and write "
                    "position_config.json.")
    ap.add_argument("--image", help="calibrate from a saved still instead of "
                                    "the live camera")
    ap.add_argument("--out", default=None,
                    help="output path (default config/position_config.json)")
    ap.add_argument("--assembly", default="subassembly")
    ap.add_argument("--elevation", type=float, default=35.0,
                    help="camera elevation in degrees, recorded for the "
                         "thesis record (default 35, from POS-FEAS-01)")
    ap.add_argument("--backend", default="geometric",
                    choices=["geometric", "classifier", "both"],
                    help="geometric needs no training and is the default; "
                         "use 'both' for recorded validation runs once the "
                         "classifier exists")
    ap.add_argument("--classifier", default="",
                    help="path to the trained position model, if you have "
                         "one already")
    args = ap.parse_args()

    out_path = args.out or default_output_path()

    # ---- get a frame ----
    if args.image:
        if not os.path.exists(args.image):
            print(f"[calibrate] No such image: {args.image}")
            return 1
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"[calibrate] Could not read {args.image}")
            return 1
        print(f"[calibrate] Using still: {args.image}")
    else:
        frame = grab_frame_from_camera()

    if frame is None:
        print("[calibrate] Nothing to calibrate from. Abandoned.")
        cv2.destroyAllWindows()
        return 1

    # ---- click the seats ----
    picks = click_seats(frame)
    cv2.destroyAllWindows()
    if not picks:
        print("[calibrate] Abandoned - nothing written.")
        return 1

    config = build_config(frame, picks, args)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # Keep BOTH a clean reference for feature tracking and an annotated copy
    # for human inspection. The tracking reference must not contain drawn labels.
    ref_png = os.path.splitext(out_path)[0] + "_reference.png"
    ref_raw_png = os.path.splitext(out_path)[0] + "_reference_raw.png"
    cv2.imwrite(ref_raw_png, frame)
    config["pose_tracking"] = {
        "enabled": True,
        "reference_image": os.path.basename(ref_raw_png),
        "min_inliers": 8,
        "min_inlier_ratio": 0.22,
        "max_reprojection_error_px": 6.0,
        "feature_count": 900,
    }
    config["provenance"]["reference_image_raw"] = os.path.basename(ref_raw_png)
    config["provenance"]["reference_image_annotated"] = os.path.basename(ref_png)
    config["provenance"]["pose_tracking"] = (
        "Body-relative feature tracking with homography. Supports gradual image "
        "motion and moderate viewpoint changes; it is not general 6-DoF pose "
        "estimation. If tracking confidence is inadequate, seat markers are hidden "
        "and Stage_01/02 confirmation is blocked."
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    marked = frame.copy()
    for i, (name, (x, y)) in enumerate(picks):
        colour = (0, 220, 0) if i == 0 else (0, 180, 255)
        cv2.circle(marked, (x, y), 16, colour, 2)
        cv2.putText(marked, name, (x + 20, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    cv2.imwrite(ref_png, marked)

    sep = config["provenance"]["seat_separation_px"]
    mmpp = config["provenance"]["mm_per_px_estimate"]
    print("\n" + "=" * 66)
    print("  CALIBRATION SAVED")
    print("=" * 66)
    print(f"  config      {out_path}")
    print(f"  reference   {ref_png}")
    print(f"  raw ref     {ref_raw_png}")
    print(f"  pose track  body-relative feature tracking (fail-safe)")
    print(f"  frame       {config['frame_size'][0]} x {config['frame_size'][1]}")
    for name, spec in config["seats"].items():
        print(f"  {name:8s}    image {tuple(spec['anchor_px_at_calibration'])}"
              f"   normalised ({spec['anchor'][0]:.3f}, {spec['anchor'][1]:.3f})")
    print(f"  separation  {sep} px  =  18.00 mm in CAD"
          + (f"   ({mmpp} mm/px)" if mmpp else ""))
    print(f"  seat gate   {config['max_anchor_distance_px']} px")
    print(f"  backend     {config['backend']}")
    print()
    print("  Next: set POSITION_CONFIG_PATH in config/settings.py to the")
    print("  path above, then run  python main.py  and check that the seat")
    print("  rings on the video sit on the real holes.")
    print("=" * 66 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
