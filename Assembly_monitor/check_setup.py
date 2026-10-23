"""
===============================================================================
check_setup.py
===============================================================================
One command that checks every part of the system that can fail SILENTLY,
and says plainly which ones are broken.

    python check_setup.py

WHY THIS EXISTS
  Three features in this system disable themselves quietly when something
  is missing, and all three did:

    * the 360-degree stage preview vanishes if its render folder is not
      found - main.py catches the error and carries on without the panel
    * position validation does nothing if it is switched off in settings,
      or if the calibration file or reference image is missing
    * the body pose tracker fails safe and hides the seat markers if it
      cannot read the reference image or find features on body_a

  Failing safe is right. Failing safe WITHOUT SAYING SO is what cost days:
  the feature looks implemented, the screen looks normal, and nothing
  tells you which of the three silent paths you are on. This script makes
  each one say so.

It changes nothing. It only reads and reports.
===============================================================================
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OK = "  OK   "
BAD = " FAIL  "
WARN = " WARN  "

results = []


def say(level, label, detail=""):
    results.append((level, label))
    print(f"[{level}] {label}")
    for line in str(detail).splitlines():
        if line.strip():
            print(f"         {line}")


def header(text):
    print()
    print("=" * 74)
    print(f"  {text}")
    print("=" * 74)


# ───────────────────────────────────────────────────────────────────────────
header("1. SETTINGS")

try:
    from config import settings as S
    say(OK, "config/settings.py imported")
except Exception as exc:                                    # noqa: BLE001
    say(BAD, "config/settings.py could not be imported", exc)
    raise SystemExit(1)

cam = (getattr(S, "FRAME_WIDTH", None), getattr(S, "FRAME_HEIGHT", None))
say(OK, f"camera resolution requested: {cam[0]} x {cam[1]}")

pos_on = getattr(S, "POSITION_VALIDATION_ENABLED", False)
say(OK if pos_on else WARN,
    f"POSITION_VALIDATION_ENABLED = {pos_on}",
    "" if pos_on else "The whole position / wrong-hole feature is OFF. "
                      "No seat markers will be drawn and no WRONG POSITION "
                      "warning can ever fire.")

prev_on = getattr(S, "STAGE_PREVIEW_ENABLED", False)
say(OK if prev_on else WARN, f"STAGE_PREVIEW_ENABLED = {prev_on}")


# ───────────────────────────────────────────────────────────────────────────
header("2. 360-DEGREE STAGE PREVIEW")

prev_dir = Path(getattr(S, "STAGE_PREVIEW_DIR", ""))
diag_dir = Path(getattr(S, "STAGE_PREVIEW_DIAGRAM_DIR", ""))

if not prev_dir.exists():
    say(BAD, f"preview folder does not exist: {prev_dir}",
        "THIS IS WHY THE 360 VIEW DISAPPEARED FROM THE WINDOW.\n"
        "main.py catches the error and continues without the panel, so\n"
        "nothing on screen tells you. Either point STAGE_PREVIEW_DIR in\n"
        "config/settings.py at the folder that holds the renders, or\n"
        "re-run the Blender preview pass to recreate them.")
else:
    subs = [d for d in prev_dir.iterdir() if d.is_dir()]
    stage_dirs = sorted(d for d in subs if d.name.lower().startswith("stage"))
    if not stage_dirs:
        say(BAD, f"preview folder has no Stage_* subfolders: {prev_dir}",
            f"found instead: {[d.name for d in subs][:8] or 'nothing'}")
    else:
        total = 0
        thin = []
        for d in stage_dirs:
            n = len(list(d.glob("*.png"))) + len(list(d.glob("*.jpg")))
            total += n
            if n == 0:
                thin.append(d.name)
        say(OK if not thin else WARN,
            f"{len(stage_dirs)} stage folders, {total} preview frames",
            "" if not thin else f"empty: {', '.join(thin)}")

if diag_dir and diag_dir.exists():
    n = len(list(diag_dir.glob("*.png")))
    say(OK, f"2D diagrams folder found ({n} images)")
else:
    say(OK, "no 2D diagram folder - the rotating renders will be used",
        f"(looked in {diag_dir})" if str(diag_dir) else "")

try:
    from visualization.stage_preview import StagePreviewPlayer
    player = StagePreviewPlayer(
        str(prev_dir),
        seconds_per_revolution=getattr(S, "STAGE_PREVIEW_SECONDS_PER_TURN", 8.0),
        panel_size=getattr(S, "STAGE_PREVIEW_PANEL_SIZE", 320),
        diagram_dir=str(diag_dir),
        mode=getattr(S, "STAGE_PREVIEW_MODE", "auto"),
    )
    if player.available:
        f = player.frame("Stage_01")
        say(OK, "preview player works",
            f"Stage_01 frame: {None if f is None else f.shape}")
    else:
        say(BAD, "preview player reports itself unavailable",
            "This is the exact condition that makes main.py drop the panel.")
except Exception as exc:                                    # noqa: BLE001
    say(BAD, "preview player raised", exc)


# ───────────────────────────────────────────────────────────────────────────
header("3. POSITION CALIBRATION")

cfg_path = Path(getattr(S, "POSITION_CONFIG_PATH", ""))
local_cfg = ROOT / "config" / "position_config.json"

if not cfg_path.exists() and local_cfg.exists():
    # The path in settings.py is absolute, so it goes stale the moment the
    # project is copied or renamed - and the only symptom is the position
    # feature silently doing nothing. Say so rather than just failing.
    say(BAD, "POSITION_CONFIG_PATH points at a file that does not exist",
        f"settings.py says:\n  {cfg_path}\n"
        f"but there IS a config here:\n  {local_cfg}\n"
        f"Set POSITION_CONFIG_PATH in config/settings.py to the second "
        f"path.\nChecking the local one below so the rest of this report "
        f"is still useful.")
    cfg_path = local_cfg

if not cfg_path.exists():
    say(BAD, f"position_config.json not found: {cfg_path}",
        "Run: python tools\\calibrate_position_anchors.py")
    cfg = None
else:
    say(OK, f"config found: {cfg_path.name}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

if cfg:
    fs = cfg.get("frame_size", [0, 0])
    if list(fs) != [cam[0], cam[1]]:
        say(BAD, f"calibration resolution {fs} != camera {list(cam)}",
            "Anchors were marked at a different frame size. Markers will\n"
            "land in the wrong place. Re-calibrate at the camera's real\n"
            "resolution.")
    else:
        say(OK, f"calibration resolution matches the camera ({fs[0]}x{fs[1]})")

    seats = cfg.get("seats", {})
    pts = [s.get("anchor_px_at_calibration") for s in seats.values()]
    if len(pts) >= 2 and all(pts):
        sep = math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1])
        radius = float(cfg.get("max_anchor_distance_px", 0))
        say(OK, f"seat separation {sep:.1f} px, acceptance radius {radius:.1f} px")

        if radius >= sep / 2:
            say(BAD, "acceptance regions OVERLAP",
                f"The radius ({radius:.1f} px) is at least half the seat\n"
                f"separation ({sep / 2:.1f} px), so a bushing in EITHER hole\n"
                f"sits inside BOTH regions and wrong-hole detection cannot\n"
                f"work. Set max_anchor_distance_px to about "
                f"{round(sep * 0.4, 1)}.")
        else:
            say(OK, f"acceptance regions are separate "
                    f"(radius < half separation {sep / 2:.1f} px)")

        if sep < 40:
            say(WARN, f"the two seats are only {sep:.0f} px apart",
                "This is the biggest practical risk to the wrong-hole check.\n"
                "YOLO box centres jitter by a few pixels, and with so small a\n"
                "gap that jitter is a large fraction of the distance between\n"
                "the holes.\n"
                "MOVE THE CAMERA CLOSER (or zoom in) so body_a fills much\n"
                "more of the frame, then re-calibrate. Doubling the part's\n"
                "size in the image doubles this separation and halves the\n"
                "relative error, which costs nothing and helps more than any\n"
                "software change.")

    for name, s in seats.items():
        a = s.get("anchor", [0, 0])
        px = s.get("anchor_px_at_calibration", [0, 0])
        if fs and fs[0] and fs[1]:
            back = [round(a[0] * fs[0]), round(a[1] * fs[1])]
            drift = math.hypot(back[0] - px[0], back[1] - px[1])
            if drift > 2:
                say(BAD, f"{name}: normalised and pixel anchors disagree",
                    f"normalised maps to {back}, stored pixel is {px}")
    say(OK, "normalised and pixel anchors agree")


# ───────────────────────────────────────────────────────────────────────────
header("4. BODY POSE TRACKER  (makes the markers follow body_a)")

ref_candidates = []
if cfg_path.exists():
    d = cfg_path.parent
    ref_candidates = [d / "position_config_reference_raw.png",
                      d / "position_config_reference.png"]
ref = next((p for p in ref_candidates if p.exists()), None)

if ref is None:
    say(BAD, "no calibration reference image found",
        "Without it the tracker cannot start, the markers are hidden and\n"
        "the banner reads BODY NOT TRACKED. Re-run the calibration tool.")
else:
    say(OK, f"reference image: {ref.name}")
    try:
        import cv2
        im = cv2.imread(str(ref))
        if im is None:
            say(BAD, "reference image could not be decoded")
        else:
            say(OK if cfg and list(cfg.get("frame_size", [])) ==
                [im.shape[1], im.shape[0]] else BAD,
                f"reference image is {im.shape[1]}x{im.shape[0]}")

            from decision.body_pose_tracker import BodyPoseTracker
            anchors = {n: tuple(s["anchor_px_at_calibration"])
                       for n, s in (cfg or {}).get("seats", {}).items()}
            t = BodyPoseTracker(str(ref), anchors)
            if t._initialization_error:
                say(BAD, "tracker failed to initialise", t._initialization_error)
            else:
                nfeat = len(t._ref_keypoints or [])
                say(OK, f"tracker initialised, body ROI {t._ref_box}, "
                        f"{nfeat} reference features")
                if nfeat < 80:
                    say(WARN, f"only {nfeat} features on body_a",
                        "The part is smooth plastic with little texture, so\n"
                        "there is not much for the matcher to lock onto. It\n"
                        "works here, but tracking will be more reliable with\n"
                        "the part larger in frame and evenly lit.")

                class _D:
                    def __init__(s, b):
                        s.class_name = "body_a"
                        s.confidence = 0.9
                        s.bounding_box = b

                r = t.update(im, [_D(t._ref_box)])
                if not r.valid:
                    say(BAD, "tracker could not track its own reference image",
                        r.reason)
                else:
                    worst = max(
                        math.hypot(r.anchors_px[n][0] - anchors[n][0],
                                   r.anchors_px[n][1] - anchors[n][1])
                        for n in anchors)
                    say(OK if worst < 3 else BAD,
                        f"self-check: anchors recovered to {worst:.2f} px "
                        f"({r.inliers} inliers)")
    except Exception as exc:                                # noqa: BLE001
        say(BAD, "tracker check raised", exc)


# ───────────────────────────────────────────────────────────────────────────
header("5. WRONG-PART DETECTION")

try:
    from decision.assembly_state_machine import AssemblyStateMachine
    warned = []
    sm = AssemblyStateMachine(on_warning=lambda w: warned.append(w))
    for _ in range(60):
        sm.update(["body_a", "knob"])
    if warned:
        say(OK, "wrong part detected", warned[0].message)
    else:
        say(BAD, "no warning for a knob at Stage_01",
            "The WRONG_PART branch is unreachable. It compares the detected\n"
            "classes against the classes used by ALL stages, but every one\n"
            "of the seven classes is used somewhere, so that set is always\n"
            "empty and the branch can never fire.")
except Exception as exc:                                    # noqa: BLE001
    say(BAD, "state machine check raised", exc)


# ───────────────────────────────────────────────────────────────────────────
header("SUMMARY")

bad = [l for lv, l in results if lv == BAD]
warn = [l for lv, l in results if lv == WARN]
print(f"  {len(results) - len(bad) - len(warn)} ok   "
      f"{len(warn)} warnings   {len(bad)} failures")
if bad:
    print("\n  MUST FIX:")
    for b in bad:
        print(f"    - {b}")
if warn:
    print("\n  WORTH FIXING:")
    for w in warn:
        print(f"    - {w}")
if not bad:
    print("\n  Nothing is silently disabled. Run main.py.")
print()
raise SystemExit(1 if bad else 0)
