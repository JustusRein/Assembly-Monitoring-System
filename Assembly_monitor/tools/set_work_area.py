"""
set_work_area.py
========================================================================
DEFINE THE WORK AREA. Writes calibration/roi_config.json.

Your last run started with this:

    [main] WARNING: ROI config not found at 'calibration/roi_config.json'.
      -> Using full frame as ROI for now.

"Full frame" meant the detector was looking at your laptop keyboard, the
room behind the bench, and several sheets of crumpled paper. It was never
trained on any of that, so it guessed - and a guess from YOLO is reported
with a confidence number that looks just as respectable as a real one.
That is where `nut 54%` on your mobile phone came from, and `knob 83%` on
a blank sheet of paper.

Drawing a box around the fixture alone is the single highest-value thing
you can do to this system today. It cuts the detector's input down to the
surface the model was actually trained on.

HOW TO RUN
----------
  (from the Assembly_monitor folder, in the project venv)

    python tools\\set_work_area.py

  Then:
    left click      add a corner
    right click     undo the last corner
    [c]             clear and start again
    [s]             SAVE (needs at least 3 corners)
    [q]             quit without saving

  Draw around the FIXTURE AND THE AREA YOU ASSEMBLE IN - not the whole
  table. Keep the keyboard, your laptop, the room behind, and any loose
  paperwork OUTSIDE the shape. Leave a small margin so the part is never
  half outside the boundary.
"""

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acquisition.camera_factory import create_camera          # noqa: E402
from config.settings import ROI_CONFIG_PATH                   # noqa: E402

WINDOW = "Define the work area"

points = []


def on_mouse(event, x, y, flags, _param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([int(x), int(y)])
        print(f"  corner {len(points)}: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN and points:
        gone = points.pop()
        print(f"  removed corner at ({gone[0]}, {gone[1]})")


def draw_overlay(frame):
    view = frame.copy()

    if len(points) >= 3:
        # Dim everything outside the shape, so what the detector will be
        # given is obvious at a glance.
        mask = np.zeros(view.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
        dark = (view * 0.25).astype(np.uint8)
        view = np.where(mask[:, :, None] > 0, view, dark)
        cv2.polylines(view, [np.asarray(points, dtype=np.int32)], True,
                      (0, 220, 0), 2)
    elif len(points) >= 2:
        cv2.polylines(view, [np.asarray(points, dtype=np.int32)], False,
                      (0, 220, 255), 2)

    for i, (px, py) in enumerate(points):
        cv2.circle(view, (px, py), 6, (0, 220, 255), -1)
        cv2.putText(view, str(i + 1), (px + 9, py - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2,
                    cv2.LINE_AA)

    bar = np.zeros((96, view.shape[1], 3), dtype=np.uint8)
    bar[:, :] = (28, 28, 28)
    cv2.putText(bar, f"{len(points)} corners placed", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(bar, "left click = add    right click = undo    "
                     "[c] clear    [s] save    [q] quit",
                (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1,
                cv2.LINE_AA)
    hint = ("Keep the keyboard, the room and loose paper OUTSIDE the shape."
            if len(points) < 3 else
            "Bright area = what the detector will see. Press [s] to save.")
    cv2.putText(bar, hint, (16, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (0, 220, 0) if len(points) >= 3 else (0, 215, 255), 1,
                cv2.LINE_AA)

    return np.vstack([view, bar])


def main():
    print(__doc__.split("HOW TO RUN")[0])

    camera = create_camera()
    if not camera.open():
        print("FATAL: could not open the camera.")
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    saved = False
    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                continue

            cv2.imshow(WINDOW, draw_overlay(frame))
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("c"):
                points.clear()
                print("  cleared")
            if key == ord("s"):
                if len(points) < 3:
                    print("  need at least 3 corners before saving")
                    continue

                path = ROI_CONFIG_PATH
                if not os.path.isabs(path):
                    path = os.path.join(
                        os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))), path)
                os.makedirs(os.path.dirname(path), exist_ok=True)

                h, w = frame.shape[:2]
                with open(path, "w") as f:
                    json.dump({
                        "points": points,
                        "frame_width": int(w),
                        "frame_height": int(h),
                    }, f, indent=2)

                area = cv2.contourArea(np.asarray(points, dtype=np.int32))
                print()
                print(f"  SAVED {len(points)} corners to {path}")
                print(f"  work area is {100.0 * area / (w * h):.1f} % of the "
                      f"frame")
                print()
                print("  Now run main.py. It will print 'ROI loaded from: ...'")
                saved = True
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if not saved:
        print("\n  Nothing was saved.")


if __name__ == "__main__":
    main()
