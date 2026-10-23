"""
check_seat_resolution.py
========================================================================
CAN THIS CAMERA SETUP TELL THE TWO BUSHING HOLES APART?

This is not a feature. It is the measurement that decides whether the
wrong-hole check is worth switching on at all, and it answers in about
twenty seconds.

WHY IT EXISTS
-------------
The wrong-hole check has to answer one question: of the two bores in
body_a, which one is the bushing sitting in? It can only answer that if
the two bores are far enough apart ON THE SENSOR for the difference to
survive detector jitter.

In the last measured setup they were 30 px apart. A YOLO box centre moves
by 8-12 px frame to frame on a part that is not moving at all, so the
decision boundary sat halfway between the two holes, 15 px from each -
inside the noise. The check was not "slightly unreliable"; it was a coin
toss, which is why it is currently switched off in settings.py.

The CAD study (POS-FEAS-01) put the true seat separation at 18.00 mm. So
this is purely a question of how many pixels that 18 mm lands on, which is
a question of how close the camera is and how much of the frame body_a
fills. Nothing about the software changes the answer.

WHAT THE NUMBERS MEAN
---------------------
  separation >= 60 px   the check is worth enabling
  40 - 60 px            marginal; it will be right most of the time and
                        occasionally confidently wrong, which for a
                        safety-style warning is worse than silence
  < 40 px               do not enable it. Move the camera closer.

HOW TO RUN
----------
  (from the Assembly_monitor folder, in the project venv)

    python tools\\check_seat_resolution.py

  Put body_a in the work area, empty, with both bores visible. Leave it
  still. Watch the number. Then move the camera closer, or zoom in, until
  the number stops going up - and read off the best you achieved.

  q = quit        s = save the current frame next to this script

WHAT IT DOES NOT DO
-------------------
It does not change any setting, enable anything, or write to any config.
It measures and reports. Switching the check on remains a deliberate
decision you make after seeing a number you are happy with.
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acquisition.camera_factory import create_camera          # noqa: E402
from processing.object_detector import ObjectDetector         # noqa: E402
from decision.seat_finder import SeatFinder                   # noqa: E402
from config.settings import (                                  # noqa: E402
    YOLO_MODEL_PATH, YOLO_CONFIDENCE, YOLO_ROLLING_WINDOW,
    YOLO_IMGSZ, YOLO_DEVICE,
)

WINDOW = "Seat resolution check"

GOOD_PX = 60.0
MARGINAL_PX = 40.0

# From the CAD feasibility study. Used only to report mm/px, which makes
# the result comparable with the study rather than being a bare pixel
# count with no reference.
CAD_SEAT_SEPARATION_MM = 18.00


def verdict(sep_px):
    if sep_px >= GOOD_PX:
        return "GOOD - the wrong-hole check is worth enabling", (0, 220, 0)
    if sep_px >= MARGINAL_PX:
        return "MARGINAL - move the camera closer", (0, 215, 255)
    return "TOO CLOSE TOGETHER - do not enable the check", (0, 0, 255)


def main():
    print(__doc__.split("HOW TO RUN")[0])
    print("Starting camera...")

    camera = create_camera()
    if not camera.open():
        print("FATAL: could not open the camera.")
        return

    detector = ObjectDetector(
        model_path=YOLO_MODEL_PATH,
        confidence=YOLO_CONFIDENCE,
        rolling_window=YOLO_ROLLING_WINDOW,
        imgsz=YOLO_IMGSZ,
        device=YOLO_DEVICE,
    )
    finder = SeatFinder(debug=False)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    samples = []          # separations from frames where both seats were found
    best = 0.0
    frames = 0
    found_frames = 0
    last_reason = ""
    started = time.time()

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                continue
            frames += 1

            result = detector.process(frame)
            objects = getattr(result, "objects", result)
            seats = finder.update(frame, objects)

            view = frame.copy()

            # Draw whatever YOLO saw, so a missing body_a is obvious.
            body_seen = False
            for obj in objects:
                name = getattr(obj, "class_name", "")
                box = getattr(obj, "bounding_box", None) or getattr(obj, "bbox", None)
                if box is None:
                    continue
                x1, y1, x2, y2 = (int(v) for v in list(box)[:4])
                col = (0, 200, 0) if name == "body_a" else (160, 160, 160)
                body_seen = body_seen or (name == "body_a")
                cv2.rectangle(view, (x1, y1), (x2, y2), col, 2)
                cv2.putText(view, name, (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

            if seats.found and len(seats.seats) == 2:
                found_frames += 1
                sep = float(seats.separation_px)
                samples.append(sep)
                best = max(best, sep)

                pts = []
                for s in seats.seats.values():
                    pts.append(s.xy)
                    cv2.circle(view, s.xy, int(max(6, s.radius)),
                               (0, 220, 255), 2)
                    cv2.circle(view, s.xy, 2, (0, 220, 255), -1)
                cv2.line(view, pts[0], pts[1], (0, 220, 255), 2)

                mid = ((pts[0][0] + pts[1][0]) // 2,
                       (pts[0][1] + pts[1][1]) // 2)
                cv2.putText(view, f"{sep:.0f} px", (mid[0] + 10, mid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2,
                            cv2.LINE_AA)
            else:
                last_reason = seats.reason or "both bores not visible"

            # ---- read-out panel ----
            median = float(np.median(samples[-90:])) if samples else 0.0
            text, colour = verdict(median)

            panel = np.zeros((150, view.shape[1], 3), dtype=np.uint8)
            panel[:, :] = (28, 28, 28)
            cv2.putText(panel, "SEAT SEPARATION", (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(panel, f"now {median:6.1f} px     best {best:6.1f} px",
                        (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.78, colour, 2,
                        cv2.LINE_AA)
            cv2.putText(panel, text, (16, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.64, colour, 2, cv2.LINE_AA)

            if median > 0:
                mm_per_px = CAD_SEAT_SEPARATION_MM / median
                sub = (f"{mm_per_px:.3f} mm/px   "
                       f"both bores found in {found_frames}/{frames} frames")
            elif not body_seen:
                sub = "body_a not detected - put it in the work area"
            else:
                sub = f"bores not found: {last_reason}"
            cv2.putText(panel, sub, (16, 128), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (170, 170, 170), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW, np.vstack([view, panel]))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   f"seat_resolution_{int(time.time())}.png")
                cv2.imwrite(out, np.vstack([view, panel]))
                print(f"saved {out}")

    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()

    # ---- summary ----
    print()
    print("=" * 66)
    print("SEAT RESOLUTION CHECK - RESULT")
    print("=" * 66)
    print(f"  frames examined       {frames}")
    print(f"  both bores found in   {found_frames} "
          f"({(100.0 * found_frames / max(1, frames)):.1f} %)")
    if samples:
        arr = np.asarray(samples, dtype=float)
        median = float(np.median(arr))
        print(f"  separation  median    {median:.1f} px")
        print(f"              best      {arr.max():.1f} px")
        print(f"              spread    {arr.std():.1f} px (std dev)")
        print(f"  scale                 {CAD_SEAT_SEPARATION_MM / median:.3f} "
              f"mm/px  (CAD separation {CAD_SEAT_SEPARATION_MM:.2f} mm)")
        print()
        print(f"  VERDICT: {verdict(median)[0]}")
        if median < GOOD_PX:
            factor = GOOD_PX / median
            print()
            print(f"  To reach {GOOD_PX:.0f} px, body_a needs to appear about "
                  f"{factor:.1f}x larger")
            print(f"  in the frame - move the camera roughly {factor:.1f}x "
                  f"closer, or zoom in by")
            print(f"  that much, and run this again.")
    else:
        print("  Both bores were never found together, so no separation "
              "could be measured.")
        print(f"  Last reason: {last_reason or 'unknown'}")
        print("  Check that body_a is in the work area, lit, and that both "
              "bores are visible.")
    print("=" * 66)
    print()
    print("  Nothing was changed. POSITION_VALIDATION_ENABLED is untouched.")
    print()
    print(f"  elapsed {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
