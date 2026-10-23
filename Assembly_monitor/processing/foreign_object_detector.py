"""
processing/foreign_object_detector.py
===============================================================================
SUBSYSTEM: Processing (SS4) - foreign object layer

THE PROBLEM THIS SOLVES
  The detector knows seven classes. A computer mouse, a screwdriver, a phone
  or a coffee cup left in the work area is not any of them, so YOLO reports
  nothing and the system says nothing. The operator can put an arbitrary
  object on the bench and the monitor will not notice it exists.

  That is a real gap for a process-verification system: "I did not recognise
  anything" and "there is nothing there" are very different statements, and
  the detector cannot tell them apart on its own.

HOW IT WORKS, AND WHY THIS WAY
  Detection cannot be the answer here - you cannot train a class for "any
  object in the world". So this works the other way round: it looks for
  anything in the work area that is NOT explained by the background and NOT
  explained by a recognised component.

      background subtraction  ->  foreground regions
      minus YOLO's boxes      ->  unexplained regions
      minus anything moving   ->  unexplained AND stationary
      held for N seconds      ->  a foreign object

  Each subtraction removes a specific false positive:

  MINUS YOLO's BOXES
      The assembly itself is foreground. Without this, body_a would be
      reported as a foreign object on every frame.

  MINUS ANYTHING MOVING
      The operator's hands are foreground, they are not a known class, and
      they are in the work area constantly. Hands move; an object placed on
      the bench does not. Requiring the region to hold still is what
      separates them, and it is the single most important rule in this file.

  HELD FOR N SECONDS
      A hand resting still for a moment is not a foreign object. Persistence
      turns a momentary coincidence into evidence.

THE BACKGROUND MODEL FREEZES UNDER FOREGROUND
  The background is updated only where there is no foreground. This matters
  more than it looks: with a plain running average, an object sitting still
  is gradually absorbed INTO the background, the foreground region fades,
  and the alert silently disappears after a few seconds - exactly when the
  object has been there longest and the warning matters most. Masked
  updating means a stationary object stays visible indefinitely, while the
  model still adapts to slow light changes everywhere else.

WHAT THIS IS NOT
  It does not identify the object. It cannot say "that is a mouse" - only
  "something is there that does not belong". That is the honest limit of a
  background-subtraction approach and it is also enough for the purpose:
  the operator can see what it is, they only need to be told to look.

  It also does not decide anything about assembly stages. Like the position
  validator, it reports; AssemblyStateMachine decides.

===============================================================================
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Defaults - all overridable from config/settings.py
# ═══════════════════════════════════════════════════════════════════════════

# Frames of quiet scene used to learn what the empty bench looks like.
DEFAULT_WARMUP_FRAMES = 40

# Greyscale difference above which a pixel counts as foreground. Low enough
# to catch a dark object on a dark bench, high enough to ignore sensor noise
# and the webcam's auto-exposure drift.
DEFAULT_DIFF_THRESHOLD = 30

# Smallest region worth reporting, as a fraction of the frame. Below this it
# is shadow, noise, or a cable. 0.6 % of a 640x480 frame is about 1800 px.
DEFAULT_MIN_AREA_FRAC = 0.006

# Largest region worth reporting. Anything bigger is a lighting change, the
# operator leaning over the bench, or the camera being moved - none of which
# is a foreign object, and all of which would otherwise fire constantly.
DEFAULT_MAX_AREA_FRAC = 0.40

# A region counts as stationary while its centre stays within this many
# pixels of where it was. Hands drift well beyond this even when "still".
DEFAULT_STATIONARY_PX = 14

# How long an unexplained, stationary region must persist before it is
# reported. The main defence against a hand resting for a moment.
DEFAULT_PERSIST_SEC = 2.5

# How much of a region must be covered by a YOLO box for the part to count
# as the explanation for it. Not 100 %, because a box rarely covers every
# pixel of its object's silhouette.
DEFAULT_OVERLAP_FRAC = 0.30

# A YOLO box with at least this much of ITS OWN area inside the region
# explains the region outright, however large the region is. A hand holding
# a part makes one big blob around a small box, and that blob is not a
# foreign object - it is the operator working. See _explained_by.
DEFAULT_CONTAINED_FRAC = 0.60

# ...but only if the contained part is a real share of the region, so one
# small nut cannot excuse a region fifty times its size.
DEFAULT_CONTAINED_MIN_REGION_FRAC = 0.06

# After a region has been explained by a detection, it stays explained for
# this long. YOLO drops a box for a frame or two all the time, and without
# this grace the assembly itself flickers into being a "foreign object"
# every time the detector blinks.
DEFAULT_EXPLAIN_GRACE_SEC = 2.5

# Background learning rate, applied only outside the foreground.
DEFAULT_BG_LEARN_RATE = 0.02

# A real object on the bench is COMPACT. A sheet of paper that has shifted,
# a shadow edge, a fold, the lip of a mat - these all appear as long thin
# strips, and in testing they were the whole of the false-alarm problem:
# 42 alerts in one session, almost all of them paper edges several hundred
# pixels wide and a few tens tall.
#
# Two shape tests remove them without touching real objects:
#   * the bounding box may not be more elongated than this
#   * the region must actually FILL a reasonable share of its own box,
#     which rejects thin diagonal edges that happen to span a square area
DEFAULT_MAX_ASPECT_RATIO = 3.2
DEFAULT_MIN_BOX_FILL = 0.35


# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ForeignRegion:
    """One unexplained thing in the work area."""
    region_id: int
    bbox: Tuple[int, int, int, int]        # x, y, w, h
    centre: Tuple[int, int]
    area_px: int
    first_seen: float
    last_seen: float
    last_moved: float
    last_explained: float = 0.0
    reported: bool = False

    def age_sec(self, now: float) -> float:
        return now - self.first_seen

    def still_sec(self, now: float) -> float:
        return now - self.last_moved

    def is_explained(self, now: float, grace: float) -> bool:
        return (now - self.last_explained) <= grace


@dataclass
class ForeignObjectResult:
    """What this frame showed. Read by the display and by main."""
    alert: bool = False
    message: Optional[str] = None
    regions: List[ForeignRegion] = field(default_factory=list)
    alerting: List[ForeignRegion] = field(default_factory=list)
    is_warmup: bool = False
    warmup_remaining: int = 0
    foreground_mask: Optional[Any] = None   # for the debug window

    @property
    def count(self) -> int:
        return len(self.alerting)


@dataclass
class ForeignObjectWarning:
    """Duck-typed to match AssemblyWarning, so SessionLogger.log_event and
    ProcessLogger.record_warning take it with no change to either."""
    warning_type: str = "FOREIGN_OBJECT"
    message: str = ""
    current_stage: Optional[str] = None
    observed: Dict[str, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    regions: List[Dict[str, Any]] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════

class ForeignObjectDetector:
    """Reports work-area regions that no recognised component explains.

    Use once per frame, after YOLO:

        result = foreign_detector.update(frame, object_result.objects)
        if result.alert:
            ...                      # the display draws it; main logs it
    """

    def __init__(
        self,
        roi_points=None,
        enabled: bool = True,
        warmup_frames: int = DEFAULT_WARMUP_FRAMES,
        diff_threshold: int = DEFAULT_DIFF_THRESHOLD,
        min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
        max_area_frac: float = DEFAULT_MAX_AREA_FRAC,
        stationary_px: int = DEFAULT_STATIONARY_PX,
        persist_sec: float = DEFAULT_PERSIST_SEC,
        overlap_frac: float = DEFAULT_OVERLAP_FRAC,
        contained_frac: float = DEFAULT_CONTAINED_FRAC,
        contained_min_region_frac: float = DEFAULT_CONTAINED_MIN_REGION_FRAC,
        explain_grace_sec: float = DEFAULT_EXPLAIN_GRACE_SEC,
        bg_learn_rate: float = DEFAULT_BG_LEARN_RATE,
        max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO,
        min_box_fill: float = DEFAULT_MIN_BOX_FILL,
        on_warning=None,
    ):
        self.enabled = bool(enabled)
        self.warmup_frames = int(warmup_frames)
        self.diff_threshold = int(diff_threshold)
        self.min_area_frac = float(min_area_frac)
        self.max_area_frac = float(max_area_frac)
        self.stationary_px = int(stationary_px)
        self.persist_sec = float(persist_sec)
        self.overlap_frac = float(overlap_frac)
        self.contained_frac = float(contained_frac)
        self.contained_min_region_frac = float(contained_min_region_frac)
        self.explain_grace_sec = float(explain_grace_sec)
        self.bg_learn_rate = float(bg_learn_rate)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.min_box_fill = float(min_box_fill)
        self._on_warning = on_warning

        self._roi_points = roi_points
        self._roi_mask = None

        self._background = None          # float32 greyscale
        self._frames_seen = 0
        self._regions: List[ForeignRegion] = []
        self._next_id = 1

        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        self.stats = {
            "frames": 0,
            "alerts_raised": 0,
            "regions_tracked": 0,
            "alert_seconds": 0.0,
        }
        self._alert_active = False
        self._alert_started = 0.0

        if self.enabled:
            print(f"[ForeignObject] enabled | warmup {self.warmup_frames} frames"
                  f" | must hold still {self.persist_sec}s")
        else:
            print("[ForeignObject] disabled in settings.")

    # ==================================================================

    def reset_background(self) -> None:
        """Re-learn the empty bench. Call after deliberately changing the
        scene - moving the fixture, changing the lighting - so the change
        is not reported forever as a foreign object."""
        self._background = None
        self._frames_seen = 0
        self._regions.clear()
        self._alert_active = False
        print("[ForeignObject] background reset - re-learning the empty bench")

    @property
    def is_warmup(self) -> bool:
        return self._frames_seen < self.warmup_frames

    # ==================================================================

    def update(self, frame, detections: Sequence[Any] = (),
               now: Optional[float] = None) -> ForeignObjectResult:
        """One frame. Never raises: any failure degrades to 'no alert'."""
        if not self.enabled or frame is None:
            return ForeignObjectResult()

        now = time.time() if now is None else float(now)
        self.stats["frames"] += 1

        try:
            return self._update(frame, detections or (), now)
        except Exception as exc:                            # noqa: BLE001
            # A fault in a supplementary check must never take the monitor
            # down. Report once and carry on blind rather than crash.
            print(f"[ForeignObject] error, ignored this frame: {exc}")
            return ForeignObjectResult()

    # ------------------------------------------------------------------
    def _update(self, frame, detections, now) -> ForeignObjectResult:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grey = cv2.GaussianBlur(grey, (5, 5), 0)
        h, w = grey.shape[:2]

        if self._roi_mask is None or self._roi_mask.shape[:2] != (h, w):
            self._roi_mask = self._build_roi_mask(h, w)

        # ---- warm-up: learn the empty bench ----
        if self._background is None:
            self._background = grey.astype(np.float32)
            self._frames_seen = 1
            return ForeignObjectResult(
                is_warmup=True,
                warmup_remaining=self.warmup_frames - 1)

        if self._frames_seen < self.warmup_frames:
            cv2.accumulateWeighted(grey, self._background, 0.15)
            self._frames_seen += 1
            return ForeignObjectResult(
                is_warmup=True,
                warmup_remaining=self.warmup_frames - self._frames_seen)

        self._frames_seen += 1

        # ---- foreground ----
        diff = cv2.absdiff(grey, cv2.convertScaleAbs(self._background))
        _, mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        if self._roi_mask is not None:
            mask = cv2.bitwise_and(mask, self._roi_mask)

        # Learn the background ONLY where there is no foreground, so a
        # stationary object is never absorbed into it. See the module
        # docstring - this is the difference between an alert that holds
        # and one that quietly vanishes while the object is still there.
        background_only = cv2.bitwise_not(mask)
        cv2.accumulateWeighted(grey, self._background, self.bg_learn_rate,
                               mask=background_only)

        # ---- regions ----
        frame_area = float(h * w)
        min_area = self.min_area_frac * frame_area
        max_area = self.max_area_frac * frame_area

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        observed = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)

            # Shape tests. A mouse, a phone, a tool or a cup is compact; a
            # shifted sheet of paper or a shadow edge is a long thin strip.
            # Without these, moving the paperwork on the bench produced a
            # continuous stream of warnings and the operator learns to
            # ignore the banner - which costs more than the feature gives.
            long_side, short_side = max(bw, bh), max(1, min(bw, bh))
            if long_side / short_side > self.max_aspect_ratio:
                continue
            if area / float(max(1, bw * bh)) < self.min_box_fill:
                continue

            observed.append({
                "bbox": (x, y, bw, bh),
                "centre": (x + bw // 2, y + bh // 2),
                "area": int(area),
            })

        boxes = self._detection_boxes(detections)
        self._track(observed, boxes, now)

        # ---- which of them are worth reporting ----
        alerting = [
            r for r in self._regions
            if not r.is_explained(now, self.explain_grace_sec)
            and r.still_sec(now) >= self.persist_sec
            and r.age_sec(now) >= self.persist_sec
        ]

        result = ForeignObjectResult(
            alert=bool(alerting),
            regions=list(self._regions),
            alerting=alerting,
            foreground_mask=mask,
        )
        if alerting:
            result.message = self._message(alerting)

        self._track_alert_state(bool(alerting), now)
        self._maybe_warn(alerting, now)
        return result

    # ------------------------------------------------------------------
    def _build_roi_mask(self, h, w):
        """Limit the search to the work area, if one was calibrated.

        Outside the ROI there is a whole room full of things that are not
        assembly components - a chair, a door, a person walking past - and
        none of them are what this check is about."""
        if self._roi_points is None:
            return None
        try:
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.asarray(self._roi_points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 255)
            return mask
        except Exception as exc:                            # noqa: BLE001
            print(f"[ForeignObject] ROI mask unusable, searching the whole "
                  f"frame instead: {exc}")
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def _detection_boxes(detections) -> List[Tuple[int, int, int, int]]:
        """YOLO boxes as (x1, y1, x2, y2). Tolerant of shape."""
        out = []
        for det in detections:
            box = None
            for attr in ("bounding_box", "bbox", "box", "xyxy"):
                val = getattr(det, attr, None)
                if val is not None:
                    box = val
                    break
            if box is None and isinstance(det, dict):
                for k in ("bounding_box", "bbox", "box", "xyxy"):
                    if det.get(k) is not None:
                        box = det[k]
                        break
            if box is None:
                continue
            try:
                x1, y1, x2, y2 = (int(v) for v in list(box)[:4])
            except (TypeError, ValueError):
                continue
            out.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        return out

    def _explained_by(self, bbox, boxes) -> bool:
        """Is this region accounted for by a recognised component?

        Two ways it can be, and BOTH are needed.

        1. Coverage. At least `overlap_frac` of the region sits under a YOLO
           box. This is the ordinary case: a part lying on the bench.

        2. Containment. A YOLO box sits almost entirely INSIDE the region.

        Test 2 is the one that was missing, and its absence was the whole of
        the false-alarm problem. When the operator picks a part up, the
        foreground blob is hand + forearm + part + the paper they drag with
        it - one connected region of maybe 800x290 px. The body_a box inside
        it is about 220x180. That is 17 % coverage, so test 1 alone calls a
        region that demonstrably CONTAINS a recognised part "unrecognised",
        and the banner fires with nothing foreign on the bench at all.

        The containment test needs the part to be a meaningful share of the
        region (`contained_min_region_frac`), otherwise a single nut box
        would excuse a region fifty times its size.
        """
        x, y, bw, bh = bbox
        region_area = float(max(1, bw * bh))
        rx1, ry1, rx2, ry2 = x, y, x + bw, y + bh

        covered = 0.0
        for bx1, by1, bx2, by2 in boxes:
            ix1, iy1 = max(rx1, bx1), max(ry1, by1)
            ix2, iy2 = min(rx2, bx2), min(ry2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = float((ix2 - ix1) * (iy2 - iy1))
            covered += inter

            # --- test 2: is this box essentially inside the region? ---
            box_area = float(max(1, (bx2 - bx1) * (by2 - by1)))
            if (inter / box_area) >= self.contained_frac and \
               (box_area / region_area) >= self.contained_min_region_frac:
                return True

        # --- test 1: is enough of the region under some box? ---
        return (covered / region_area) >= self.overlap_frac

    # ------------------------------------------------------------------
    def _track(self, observed, boxes, now) -> None:
        """Match this frame's regions to the ones already being followed.

        Nearest-centre matching, which is adequate because the regions that
        matter are by definition the ones that are not moving. A region that
        darts around the frame fails the stationary test anyway, so a more
        elaborate tracker would buy nothing here.
        """
        unmatched = list(self._regions)
        seen = []

        for obs in observed:
            cx, cy = obs["centre"]
            best, best_d = None, float("inf")
            for r in unmatched:
                d = math.hypot(cx - r.centre[0], cy - r.centre[1])
                if d < best_d:
                    best, best_d = r, d

            # Generous match radius: the same object's contour shifts a bit
            # frame to frame as lighting and occlusion change.
            if best is not None and best_d <= max(self.stationary_px * 4, 40):
                if best_d > self.stationary_px:
                    best.last_moved = now       # it moved: restart the clock
                best.bbox = obs["bbox"]
                best.centre = obs["centre"]
                best.area_px = obs["area"]
                best.last_seen = now
                if self._explained_by(obs["bbox"], boxes):
                    best.last_explained = now
                unmatched.remove(best)
                seen.append(best)
            else:
                r = ForeignRegion(
                    region_id=self._next_id,
                    bbox=obs["bbox"],
                    centre=obs["centre"],
                    area_px=obs["area"],
                    first_seen=now,
                    last_seen=now,
                    last_moved=now,
                    last_explained=(now if self._explained_by(obs["bbox"], boxes)
                                    else 0.0),
                )
                self._next_id += 1
                self.stats["regions_tracked"] += 1
                seen.append(r)

        # Drop regions that have not been seen for a moment. Short, because
        # an object that genuinely disappeared has been picked up, and the
        # warning should clear as soon as the operator removes it.
        keep_window = 0.8
        self._regions = seen + [r for r in unmatched
                                if (now - r.last_seen) <= keep_window]

    # ------------------------------------------------------------------
    def _message(self, alerting) -> str:
        n = len(alerting)
        if n == 1:
            return ("Unrecognised object in the work area. Remove anything "
                    "that is not part of this assembly.")
        return (f"{n} unrecognised objects in the work area. Remove anything "
                f"that is not part of this assembly.")

    def _track_alert_state(self, active, now) -> None:
        if active and not self._alert_active:
            self._alert_active = True
            self._alert_started = now
        elif not active and self._alert_active:
            self._alert_active = False
            self.stats["alert_seconds"] += round(now - self._alert_started, 2)

    def _maybe_warn(self, alerting, now) -> None:
        """One warning per object, not one per frame."""
        fresh = [r for r in alerting if not r.reported]
        if not fresh:
            return
        for r in fresh:
            r.reported = True
        self.stats["alerts_raised"] += 1

        warning = ForeignObjectWarning(
            warning_type="FOREIGN_OBJECT",
            message=self._message(alerting),
            timestamp=now,
            observed={},
            regions=[{
                "region_id": r.region_id,
                "bbox": list(r.bbox),
                "area_px": r.area_px,
                "still_sec": round(r.still_sec(now), 2),
            } for r in alerting],
        )
        if self._on_warning:
            self._on_warning(warning)
        print(f"[FOREIGN] {len(alerting)} unrecognised object(s) in the "
              f"work area")

    # ------------------------------------------------------------------
    def markers(self) -> List[Dict[str, Any]]:
        """Boxes for the display to draw around what it found."""
        now = time.time()
        return [{
            "bbox": r.bbox,
            "age_sec": round(r.age_sec(now), 1),
            "alerting": (not r.is_explained(now, self.explain_grace_sec)
                         and r.still_sec(now) >= self.persist_sec),
        } for r in self._regions]


# ═══════════════════════════════════════════════════════════════════════════
# Self-test: synthetic frames, no camera needed
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    class Det:
        def __init__(self, x1, y1, x2, y2):
            self.bounding_box = (x1, y1, x2, y2)
            self.class_name = "body_a"
            self.confidence = 0.9

    H, W = 480, 640
    rng = np.random.default_rng(3)

    def bench():
        """An empty bench with a little sensor noise."""
        f = np.full((H, W, 3), 85, np.uint8)
        f[300:, :] = 70
        n = rng.normal(0, 2.2, (H, W, 3))
        return np.clip(f.astype(np.float32) + n, 0, 255).astype(np.uint8)

    def put(f, x, y, w, h, colour):
        f[y:y + h, x:x + w] = colour
        return f

    d = ForeignObjectDetector(persist_sec=1.0, warmup_frames=25)
    t = 0.0
    STEP = 1 / 25.0

    def run(label, nframes, draw=None, dets=(), expect=None):
        global t
        res = None
        for i in range(nframes):
            f = bench()
            if draw:
                draw(f, i)
            res = d.update(f, dets, now=t)
            t += STEP
        got = "ALERT" if res.alert else "quiet"
        ok = "" if expect is None else ("  OK" if (res.alert == expect)
                                        else "  <-- UNEXPECTED")
        print(f"  {label:52s} {got}{ok}")
        return res

    print("\nForeignObjectDetector self-test")
    print("-" * 72)
    run("warm-up on the empty bench", 30, expect=False)
    run("empty bench, settled", 25, expect=False)

    print("\n  a hand moving across the bench (must stay quiet)")
    run("hand sweeping left to right",
        60, draw=lambda f, i: put(f, 60 + i * 7, 180, 110, 150, 55),
        expect=False)
    run("bench empty again", 30, expect=False)

    print("\n  a recognised component placed and left (must stay quiet)")
    part = lambda f, i: put(f, 250, 230, 120, 90, 190)
    run("body_a on the bench, YOLO sees it",
        90, draw=part, dets=[Det(250, 230, 370, 320)], expect=False)

    print("\n  an unrecognised object placed and left (must ALERT)")
    mouse = lambda f, i: put(f, 250, 230, 120, 90, 190)
    r = run("same object, YOLO reports nothing",
            90, draw=mouse, dets=[], expect=True)
    if r.message:
        print(f"      screen: {r.message}")

    print("\n  object removed (alert must clear)")
    run("bench empty again", 40, expect=False)

    print("\n  stats:", {k: v for k, v in d.stats.items() if k != "frames"})
    print("-" * 72)
    print("Expected: quiet for warm-up, hands, and recognised parts;")
    print("ALERT only for the stationary object YOLO cannot explain.\n")
