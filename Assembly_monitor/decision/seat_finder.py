"""
decision/seat_finder.py
===============================================================================
Finds the two bushing seats IN THE LIVE IMAGE, every frame, by looking at the
part itself. No calibration clicks, no stored screen coordinates.

WHY THIS REPLACES THE CALIBRATED-ANCHOR APPROACH
  The original design stored the two seat centres as image coordinates, marked
  once by hand. Those points are fixed to the SCREEN, not to the part, so the
  moment body_a is moved or turned the markers stay behind. That was a stated
  limitation of that design, not a bug: it assumed a fixed fixture.

  The pose tracker was added to lift that limitation by matching image features
  on body_a and warping the stored anchors. It cannot work reliably here, and
  the reason is structural rather than a tuning problem:

    * a homography maps a PLANE. body_a is a 3-D part, so once it is turned to
      a different face there is no homography that maps the old view to the
      new one.
    * the part is smooth printed plastic with almost no texture - about 50
      SIFT features on the whole body, most of them on edges and shadows that
      move when the part moves.

  With few features and a wrong model, RANSAC still returns "a" homography.
  It passes an inlier count, it passes a reprojection check, and it is
  geometrically meaningless: in the failing frame both seat anchors collapsed
  onto nearly the same point, about 90 px away from either real hole.

  A confidently wrong marker is worse than no marker. The operator is told to
  put the bushing in a place that is not a hole.

WHAT THIS DOES INSTEAD
  It finds the holes directly, because they are the most visually distinctive
  thing about this part: circular regions inside a solid green silhouette that
  are not green plastic - either a dark empty bore, or the grey metal of a
  bushing already seated.

      green silhouette of body_a
        -> fill it
        -> regions inside that are not plastic
        -> keep the round ones
        -> expect: one large bore + two small seats of similar size, close
           together
        -> label them by distance from the large bore

  Measured on the calibration photograph, this recovers the two seats to
  0.7 px and 1.1 px of the points that were marked by hand. It needs no
  calibration at all, and it works at whatever pose the part happens to be in,
  because it is looking at the part rather than remembering a screen position.

WHY THE LABELLING RULE IS WHAT IT IS
  Knowing where the two holes are is not enough; the system has to say which
  one takes the FIRST bushing. Any rule based on the image (left, top) breaks
  as soon as the part is turned round.

  The rule used here is a property of the part itself: of the two seats, the
  one FARTHER from the large bore is Seat_1 and the nearer is Seat_2. That
  ordering is fixed by the geometry of the part, so it survives rotation and
  viewpoint. It is read once from the calibration photograph, so if the
  convention is ever the other way round it is corrected in one place.

WHEN IT REFUSES TO ANSWER
  The structural expectation - one big bore plus two similar small holes close
  together - is deliberately strict. If the part is edge-on, a hand covers a
  seat, or the lighting washes the holes out, the pattern will not be found
  and this returns "not confident" rather than guessing. The display then asks
  the operator to show the part more clearly. That is the behaviour the
  homography approach lacked, and the reason it produced the wrong answer
  confidently instead of saying it did not know.

===============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Tuning - every value is a ratio to the part's own size or colour, never an
# absolute pixel or brightness, so moving the camera closer or changing the
# lamp does not invalidate them.
# ═══════════════════════════════════════════════════════════════════════════

# HSV band for the green printed part. Wide on purpose: it only has to
# separate the part from the bench, and the per-frame colour statistics below
# do the precise work.
GREEN_LO = (40, 30, 3)
GREEN_HI = (115, 255, 235)

# A candidate region must be at least this fraction of the part's area, and
# at most this fraction. Rejects speckle and the part's own shadow.
MIN_AREA_FRAC = 0.0015
MAX_AREA_FRAC = 0.2500

# How round a region must be, as circularity  4*pi*area / perimeter^2.
# 1.0 is a perfect circle. The bores measure about 0.70 in the calibration
# photograph, and a bore seen at an angle is an ellipse, so this cannot be
# strict. Circularity is used rather than area / (pi r^2) of the minimum
# enclosing circle: that circle is generous around a slightly ragged contour,
# which put the real bores at 0.34 and would have rejected every one of them.
MIN_CIRCULARITY = 0.45

# The two seats must be similar in size - they take identical bushings.
MAX_SEAT_AREA_RATIO = 2.6

# The two seats sit on the same flange, so they are close together relative to
# the whole part. In CAD they are 18.00 mm apart on a part whose overall
# extent is of the order of 120 mm, about 0.15 of the diagonal; 0.20 leaves
# headroom for perspective without admitting unrelated pairs.
#
# This bound does most of the work of rejecting false pairs, and it has to be
# this tight. With it at 0.45 the finder preferred two unrelated dark spots
# 145 px apart - they happened to be closer in AREA than the true pair - over
# the two real holes 25 px apart. Closeness is the stronger signal of the two,
# and the scoring below now says so as well.
MAX_SEAT_SEPARATION_FRAC = 0.20

# ... but not on top of each other.
MIN_SEAT_SEPARATION_FRAC = 0.02

# A seat counts as holding a metal bushing when its saturation is well below
# the plastic's. Metal is grey; the plastic is strongly coloured.
METAL_SATURATION_FRAC = 0.55
METAL_MIN_VALUE_FRAC = 0.45

# Frames of agreement before the finder's answer is treated as settled.
DEFAULT_STABLE_FRAMES = 5

# How far a seat may move between frames and still count as the same seat,
# as a fraction of the part's diagonal.
SEAT_MATCH_FRAC = 0.12


# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FoundSeat:
    name: str                       # "Seat_1" | "Seat_2"
    centre: Tuple[float, float]     # pixels, in the full frame
    radius: float
    occupied: bool                  # a bushing appears to be in it
    area: float
    distance_from_bore: float

    @property
    def xy(self) -> Tuple[int, int]:
        return (int(round(self.centre[0])), int(round(self.centre[1])))


@dataclass
class SeatFindResult:
    found: bool = False
    reason: str = ""
    seats: Dict[str, FoundSeat] = field(default_factory=dict)
    bore: Optional[Tuple[float, float]] = None
    body_box: Optional[Tuple[int, int, int, int]] = None
    stable: bool = False
    frames_held: int = 0
    separation_px: float = 0.0

    def seat(self, name: str) -> Optional[FoundSeat]:
        return self.seats.get(name)


# ═══════════════════════════════════════════════════════════════════════════

class SeatFinder:
    """Locates Seat_1 and Seat_2 on body_a in the current frame."""

    def __init__(
        self,
        seat_1_is_farther_from_bore: bool = True,
        stable_frames: int = DEFAULT_STABLE_FRAMES,
        min_body_box_px: int = 60,
        debug: bool = False,
    ):
        # Read off the calibration photograph once: in that image the seat
        # marked as the FIRST bushing seat was the farther of the two from
        # the large bore. If your convention is the other way round, this is
        # the single place to change it.
        self.seat_1_is_farther = bool(seat_1_is_farther_from_bore)
        self.stable_frames = int(stable_frames)
        self.min_body_box_px = int(min_body_box_px)
        self.debug = bool(debug)

        self._last: Optional[SeatFindResult] = None
        self._held = 0
        self._last_centres: Dict[str, Tuple[float, float]] = {}

        self.stats = {"frames": 0, "found": 0, "refused": 0}

    # ------------------------------------------------------------------
    def update(self, frame, detections: Sequence[Any]) -> SeatFindResult:
        """One frame. Never raises; any failure becomes 'not confident'."""
        self.stats["frames"] += 1
        try:
            result = self._find(frame, detections)
        except Exception as exc:                            # noqa: BLE001
            result = SeatFindResult(False, f"seat finder error: {exc}")

        # Temporal agreement, so a one-frame fluke never drives the display.
        if result.found and self._last and self._last.found:
            moved = self._max_movement(result)
            diag = self._diagonal(result.body_box)
            if diag and moved <= SEAT_MATCH_FRAC * diag:
                self._held += 1
            else:
                self._held = 1
        elif result.found:
            self._held = 1
        else:
            self._held = 0

        result.frames_held = self._held
        result.stable = result.found and self._held >= self.stable_frames

        if result.found:
            self.stats["found"] += 1
            self._last_centres = {n: s.centre for n, s in result.seats.items()}
        else:
            self.stats["refused"] += 1

        self._last = result
        return result

    # ------------------------------------------------------------------
    def _max_movement(self, result: SeatFindResult) -> float:
        if not self._last_centres:
            return 0.0
        worst = 0.0
        for name, seat in result.seats.items():
            prev = self._last_centres.get(name)
            if prev:
                worst = max(worst, math.dist(seat.centre, prev))
        return worst

    @staticmethod
    def _diagonal(box) -> float:
        if not box:
            return 0.0
        x1, y1, x2, y2 = box
        return math.hypot(x2 - x1, y2 - y1)

    # ------------------------------------------------------------------
    def _find(self, frame, detections) -> SeatFindResult:
        if frame is None:
            return SeatFindResult(False, "no camera frame")

        box = self._body_box(detections, frame.shape)
        if box is None:
            return SeatFindResult(False, "body_a not detected")
        x1, y1, x2, y2 = box
        if (x2 - x1) < self.min_body_box_px or (y2 - y1) < self.min_body_box_px:
            return SeatFindResult(
                False, "body_a is too small in the image - move the camera "
                       "closer", body_box=box)

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        green = cv2.inRange(hsv, np.array(GREEN_LO, np.uint8),
                            np.array(GREEN_HI, np.uint8))
        green = cv2.morphologyEx(
            green, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))

        contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return SeatFindResult(False, "the green part is not visible",
                                  body_box=box)
        part = max(contours, key=cv2.contourArea)
        part_area = float(cv2.contourArea(part))
        if part_area < 400:
            return SeatFindResult(False, "the green part is too small to read",
                                  body_box=box)

        silhouette = np.zeros_like(green)
        cv2.drawContours(silhouette, [part], -1, 255, -1)
        # Pull in from the outline so the silhouette edge itself is never
        # mistaken for a hole. Small, because the seats sit near the edge of
        # the flange and a heavy erosion deletes them.
        inside = cv2.erode(
            silhouette,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        if inside.sum() < 200:
            return SeatFindResult(False, "the part is too small to read",
                                  body_box=box)

        S = hsv[:, :, 1].astype(np.int16)
        V = hsv[:, :, 2].astype(np.int16)

        # The part's own lit colour. Measured from the brighter half of the
        # silhouette so shadowed faces and the holes themselves do not drag
        # the reference down - that was what made a fixed or percentile
        # threshold swing wildly between one photograph and the next.
        lit = V[inside] > np.percentile(V[inside], 55)
        if lit.sum() < 50:
            return SeatFindResult(False, "the part is too dark to read",
                                  body_box=box)
        s_plastic = float(np.median(S[inside][lit]))
        v_plastic = float(np.median(V[inside][lit]))

        # A hole is a region that is markedly darker than the lit plastic, or
        # markedly less saturated - the first catches an empty bore, the
        # second catches the grey metal of a bushing already seated.
        dark = V < max(v_plastic * 0.45, 25)
        grey = S < max(s_plastic * METAL_SATURATION_FRAC, 25)
        cand = ((dark | grey) & inside).astype(np.uint8) * 255
        cand = cv2.morphologyEx(
            cand, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        cand = cv2.morphologyEx(
            cand, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        regions = []
        rc, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        for c in rc:
            a = float(cv2.contourArea(c))
            if a < part_area * MIN_AREA_FRAC or a > part_area * MAX_AREA_FRAC:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if r < 3:
                continue
            per = cv2.arcLength(c, True)
            if per <= 0 or (4.0 * math.pi * a) / (per * per) < MIN_CIRCULARITY:
                continue
            m = np.zeros_like(green)
            cv2.circle(m, (int(cx), int(cy)), max(2, int(r * 0.5)), 255, -1)
            regions.append({
                "area": a,
                "centre": (cx + x1, cy + y1),      # back to full-frame pixels
                "radius": r,
                "S": float(np.median(S[m > 0])),
                "V": float(np.median(V[m > 0])),
            })

        if self.debug:
            print(f"[SeatFinder] part area {part_area:.0f}, "
                  f"{len(regions)} circular regions, "
                  f"plastic S={s_plastic:.0f} V={v_plastic:.0f}")

        if len(regions) < 3:
            return SeatFindResult(
                False,
                f"only {len(regions)} holes visible - turn the part so both "
                f"bushing holes and the large bore can be seen",
                body_box=box)

        regions.sort(key=lambda d: -d["area"])
        bore = regions[0]

        # The two seats: of the remaining regions, the most similar pair that
        # also sits close together. Testing the pair as a UNIT is what makes
        # this robust - two identical holes a short distance apart is a strong
        # signature, far harder to satisfy by accident than any single region.
        rest = regions[1:]
        diag = math.hypot(x2 - x1, y2 - y1)
        best, best_score = None, None
        for i in range(len(rest)):
            for j in range(i + 1, len(rest)):
                a, b = rest[i], rest[j]
                ratio = max(a["area"], b["area"]) / max(min(a["area"], b["area"]), 1)
                if ratio > MAX_SEAT_AREA_RATIO:
                    continue
                sep = math.dist(a["centre"], b["centre"])
                if sep > MAX_SEAT_SEPARATION_FRAC * diag:
                    continue
                if sep < MIN_SEAT_SEPARATION_FRAC * diag:
                    continue
                # Separation is weighted far above area similarity: two
                # identical-looking marks at opposite ends of the part are
                # not a seat pair, whereas two slightly different-sized
                # holes a few millimetres apart almost certainly are.
                score = ratio + 6.0 * (sep / max(diag, 1.0))
                if best_score is None or score < best_score:
                    best_score, best = score, (a, b)

        if best is None:
            return SeatFindResult(
                False,
                "could not identify a matching pair of bushing holes - show "
                "the flange more clearly",
                body_box=box)

        a, b = best
        da = math.dist(a["centre"], bore["centre"])
        db = math.dist(b["centre"], bore["centre"])
        farther, nearer = (a, b) if da >= db else (b, a)
        first, second = ((farther, nearer) if self.seat_1_is_farther
                         else (nearer, farther))

        def make(name, d):
            occupied = (d["S"] < s_plastic * METAL_SATURATION_FRAC
                        and d["V"] > v_plastic * METAL_MIN_VALUE_FRAC)
            return FoundSeat(
                name=name, centre=d["centre"], radius=d["radius"],
                occupied=bool(occupied), area=d["area"],
                distance_from_bore=math.dist(d["centre"], bore["centre"]))

        seats = {"Seat_1": make("Seat_1", first),
                 "Seat_2": make("Seat_2", second)}

        return SeatFindResult(
            found=True,
            reason="both bushing holes located on body_a",
            seats=seats,
            bore=bore["centre"],
            body_box=box,
            separation_px=math.dist(first["centre"], second["centre"]),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _body_box(detections, shape) -> Optional[Tuple[int, int, int, int]]:
        """The highest-confidence body_a box, padded a little."""
        best, best_conf = None, -1.0
        for det in detections or []:
            if str(getattr(det, "class_name", "")) != "body_a":
                continue
            conf = float(getattr(det, "confidence", 0.0))
            box = getattr(det, "bounding_box", None)
            if box is None or len(box) < 4 or conf <= best_conf:
                continue
            best, best_conf = [int(round(float(v))) for v in box[:4]], conf
        if best is None:
            return None
        h, w = shape[:2]
        x1, y1, x2, y2 = best
        px, py = max(4, int((x2 - x1) * 0.06)), max(4, int((y2 - y1) * 0.06))
        return (max(0, x1 - px), max(0, y1 - py),
                min(w, x2 + px), min(h, y2 + py))

    # ------------------------------------------------------------------
    def markers(self, required_seat: Optional[str] = None) -> List[Dict[str, Any]]:
        """Drawable markers for the display.

        Returns a single 'not found' entry rather than stale positions when
        the holes cannot be seen. Showing where the holes were a second ago is
        the mistake that started all of this.
        """
        r = self._last
        if r is None or not r.found:
            return [{
                "seats_found": False,
                "message": "LOOKING FOR THE BUSHING HOLES - show body_a clearly",
                "detail": (r.reason if r else "no frame processed yet"),
            }]
        out = []
        for name, seat in r.seats.items():
            out.append({
                "seats_found": True,
                "name": name,
                "label": "1st hole" if name == "Seat_1" else "2nd hole",
                "xy": seat.xy,
                "radius": int(round(seat.radius)),
                "occupied": seat.occupied,
                "required": (required_seat == name),
                "stable": r.stable,
            })
        return out

    def reset(self) -> None:
        self._last = None
        self._held = 0
        self._last_centres.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Standalone check: run this on your own photographs before trusting it.
#
#   python decision\seat_finder.py  <image-or-folder> [...]
#
# For each image it prints what it found and writes <name>_seats.png with the
# result drawn on. Use clean camera frames, NOT screenshots of the monitor
# window - the overlay graphics drawn on a screenshot are themselves
# non-plastic regions inside the part and will be picked up as holes.
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path

    paths: List[Path] = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                paths.extend(sorted(p.glob(ext)))
        elif p.exists():
            paths.append(p)

    if not paths:
        print(__doc__.split("Standalone check")[0][-400:])
        print("\nusage:  python decision/seat_finder.py <image-or-folder>")
        raise SystemExit(1)

    class _Whole:
        """Stands in for a YOLO body_a box when testing on a still."""
        class_name = "body_a"
        confidence = 1.0

        def __init__(self, shape):
            self.bounding_box = (0, 0, shape[1], shape[0])

    ok = 0
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            print(f"{p.name:40s} could not be read")
            continue
        f = SeatFinder(debug=True)
        r = f.update(im, [_Whole(im.shape)])
        if not r.found:
            print(f"{p.name:40s} NOT FOUND - {r.reason}")
            continue
        ok += 1
        s1, s2 = r.seats["Seat_1"], r.seats["Seat_2"]
        print(f"{p.name:40s} OK  "
              f"Seat_1 {s1.xy} {'occupied' if s1.occupied else 'empty':8s} | "
              f"Seat_2 {s2.xy} {'occupied' if s2.occupied else 'empty':8s} | "
              f"{r.separation_px:.0f} px apart")
        out = im.copy()
        if r.bore:
            cv2.circle(out, (int(r.bore[0]), int(r.bore[1])), 6, (200, 200, 0), 2)
            cv2.putText(out, "bore", (int(r.bore[0]) + 8, int(r.bore[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)
        for name, seat in r.seats.items():
            col = (0, 200, 0) if name == "Seat_1" else (0, 170, 255)
            cv2.circle(out, seat.xy, max(10, int(seat.radius * 1.6)), col, 2)
            cv2.putText(out, ("1st hole" if name == "Seat_1" else "2nd hole")
                        + (" (full)" if seat.occupied else ""),
                        (seat.xy[0] + 14, seat.xy[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        cv2.imwrite(str(p.with_name(p.stem + "_seats.png")), out)

    print(f"\n{ok} of {len(paths)} images read successfully.")
    print("Check each *_seats.png: the two rings must sit on the two real")
    print("holes, and '1st hole' must be the one you want filled first.")
    print("If 1st and 2nd are swapped, set seat_1_is_farther_from_bore=False.")
