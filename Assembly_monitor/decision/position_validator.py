"""
decision/position_validator.py
===============================================================================
SUBSYSTEM: Decision (SS5) - position layer

WHAT THIS IS FOR
  YOLO answers "which parts are on the bench, and how many". It does not
  answer "is this bushing in the RIGHT hole". The two bushing seats in
  body_a are 18.00 mm apart (1.63x the part's own size), so a bushing in
  the wrong seat produces the same class counts as a bushing in the right
  one, and the state machine alone would confirm Stage_01 either way.

  This module answers exactly one question:

      Is the detected part in the seat that this stage requires?

  Nothing else. It does NOT decide the assembly stage, it does not advance
  anything and it never writes to the state machine. AssemblyStateMachine
  stays the single authority on stages; it merely asks this module for
  permission before confirming a stage that has a seat requirement.

WHY THERE ARE TWO BACKENDS
  Experiment POS-FEAS-01 established that the two seats ARE distinguishable
  from one oblique camera (3.38 % of the seat-region pixels change at 35
  degrees elevation). That is a statement about observability, not about
  any particular algorithm. Two algorithms can exploit it:

  "geometric"  - calibrates seat centres in one reference view, then tracks
                 body_a visually and transforms those seat anchors into the
                 current frame. A bushing is assigned to the nearer tracked
                 seat. Needs no training or GPU. If the body cannot be tracked,
                 markers are hidden and stage confirmation is blocked safely.

  "classifier" - the trained binary model (correct / wrong_position) run on
                 a crop around the seats. Needs the synthetic dataset and a
                 training run first, and cannot by itself say which seat is
                 occupied - only that the pattern looks wrong.

  "both"       - both must agree before a WRONG verdict is allowed to block
                 a stage. This is the conservative setting and the one to
                 use for the recorded validation runs.

  The backend is chosen in position_config.json, so the live system does
  not have to change when the classifier becomes available.

BODY-RELATIVE TRACKING AND LIMITS
  The reference calibration image is used to initialise feature matching.
  Subsequent frames track body_a incrementally and transform seat anchors,
  so the marks move with the part instead of staying on the screen. This
  handles translation, scale, in-plane rotation and moderate view change. It
  is not full 6-DoF pose estimation: if the visible face changes too much,
  tracking fails safely, markers are hidden and Stage_01/02 cannot confirm.
  Start the monitor with body_a in its calibration view, then move it while
  keeping enough of the body visible for frame-to-frame tracking.

TEMPORAL BEHAVIOUR
  YOLO jitters, hands pass over the seats, and an empty bore can read as a
  bushing for a few frames, so a verdict must hold for a configurable
  number of consecutive frames before it counts as stable. Stability then
  governs exactly one thing: whether the OPERATOR is told. An operator
  accused of a mistake on every stray frame stops trusting the system.

  Withholding a stage confirmation is the opposite case and is immediate.
  Any wrong verdict defers the confirmation, stable or not, because the
  two mistakes are not symmetric: a deferred stage costs one frame and is
  undone by the next, while a stage confirmed around a mis-seated bushing
  poisons the confirmed baseline that every later stage is judged against.
  See PositionVerdict.blocks_stage.

  "UNKNOWN" (part not visible, or not near either seat) never blocks:
  absence of evidence is not evidence of a wrong position, and a validator
  that blocked on no data would stall the whole system the moment a hand
  covered the work.

===============================================================================
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from decision.body_pose_tracker import BodyPoseTracker, PoseTrackResult
except ModuleNotFoundError:  # support: python decision\position_validator.py
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from decision.body_pose_tracker import BodyPoseTracker, PoseTrackResult


# ═══════════════════════════════════════════════════════════════════════════
# Verdict vocabulary
# ═══════════════════════════════════════════════════════════════════════════

CORRECT = "CORRECT"
WRONG = "WRONG"
UNKNOWN = "UNKNOWN"
NOT_APPLICABLE = "NOT_APPLICABLE"   # this stage has no seat requirement


# ═══════════════════════════════════════════════════════════════════════════
# Defaults - every one of these is overridable from position_config.json
# ═══════════════════════════════════════════════════════════════════════════

# How many consecutive frames a verdict must hold before it is "stable".
# ~8 frames is a third of a second at 25 fps: long enough to survive YOLO
# jitter, short enough that the operator is told immediately in human terms.
DEFAULT_STABLE_FRAMES_WRONG = 8
DEFAULT_STABLE_FRAMES_CORRECT = 8

# A detection further than this from BOTH seat anchors is not treated as
# being in a seat at all - it is a part lying loose on the bench. In pixels
# at the calibrated frame size.
DEFAULT_MAX_ANCHOR_DISTANCE_PX = 70.0

# How long a seat stays "remembered as occupied" after the last frame that
# supported it. Matches the state machine's own recent-memory reasoning: a
# hand covering the work must not flip a correct seat back to unknown.
DEFAULT_SEAT_MEMORY_SEC = 3.0

# Minimum YOLO confidence for a detection to be used for a position
# decision. Deliberately higher than the detection threshold: a 30 %-
# confidence box is good enough to count a part, not good enough to accuse
# the operator of a mistake.
DEFAULT_MIN_CONFIDENCE = 0.45


# ═══════════════════════════════════════════════════════════════════════════
# Data carriers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SeatSpec:
    """One physical seat (hole) in the assembly, marked in the camera image."""
    name: str
    anchor_norm: Tuple[float, float]     # (x, y) in 0..1 of the frame
    class_name: str = "bushing"
    label: Optional[str] = None          # human wording, e.g. "left hole"

    def anchor_px(self, frame_w: int, frame_h: int) -> Tuple[float, float]:
        return (self.anchor_norm[0] * frame_w, self.anchor_norm[1] * frame_h)

    def human(self) -> str:
        return self.label or self.name.replace("_", " ")


@dataclass
class SeatRequirement:
    """'At this stage, this seat must hold a part of this class.'"""
    seat: str
    class_name: str = "bushing"
    ordinal: Optional[int] = None        # 1st bushing, 2nd bushing - wording only

    def human(self) -> str:
        if self.ordinal == 1:
            return f"1st {self.class_name}"
        if self.ordinal == 2:
            return f"2nd {self.class_name}"
        if self.ordinal:
            return f"{self.ordinal}. {self.class_name}"
        return self.class_name


@dataclass
class SeatObservation:
    """What the validator currently believes about one seat."""
    seat: str
    occupied: bool = False
    class_name: Optional[str] = None
    confidence: float = 0.0
    distance_px: float = float("inf")
    from_memory: bool = False
    last_seen: float = 0.0


@dataclass
class PositionVerdict:
    """The complete answer for one frame. Read-only as far as callers care."""
    verdict: str = UNKNOWN               # CORRECT | WRONG | UNKNOWN | NOT_APPLICABLE
    stable: bool = False                 # has it held long enough to act on?
    stage: Optional[str] = None
    message: Optional[str] = None        # operator wording, shown on screen
    detail: Optional[str] = None         # short engineering wording, logged
    seats: Dict[str, SeatObservation] = field(default_factory=dict)
    wrong_seats: List[str] = field(default_factory=list)
    missing_seats: List[str] = field(default_factory=list)
    frames_held: int = 0
    backend: str = "geometric"
    # Agreement record, only filled when backend == "both".
    backend_votes: Dict[str, str] = field(default_factory=dict)
    # Set when the verdict is UNKNOWN *because* a required seat has never
    # been seen filled, as opposed to merely being hidden this instant.
    awaiting_evidence: bool = False
    pose_valid: bool = True
    awaiting_pose: bool = False
    pose_detail: Optional[str] = None

    @property
    def blocks_stage(self) -> bool:
        """May this verdict stop a stage from being confirmed?

        ANY wrong verdict blocks, stable or not. Stability deliberately
        does NOT gate this, and the asymmetry is the whole point:

          * Withholding a confirmation costs nothing and is undone by the
            very next frame. The stage simply has not passed YET.
          * Confirming a stage that is physically wrong is expensive: the
            confirmed baseline then treats the mis-seated bushing as
            correctly installed, every later stage is judged against that
            baseline, and only a sustained contradiction or a manual reset
            undoes it.

        So the two uses of the stability window are separated. Blocking is
        immediate, because the cheap mistake is the safe one. Telling the
        OPERATOR is what waits for stability, so a single jittery frame
        never throws a red banner across the video.

        This is also why the state machine asks the gate LAST: it has
        already satisfied itself on counts, ordering and its own temporal
        agreement, so by the time this is consulted a single frame of
        position doubt is enough reason to wait one more frame.

        ``awaiting_evidence`` blocks for a different reason: a seat this
        stage requires has never ONCE been seen filled in this session. A
        seat that has never been observed occupied is not evidence of
        correct assembly, so the stage waits rather than being granted on
        the strength of class counts alone - which is exactly the gap this
        whole module exists to close.
        """
        return self.verdict == WRONG or self.awaiting_evidence or self.awaiting_pose

    @property
    def warns_operator(self) -> bool:
        """Should the red banner and the logged warning fire?

        This is the half that DOES need the stability window - an operator
        told they are wrong on every stray frame stops believing the
        system, which is worse than being told slightly later.
        """
        return self.verdict == WRONG and self.stable

    def as_log_record(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "stable": self.stable,
            "stage": self.stage,
            "message": self.message,
            "detail": self.detail,
            "wrong_seats": list(self.wrong_seats),
            "missing_seats": list(self.missing_seats),
            "frames_held": self.frames_held,
            "backend": self.backend,
            "backend_votes": dict(self.backend_votes),
            "pose_valid": self.pose_valid,
            "awaiting_pose": self.awaiting_pose,
            "pose_detail": self.pose_detail,
            "seats": {
                name: {
                    "occupied": obs.occupied,
                    "class": obs.class_name,
                    "confidence": round(obs.confidence, 3),
                    "distance_px": (None if math.isinf(obs.distance_px)
                                    else round(obs.distance_px, 1)),
                    "from_memory": obs.from_memory,
                }
                for name, obs in self.seats.items()
            },
        }


@dataclass
class PositionWarning:
    """Duck-typed to match AssemblyWarning so ProcessLogger.record_warning
    and SessionLogger accept it with no change to either of them."""
    warning_type: str = "WRONG_POSITION"
    message: str = ""
    current_stage: Optional[str] = None
    observed: Dict[str, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    seats: Dict[str, Any] = field(default_factory=dict)
    detail: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

class PositionConfig:
    """position_config.json, with every field optional and defaulted.

    A missing or unreadable file is NOT fatal: the validator reports itself
    as unavailable and the system behaves exactly as it did before this
    module existed. That property is deliberate - a calibration file is an
    easy thing to forget, and losing position checking is far better than
    losing the monitor.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None,
                 source_path: Optional[str] = None):
        data = data or {}
        self.source_path = source_path
        self.assembly = data.get("assembly", "unknown")
        self.camera_elevation_deg = data.get("camera_elevation_deg")
        self.frame_size = tuple(data.get("frame_size", (640, 480)))
        self.backend = str(data.get("backend", "geometric")).lower()
        self.provenance = data.get("provenance") or {}

        # Body-relative tracking is enabled for a real loaded config. Unit/self-tests
        # constructed in memory keep the old static-anchor behavior unless opted in.
        pose_cfg = data.get("pose_tracking") or {}
        self.pose_tracking_enabled = bool(
            data.get("pose_tracking_enabled", pose_cfg.get("enabled", bool(source_path)))
        )
        reference_value = (
            pose_cfg.get("reference_image")
            or data.get("reference_image_path")
            or self.provenance.get("reference_image_raw")
            or self.provenance.get("reference_image")
        )
        config_dir = os.path.dirname(os.path.abspath(source_path)) if source_path else os.getcwd()
        candidates = []
        if reference_value:
            ref_path = str(reference_value)
            candidates.append(ref_path if os.path.isabs(ref_path) else os.path.join(config_dir, ref_path))
        # Prefer an unannotated frame created by the upgraded calibrator. For the
        # already-calibrated project, use the existing annotated reference image.
        candidates.extend([
            os.path.join(config_dir, "position_config_reference_raw.png"),
            os.path.join(config_dir, "position_config_reference.png"),
        ])
        self.reference_image_path = next((x for x in candidates if os.path.isfile(x)),
                                         candidates[0] if candidates else "")
        self.pose_min_inliers = int(pose_cfg.get("min_inliers", 8))
        self.pose_min_inlier_ratio = float(pose_cfg.get("min_inlier_ratio", 0.22))
        self.pose_max_reprojection_error_px = float(
            pose_cfg.get("max_reprojection_error_px", 6.0)
        )
        self.pose_feature_count = int(pose_cfg.get("feature_count", 900))

        self.max_anchor_distance_px = float(
            data.get("max_anchor_distance_px", DEFAULT_MAX_ANCHOR_DISTANCE_PX))
        self.seat_memory_sec = float(
            data.get("seat_memory_sec", DEFAULT_SEAT_MEMORY_SEC))
        self.min_confidence = float(
            data.get("min_confidence", DEFAULT_MIN_CONFIDENCE))
        self.stable_frames_wrong = int(
            data.get("stable_frames_wrong", DEFAULT_STABLE_FRAMES_WRONG))
        self.stable_frames_correct = int(
            data.get("stable_frames_correct", DEFAULT_STABLE_FRAMES_CORRECT))

        # When True, a stage whose required seat has never been observed
        # filled is held back rather than confirmed on class counts alone.
        # This is the setting that makes "Stage_01 must not confirm with
        # the bushing in the wrong hole" true in both directions: not just
        # "the wrong hole is filled" but "the right hole was never seen
        # filled". Turn it off only if the detector genuinely cannot see a
        # seated bushing, in which case the manual stage keys remain the
        # documented override.
        self.require_seat_evidence = bool(
            data.get("require_seat_evidence", True))

        # ---- seats ----
        self.seats: Dict[str, SeatSpec] = {}
        for name, spec in (data.get("seats") or {}).items():
            if isinstance(spec, dict):
                anchor = spec.get("anchor") or spec.get("anchor_norm")
                if anchor is None or len(anchor) < 2:
                    continue
                self.seats[name] = SeatSpec(
                    name=name,
                    anchor_norm=(float(anchor[0]), float(anchor[1])),
                    class_name=spec.get("class", "bushing"),
                    label=spec.get("label"),
                )
            elif isinstance(spec, (list, tuple)) and len(spec) >= 2:
                self.seats[name] = SeatSpec(
                    name=name, anchor_norm=(float(spec[0]), float(spec[1])))

        # ---- per-stage seat requirements ----
        self.stage_requirements: Dict[str, List[SeatRequirement]] = {}
        for stage, reqs in (data.get("stage_requirements") or {}).items():
            parsed: List[SeatRequirement] = []
            for i, r in enumerate(reqs or [], start=1):
                if isinstance(r, str):
                    parsed.append(SeatRequirement(seat=r, ordinal=i))
                elif isinstance(r, dict) and r.get("seat"):
                    parsed.append(SeatRequirement(
                        seat=r["seat"],
                        class_name=r.get("class", "bushing"),
                        ordinal=r.get("ordinal", i),
                    ))
            if parsed:
                self.stage_requirements[stage] = parsed

        # ---- classifier block ----
        clf = data.get("classifier") or {}
        self.classifier_model_path = clf.get("model_path") or ""
        self.classifier_input_size = int(clf.get("input_size", 96))
        self.classifier_crop_margin_px = float(clf.get("crop_margin_px", 80))
        self.classifier_min_confidence = float(clf.get("min_confidence", 0.75))
        self.classifier_correct_label = clf.get("correct_label", "correct")
        self.classifier_wrong_label = clf.get("wrong_label", "wrong_position")

        # Provenance was loaded above because it can also name the reference image.

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[str]) -> "PositionConfig":
        if not path or not os.path.exists(path):
            return cls(None, source_path=path)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return cls(json.load(fh), source_path=path)
        except Exception as exc:                                # noqa: BLE001
            print(f"[PositionValidator] Could not read {path}: {exc}")
            return cls(None, source_path=path)

    # ------------------------------------------------------------------
    @property
    def is_usable(self) -> bool:
        """Enough information present to make a position decision at all."""
        return bool(self.seats) and bool(self.stage_requirements)

    def requirements_for(self, stage: Optional[str]) -> List[SeatRequirement]:
        if not stage:
            return []
        return self.stage_requirements.get(stage, [])

    def describe(self) -> str:
        seats = ", ".join(
            f"{s.name}@({s.anchor_norm[0]:.3f},{s.anchor_norm[1]:.3f})"
            for s in self.seats.values()) or "none"
        stages = ", ".join(
            f"{k}[{'+'.join(r.seat for r in v)}]"
            for k, v in self.stage_requirements.items()) or "none"
        pose = ("body-relative tracking" if self.pose_tracking_enabled
                else "static anchors (self-test/legacy)")
        return (f"backend={self.backend} | seats: {seats} | "
                f"stages: {stages} | frame={self.frame_size} | {pose}")


# ═══════════════════════════════════════════════════════════════════════════
# Backend 1: geometric seat assignment
# ═══════════════════════════════════════════════════════════════════════════

class GeometricSeatBackend:
    """Assign detected parts to the nearest calibrated seat anchor.

    No training. Fully interpretable: every decision reduces to "this box
    centre was N pixels from Seat_1 and M pixels from Seat_2". It is also
    the only backend that can tell the operator WHICH hole is wrong, which
    is the whole point of the message on screen.
    """

    name = "geometric"

    def __init__(self, config: PositionConfig):
        self.config = config

    # ------------------------------------------------------------------
    def assign(self, detections: Sequence[Any], frame_shape: Tuple[int, int],
               classes_of_interest: Sequence[str],
               anchors_px: Optional[Dict[str, Tuple[float, float]]] = None
               ) -> Dict[str, SeatObservation]:
        """Return {seat_name: SeatObservation} for this single frame.

        Assignment is global, not greedy per detection: with two seats and
        two bushings, choosing the nearest seat for each box independently
        can put both bushings in the same seat. The permutation with the
        smallest total distance is used instead, which for two seats is
        two comparisons.
        """
        fh, fw = frame_shape[0], frame_shape[1]
        seats = list(self.config.seats.values())
        out = {s.name: SeatObservation(seat=s.name) for s in seats}
        if not seats:
            return out

        wanted = {c for c in classes_of_interest}
        cands = []
        for det in detections:
            cls = _det_class(det)
            if wanted and cls not in wanted:
                continue
            conf = _det_confidence(det)
            if conf < self.config.min_confidence:
                continue
            centre = _det_centre(det)
            if centre is None:
                continue
            cands.append((cls, conf, centre))

        if not cands:
            return out

        max_d = self.config.max_anchor_distance_px
        # Scale the gate with the frame if the live resolution differs from
        # the one the anchors were calibrated at, so a 1280-wide frame does
        # not silently become twice as strict.
        cal_w = float(self.config.frame_size[0] or fw)
        if cal_w > 0 and fw > 0:
            max_d *= (fw / cal_w)

        anchors = [
            (anchors_px[s.name] if anchors_px and s.name in anchors_px
             else s.anchor_px(fw, fh))
            for s in seats
        ]

        # Cost of every (detection, seat) pair.
        def dist(c_idx: int, s_idx: int) -> float:
            cx, cy = cands[c_idx][2]
            ax, ay = anchors[s_idx]
            return math.hypot(cx - ax, cy - ay)

        n_c, n_s = len(cands), len(seats)

        # Maximum-cardinality, minimum-cost matching between detections and
        # seats, considering only pairs within max_d.
        #
        # Partial matchings MUST be allowed. Two bushings sitting close
        # together near one seat (one installed, one still in the hand above
        # it) admits no complete two-seat assignment, and an all-or-nothing
        # search would then report BOTH seats empty - throwing away the one
        # fact the frame actually establishes. The largest feasible matching
        # is used instead, and the seats it leaves unassigned are reported
        # as unoccupied, which is what they are as far as this frame knows.
        feasible = {
            (di, si): dist(di, si)
            for di in range(n_c) for si in range(n_s)
            if dist(di, si) <= max_d
        }
        if not feasible:
            return out

        best_map: List[Tuple[int, int]] = []
        best_cost = float("inf")
        for size in range(min(n_c, n_s), 0, -1):
            for det_combo in permutations(range(n_c), size):
                for seat_combo in permutations(range(n_s), size):
                    pairs = list(zip(det_combo, seat_combo))
                    if any(p not in feasible for p in pairs):
                        continue
                    cost = sum(feasible[p] for p in pairs)
                    if cost < best_cost:
                        best_cost = cost
                        best_map = pairs
            if best_map:
                break       # largest feasible size wins; stop searching

        if not best_map:
            return out

        now = time.time()
        for di, si in best_map:
            cls, conf, centre = cands[di]
            seat = seats[si]
            obs = out[seat.name]
            obs.occupied = True
            obs.class_name = cls
            obs.confidence = conf
            obs.distance_px = dist(di, si)
            obs.from_memory = False
            obs.last_seen = now
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Backend 2: trained binary classifier
# ═══════════════════════════════════════════════════════════════════════════

class ClassifierBackend:
    """Binary correct / wrong_position model on a crop around the seats.

    Loaded lazily, and a missing model is never fatal - it reports
    unavailable and the validator falls back to the geometric backend. The
    model cannot say WHICH seat is wrong, so it only ever contributes a
    yes/no vote; the wording shown to the operator always comes from the
    geometric assignment.
    """

    name = "classifier"

    def __init__(self, config: PositionConfig):
        self.config = config
        self._model = None
        self._tried = False
        self._names: Dict[int, str] = {}

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._tried:
            return
        self._tried = True
        path = self.config.classifier_model_path
        if not path or not os.path.exists(path):
            if path:
                print(f"[PositionValidator] Classifier not found: {path}")
            return
        try:
            from ultralytics import YOLO          # local import: optional dep
            self._model = YOLO(path)
            names = getattr(self._model, "names", {}) or {}
            self._names = {int(k): str(v) for k, v in names.items()}
            print(f"[PositionValidator] Position classifier loaded: {path}")
            print(f"[PositionValidator]   classes: {sorted(self._names.values())}")
        except Exception as exc:                                # noqa: BLE001
            print(f"[PositionValidator] Classifier could not be loaded: {exc}")
            self._model = None

    # ------------------------------------------------------------------
    def seat_crop(self, frame, frame_shape: Tuple[int, int],
                  anchors_px: Optional[Dict[str, Tuple[float, float]]] = None):
        """Square crop centred on the current body-relative seat locations."""
        if frame is None or not self.config.seats:
            return None
        fh, fw = frame_shape[0], frame_shape[1]
        pts = [
            (anchors_px[s.name] if anchors_px and s.name in anchors_px
             else s.anchor_px(fw, fh))
            for s in self.config.seats.values()
        ]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        spread = 0.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                spread = max(spread, math.hypot(pts[i][0] - pts[j][0],
                                                pts[i][1] - pts[j][1]))
        half = spread / 2.0 + self.config.classifier_crop_margin_px
        x0 = int(max(0, cx - half))
        y0 = int(max(0, cy - half))
        x1 = int(min(fw, cx + half))
        y1 = int(min(fh, cy + half))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return frame[y0:y1, x0:x1]

    # ------------------------------------------------------------------
    def vote(self, frame, frame_shape: Tuple[int, int],
             anchors_px: Optional[Dict[str, Tuple[float, float]]] = None
             ) -> Tuple[str, float]:
        """Return (verdict, confidence). UNKNOWN when it cannot decide."""
        self._ensure_loaded()
        if self._model is None:
            return UNKNOWN, 0.0
        crop = self.seat_crop(frame, frame_shape, anchors_px=anchors_px)
        if crop is None or crop.size == 0:
            return UNKNOWN, 0.0
        try:
            res = self._model.predict(crop, verbose=False)
        except Exception as exc:                                # noqa: BLE001
            print(f"[PositionValidator] Classifier inference failed: {exc}")
            return UNKNOWN, 0.0
        if not res:
            return UNKNOWN, 0.0

        probs = getattr(res[0], "probs", None)
        if probs is None:
            return UNKNOWN, 0.0
        try:
            top = int(probs.top1)
            conf = float(probs.top1conf)
        except Exception:                                       # noqa: BLE001
            return UNKNOWN, 0.0

        label = self._names.get(top, str(top))
        if conf < self.config.classifier_min_confidence:
            return UNKNOWN, conf
        if label == self.config.classifier_correct_label:
            return CORRECT, conf
        if label == self.config.classifier_wrong_label:
            return WRONG, conf
        return UNKNOWN, conf


# ═══════════════════════════════════════════════════════════════════════════
# The validator
# ═══════════════════════════════════════════════════════════════════════════

class PositionValidator:
    """Answers 'is the part in the required seat' and nothing else.

    Typical use, once per frame, after YOLO and BEFORE the state machine:

        verdict = position_validator.update(
            stage_name = stage_machine.target_stage_name,
            detections = object_result.objects,
            frame      = frame,
        )
        stage_machine.update(observed_classes)      # gate is read internally

    The state machine is given ``validator.gate`` at construction time and
    calls it only at the moment it is about to confirm a stage.
    """

    def __init__(self, config: PositionConfig,
                 on_warning=None,
                 enabled: bool = True):
        self.config = config
        self.enabled = bool(enabled) and config.is_usable
        self._on_warning = on_warning

        self._geometric = GeometricSeatBackend(config)
        self._classifier = ClassifierBackend(config)
        self._pose_tracker = None
        self._pose_result: Optional[PoseTrackResult] = None
        self._current_anchors_px: Optional[Dict[str, Tuple[float, float]]] = None
        if self.enabled and config.pose_tracking_enabled:
            reference_anchors = {
                seat.name: seat.anchor_px(int(config.frame_size[0]), int(config.frame_size[1]))
                for seat in config.seats.values()
            }
            # Prefer exact clicked coordinates over rounded normalized values.
            try:
                with open(config.source_path, "r", encoding="utf-8") as fh:
                    raw_config = json.load(fh)
                for name, spec in (raw_config.get("seats") or {}).items():
                    px = spec.get("anchor_px_at_calibration") if isinstance(spec, dict) else None
                    if px and len(px) >= 2 and name in reference_anchors:
                        reference_anchors[name] = (float(px[0]), float(px[1]))
            except Exception:
                pass
            self._pose_tracker = BodyPoseTracker(
                config.reference_image_path, reference_anchors,
                min_inliers=config.pose_min_inliers,
                min_inlier_ratio=config.pose_min_inlier_ratio,
                max_reprojection_error_px=config.pose_max_reprojection_error_px,
                feature_count=config.pose_feature_count,
            )
            if self._pose_tracker._initialization_error:
                print("[PositionValidator] body tracker will fail safe: "
                      + self._pose_tracker._initialization_error)

        # Sticky per-seat memory: seat -> SeatObservation. Short-lived,
        # used to keep the on-screen occupancy steady while hands move.
        self._memory: Dict[str, SeatObservation] = {}

        # Per-seat LATCH: a seat that has been observed filled, steadily,
        # at some point in this session. Never expires, because the part it
        # records is pressed in and will not be seen again - the state
        # machine keeps a confirmed component baseline for the same reason.
        # Cleared only by reset(), so a manual stage jump cannot inherit
        # evidence from before it.
        self._latched: Dict[str, float] = {}
        self._seat_streak: Dict[str, int] = {}

        # Temporal stability counters
        self._last_raw_verdict = UNKNOWN
        self._frames_held = 0

        # The latest verdict, exposed for the display and consulted by the
        # gate. Kept as the single source of truth so the screen, the log
        # and the stage decision can never disagree with each other.
        self.last_verdict = PositionVerdict(verdict=UNKNOWN)

        # Warnings are raised once per episode, not once per frame.
        self._warning_active_for: Optional[str] = None

        # Counters for the validation report.
        self.stats = {
            "frames": 0,
            "wrong_episodes": 0,
            "correct_confirmations": 0,
            "blocked_confirmations": 0,
        }

        if self.enabled:
            print("[PositionValidator] enabled | " + config.describe())
        else:
            reason = ("disabled in settings" if not enabled
                      else "no usable position_config.json "
                           "(run tools/calibrate_position_anchors.py)")
            print(f"[PositionValidator] NOT active - {reason}. "
                  "Stage confirmation is unchanged.")

    # ==================================================================
    # Per-frame entry point
    # ==================================================================

    def update(self, stage_name: Optional[str], detections: Sequence[Any],
               frame=None) -> PositionVerdict:
        """Evaluate one camera frame using body-relative seat coordinates.

        Live position checking never falls back to fixed screen anchors when
        body_a tracking fails. It hides the markers and pauses Stage_01/02
        confirmation until the body can be tracked again.
        """
        if not self.enabled:
            self.last_verdict = PositionVerdict(
                verdict=NOT_APPLICABLE, stage=stage_name,
                backend=self.config.backend)
            return self.last_verdict

        self.stats["frames"] += 1
        frame_shape = _frame_shape(frame, self.config.frame_size)
        reqs = self.config.requirements_for(stage_name)

        # Update the pose tracker on every frame, including stages with no
        # seat requirement, so markers always follow the body when visible.
        pose_valid = True
        pose_detail = "static anchors"
        anchors_px = None
        if self.config.pose_tracking_enabled:
            if self._pose_tracker is None:
                pose_valid = False
                pose_detail = "body pose tracker is unavailable"
            else:
                self._pose_result = self._pose_tracker.update(frame, detections or [])
                pose_valid = bool(self._pose_result.valid)
                pose_detail = self._pose_result.reason
                anchors_px = self._pose_result.anchors_px if pose_valid else None
            self._current_anchors_px = anchors_px
        else:
            fh, fw = frame_shape
            self._current_anchors_px = {
                seat.name: seat.anchor_px(fw, fh) for seat in self.config.seats.values()
            }
            anchors_px = self._current_anchors_px

        if not reqs:
            self._reset_stability()
            self.last_verdict = PositionVerdict(
                verdict=NOT_APPLICABLE, stage=stage_name,
                backend=self.config.backend,
                detail=(f"body tracking: {pose_detail}" if not pose_valid
                        else f"{stage_name}: no seat requirement"),
                pose_valid=pose_valid, pose_detail=pose_detail)
            return self.last_verdict

        if self.config.pose_tracking_enabled and not pose_valid:
            self._reset_stability()
            waiting_for = ", ".join(self._seat_name(r.seat) for r in reqs)
            message = (
                "BODY NOT TRACKED — position check paused. Keep body_a visible; "
                "return it briefly to its calibrated view if tracking does not recover."
            )
            self.last_verdict = PositionVerdict(
                verdict=UNKNOWN, stage=stage_name, message=message,
                detail=f"body_a pose not tracked: {pose_detail}",
                missing_seats=[r.seat for r in reqs], backend=self.config.backend,
                awaiting_pose=True, pose_valid=False, pose_detail=pose_detail)
            return self.last_verdict

        classes = {r.class_name for r in reqs}

        # ---- 1. where are the parts relative to the CURRENT body pose ----
        live_seats = self._geometric.assign(
            detections or [], frame_shape, classes, anchors_px=anchors_px)
        live_counts = self._live_counts(detections or [], classes)
        self._update_latches(live_seats, reqs)
        seats = self._apply_memory(dict(live_seats), live_counts)

        # ---- 2. what does that mean for THIS stage ----
        raw, wrong_seats, missing, detail, message = self._judge(
            reqs, seats, live_seats)

        # ---- 3. optional second opinion from the trained model ----
        votes: Dict[str, str] = {"geometric": raw}
        if self.config.backend in ("classifier", "both"):
            clf_verdict, clf_conf = self._classifier.vote(
                frame, frame_shape, anchors_px=anchors_px)
            votes["classifier"] = clf_verdict
            if self.config.backend == "classifier":
                raw = clf_verdict if clf_verdict != UNKNOWN else raw
            else:
                if raw == WRONG and clf_verdict != WRONG:
                    raw = UNKNOWN
                    detail = (f"geometric says wrong, classifier says "
                              f"{clf_verdict.lower()} ({clf_conf:.2f}) - not acted on")
                    message = None

        # ---- 4. temporal stability ----
        if raw == self._last_raw_verdict:
            self._frames_held += 1
        else:
            self._last_raw_verdict = raw
            self._frames_held = 1
        need = (self.config.stable_frames_wrong if raw == WRONG
                else self.config.stable_frames_correct)
        stable = raw in (CORRECT, WRONG) and self._frames_held >= need

        verdict = PositionVerdict(
            verdict=raw, stable=stable, stage=stage_name, message=message,
            detail=detail, seats=seats, wrong_seats=wrong_seats,
            missing_seats=missing, frames_held=self._frames_held,
            backend=self.config.backend, backend_votes=votes,
            awaiting_evidence=(raw == UNKNOWN and bool(missing)
                               and self.config.require_seat_evidence),
            pose_valid=pose_valid, pose_detail=pose_detail)
        self.last_verdict = verdict
        self._maybe_warn(verdict, detections)
        return verdict

    # ==================================================================
    # The gate the state machine calls
    # ==================================================================

    def gate(self, stage_name: Optional[str]):
        """May ``stage_name`` be confirmed, as far as POSITION is concerned?

        Returns (allow, detail) or (allow, detail, status). This is the
        ONLY thing the state machine
        asks of this module, and it is advisory on exactly one axis: seat
        position. The state machine still has to be satisfied on counts,
        order and temporal agreement before it even asks.

        allow is False for ANY wrong verdict about THIS stage, including one
        that has not yet held long enough to be reported to the operator -
        see PositionVerdict.blocks_stage for why blocking is immediate while
        warning waits.

        An unknown position never blocks: absence of evidence is not
        evidence of a wrong position, and a check that stalled the monitor
        whenever a hand covered the seats would be worse than no check.

        Known trade-off, stated rather than hidden: a loose part left lying
        within the seat gate of a seat this stage does not need reads as a
        wrong position and will hold the stage back. The seat rings drawn
        on the video make that visible immediately, and the gate radius is
        derived from the measured seat separation to keep it tight.
        """
        if not self.enabled:
            return True, None
        if not self.config.requirements_for(stage_name):
            return True, None

        v = self.last_verdict
        if v.stage != stage_name:
            # The verdict on screen is about a different stage; no opinion.
            return True, None
        if v.blocks_stage:
            self.stats["blocked_confirmations"] += 1
            if v.awaiting_pose:
                return (False, v.detail or "body_a pose is not tracked", "WAITING_POSITION")
            if v.verdict == WRONG:
                # A part IS in a hole it should not be in. That is a
                # mistake, and the operator is told so.
                return (False,
                        v.detail or "part in the wrong seat",
                        "WRONG_POSITION")
            # Nothing is wrong yet - the system has simply not seen the
            # seat filled. Reporting that as WRONG POSITION would accuse
            # the operator of a mistake they have not made, and an operator
            # told they are wrong when they are not stops reading the
            # screen at all.
            return (False,
                    v.detail or "the required seat has not been seen filled",
                    "WAITING_POSITION")
        if v.verdict == CORRECT and v.stable:
            self.stats["correct_confirmations"] += 1
        return True, None

    # ==================================================================
    # Display helpers
    # ==================================================================

    def seat_markers(self, frame_shape) -> List[Dict[str, Any]]:
        """Current body-relative seat markers. Never draw stale screen anchors."""
        if not self.enabled:
            return []
        v = self.last_verdict
        if self.config.pose_tracking_enabled and (not v.pose_valid or not self._current_anchors_px):
            detail = v.pose_detail or v.detail or "body_a not tracked"
            return [{
                "pose_valid": False,
                "message": "BODY NOT TRACKED — seat markers hidden; position check paused.",
                "detail": detail,
            }]

        fh, fw = _frame_shape(None, self.config.frame_size) \
            if frame_shape is None else (frame_shape[0], frame_shape[1])
        out = []
        for seat in self.config.seats.values():
            if self._current_anchors_px and seat.name in self._current_anchors_px:
                x, y = self._current_anchors_px[seat.name]
            else:
                x, y = seat.anchor_px(fw, fh)
            obs = v.seats.get(seat.name)
            required = any(r.seat == seat.name for r in self.config.requirements_for(v.stage))
            out.append({
                "pose_valid": True,
                "name": seat.name,
                "label": seat.human(),
                "xy": (int(round(x)), int(round(y))),
                "occupied": bool(obs and obs.occupied) or seat.name in self._latched,
                "required": required,
                "wrong": seat.name in v.wrong_seats,
                "latched": seat.name in self._latched,
            })
        return out

    @property
    def status_text(self) -> Optional[str]:
        """One short line for the sidebar, or None when there is nothing
        worth saying."""
        if not self.enabled:
            return None
        v = self.last_verdict
        if not v.pose_valid and self.config.pose_tracking_enabled:
            return "tracking body_a"
        if v.verdict == WRONG:
            return "WRONG POSITION" if v.stable else "checking position..."
        if v.verdict == CORRECT:
            return "position OK" if v.stable else "checking position..."
        if v.verdict == NOT_APPLICABLE:
            return None
        if v.awaiting_evidence:
            return "waiting for the seat"
        return "position unknown"

    # ==================================================================
    # Internals
    # ==================================================================

    def _judge(self, reqs: List[SeatRequirement],
               seats: Dict[str, SeatObservation],
               live_seats: Dict[str, SeatObservation]):
        """Turn seat occupancy into a verdict for one stage.

        The rules, in order:
          * a part visible NOW in a seat this stage does not require
            -> WRONG
          * every required seat latched (seen filled at least once)
            -> CORRECT
          * a required seat never yet seen filled
            -> UNKNOWN, awaiting evidence (blocks the stage)
          * anything else -> UNKNOWN (does not block)

        Two different notions of occupancy are used on purpose, and mixing
        them up was a real bug worth naming:

        LIVE occupancy decides WRONG. A part is only in the wrong hole if
        it is in the wrong hole right now. Judging that from remembered
        occupancy means the system goes on reporting the mistake for
        seconds after the operator has fixed it, so the correction appears
        to do nothing.

        LATCHED occupancy decides CORRECT. A bushing pressed into body_a
        is hidden permanently and is never detected again - the state
        machine's own confirmed baseline exists for the same reason. So a
        seat that has been observed filled stays accounted for, for the
        rest of the session, and a required seat only has to be seen once,
        during the stage that installs it.
        """
        required = {r.seat: r for r in reqs}
        forbidden = [name for name in self.config.seats if name not in required]

        # WRONG is judged on what is visible now, never on memory.
        wrong_seats = [name for name in forbidden
                       if live_seats.get(name) and live_seats[name].occupied]
        # CORRECT is judged on what has ever been established.
        missing = [name for name in required if name not in self._latched]

        if wrong_seats:
            # Name the part by its ordinal in this stage, which is how the
            # operator thinks about it ("the 1st bushing"), and name the
            # seat it should be in.
            req = reqs[0]
            target_seat = self.config.seats.get(req.seat)
            wrong_seat = self.config.seats.get(wrong_seats[0])
            part = req.human()
            message = (
                f"{part.capitalize()} is in the wrong hole. "
                f"Move it to the correct hole shown in the "
                f"{(self.last_verdict.stage or 'current stage').replace('_', ' ')} "
                f"preview."
            )
            detail = (
                f"{part} found in {wrong_seat.human() if wrong_seat else wrong_seats[0]}, "
                f"required in {target_seat.human() if target_seat else req.seat}"
            )
            return WRONG, wrong_seats, missing, detail, message

        if not missing:
            names = ", ".join(
                f"{r.human()} in {self._seat_name(r.seat)}" for r in reqs)
            return CORRECT, [], [], f"seated correctly: {names}", None

        # A required seat has never been seen filled. Not an accusation -
        # the system simply has not yet observed the thing it is being
        # asked to certify - but not something to confirm a stage on
        # either, so this blocks and says plainly what it is waiting for.
        waiting = ", ".join(
            f"{required[s].human()} in the {self._seat_name(s)}"
            for s in missing)
        return (UNKNOWN, [], missing,
                f"waiting to see {waiting}", None)

    def _seat_name(self, seat: str) -> str:
        spec = self.config.seats.get(seat)
        return spec.human() if spec else seat

    # ------------------------------------------------------------------
    def _update_latches(self, live_seats: Dict[str, SeatObservation],
                        reqs: List[SeatRequirement]) -> None:
        """Latch a seat once it has been LIVE-occupied for enough frames.

        A streak is required rather than a single frame because one noisy
        box near a seat would otherwise permanently certify that seat for
        the rest of the session - and a latch is deliberately hard to
        undo, so the bar to setting one has to be correspondingly high.

        Only seats relevant to the current stage are latched. A bushing
        that happens to drift past a seat the stage does not care about
        must not quietly pre-certify a later stage.
        """
        relevant = {r.seat for r in reqs}
        for name, obs in live_seats.items():
            if name not in relevant:
                continue
            if obs.occupied:
                self._seat_streak[name] = self._seat_streak.get(name, 0) + 1
                if (name not in self._latched
                        and self._seat_streak[name]
                        >= self.config.stable_frames_correct):
                    self._latched[name] = time.time()
                    print(f"[POSITION] {self._seat_name(name)} confirmed "
                          f"occupied - established for this session")
            else:
                self._seat_streak[name] = 0

    # ------------------------------------------------------------------
    def _live_counts(self, detections: Sequence[Any],
                     classes: set) -> Dict[str, int]:
        """How many parts of each relevant class are visible RIGHT NOW,
        at a confidence high enough to reason about position."""
        counts: Dict[str, int] = {}
        for det in detections:
            cls = _det_class(det)
            if cls not in classes:
                continue
            if _det_confidence(det) < self.config.min_confidence:
                continue
            counts[cls] = counts.get(cls, 0) + 1
        return counts

    def _apply_memory(self, seats: Dict[str, SeatObservation],
                      live_counts: Dict[str, int]
                      ) -> Dict[str, SeatObservation]:
        """Keep a seat 'occupied' for a short while after it was last seen.

        WHY MEMORY IS NEEDED
          The operator's hand covers the seats for most of the working
          time. Without memory the verdict oscillates between CORRECT and
          UNKNOWN several times a second and never settles, which would
          make the check useless in practice rather than merely wrong in
          theory.

        WHY IT MUST BE CAPPED BY WHAT IS VISIBLE
          Memory alone remembers a MISTAKE just as faithfully as a correct
          seating. When the operator moves a bushing out of the wrong hole
          into the right one, the wrong hole is still "remembered occupied"
          for the length of the memory window - so the system goes on
          reporting the wrong position for seconds after it was fixed, and
          the correction appears to do nothing. That is the one failure
          mode an operator will not forgive.

          The cap closes it: memory may never claim more parts of a class
          than the frame can account for. One bushing visible, already
          assigned to a seat, leaves nothing to put in any other seat, so
          the stale seat is dropped immediately and the fix registers on
          the next frame. When the class is not visible at all - genuine
          occlusion, the case memory exists for - the cap does not bind and
          every remembered seat is kept.
        """
        now = time.time()
        window = self.config.seat_memory_sec

        merged: Dict[str, SeatObservation] = {}
        assigned_by_class: Dict[str, int] = {}

        # Live assignments always win and refresh the memory.
        for name, obs in seats.items():
            if obs.occupied:
                self._memory[name] = obs
                merged[name] = obs
                cls = obs.class_name or ""
                assigned_by_class[cls] = assigned_by_class.get(cls, 0) + 1

        # Remaining memory budget per class.
        budget: Dict[str, int] = {}
        for cls, live in live_counts.items():
            budget[cls] = max(0, live - assigned_by_class.get(cls, 0))

        # Candidate memory fills, most recently seen first, so that when the
        # budget is tight the freshest evidence is the evidence kept.
        candidates = sorted(
            ((name, self._memory.get(name)) for name in seats
             if name not in merged),
            key=lambda kv: (kv[1].last_seen if kv[1] else 0.0),
            reverse=True,
        )

        for name, remembered in candidates:
            if remembered is None or (now - remembered.last_seen) > window:
                self._memory.pop(name, None)
                merged[name] = seats[name]
                continue

            cls = remembered.class_name or ""
            if cls in live_counts:
                # The class IS visible, so the frame can account for its
                # parts; memory may only fill what is left over.
                if budget.get(cls, 0) <= 0:
                    self._memory.pop(name, None)
                    merged[name] = seats[name]
                    continue
                budget[cls] -= 1

            merged[name] = SeatObservation(
                seat=name,
                occupied=True,
                class_name=remembered.class_name,
                confidence=remembered.confidence,
                distance_px=remembered.distance_px,
                from_memory=True,
                last_seen=remembered.last_seen,
            )

        # Anything neither live nor remembered stays as the raw observation.
        for name, obs in seats.items():
            merged.setdefault(name, obs)
        return merged

    def _reset_stability(self) -> None:
        self._last_raw_verdict = UNKNOWN
        self._frames_held = 0

    def _maybe_warn(self, verdict: PositionVerdict,
                    detections: Sequence[Any]) -> None:
        """Fire on_warning once when a wrong position becomes stable, and
        again only after the situation has cleared and recurred."""
        key = None
        if verdict.warns_operator:
            key = f"{verdict.stage}:{','.join(verdict.wrong_seats)}"

        if key and key != self._warning_active_for:
            self._warning_active_for = key
            self.stats["wrong_episodes"] += 1
            counts: Dict[str, int] = {}
            for det in detections or []:
                cls = _det_class(det)
                if cls:
                    counts[cls] = counts.get(cls, 0) + 1
            warning = PositionWarning(
                warning_type="WRONG_POSITION",
                message=verdict.message or "Part in the wrong position.",
                current_stage=verdict.stage,
                observed=counts,
                timestamp=time.time(),
                seats=verdict.as_log_record()["seats"],
                detail=verdict.detail,
            )
            if self._on_warning:
                self._on_warning(warning)
            print(f"[POSITION] WRONG: {verdict.detail}")

        elif not key and self._warning_active_for and verdict.verdict == CORRECT:
            print("[POSITION] corrected - position now OK")
            self._warning_active_for = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all position evidence. Called when the state machine is
        reset or manually jumped, so observations from before the jump can
        never satisfy a check after it."""
        self._memory.clear()
        self._latched.clear()
        self._seat_streak.clear()
        self._reset_stability()
        self._warning_active_for = None
        # Keep the visual track when only resetting assembly progress. The
        # body may already be in a new pose; resetting the tracker here would
        # force a return to the calibration view for an unrelated stage reset.
        self._current_anchors_px = None
        self._pose_result = None
        self.last_verdict = PositionVerdict(verdict=UNKNOWN, pose_valid=False,
                                            awaiting_pose=self.config.pose_tracking_enabled)

    @property
    def latched_seats(self) -> Dict[str, float]:
        """Seats established as occupied in this session, for the log."""
        return dict(self._latched)


# ═══════════════════════════════════════════════════════════════════════════
# Detection-shape helpers
# ═══════════════════════════════════════════════════════════════════════════
# ObjectDetector yields objects with .class_name, .confidence and
# .bounding_box = (x1, y1, x2, y2). These helpers also accept dicts and
# plain tuples so the validator can be unit-tested without the detector,
# and so a future detector change does not break this module.

def _det_class(det: Any) -> Optional[str]:
    for attr in ("class_name", "cls_name", "name", "label"):
        val = getattr(det, attr, None)
        if isinstance(val, str):
            return val
    if isinstance(det, dict):
        for k in ("class_name", "class", "name", "label"):
            if isinstance(det.get(k), str):
                return det[k]
    return None


def _det_confidence(det: Any) -> float:
    for attr in ("confidence", "conf", "score"):
        val = getattr(det, attr, None)
        if isinstance(val, (int, float)):
            return float(val)
    if isinstance(det, dict):
        for k in ("confidence", "conf", "score"):
            if isinstance(det.get(k), (int, float)):
                return float(det[k])
    return 1.0          # no confidence reported: do not discard the box


def _det_centre(det: Any) -> Optional[Tuple[float, float]]:
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
        cx = getattr(det, "cx", None)
        cy = getattr(det, "cy", None)
        if isinstance(cx, (int, float)) and isinstance(cy, (int, float)):
            return float(cx), float(cy)
        return None
    try:
        vals = [float(v) for v in list(box)[:4]]
    except Exception:                                           # noqa: BLE001
        return None
    if len(vals) < 4:
        return None
    x1, y1, x2, y2 = vals
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _frame_shape(frame, fallback: Tuple[int, int]) -> Tuple[int, int]:
    """Return (height, width). fallback is (width, height) from the config."""
    if frame is not None:
        try:
            return int(frame.shape[0]), int(frame.shape[1])
        except Exception:                                       # noqa: BLE001
            pass
    return int(fallback[1]), int(fallback[0])


# ═══════════════════════════════════════════════════════════════════════════
# Factory used by main.py
# ═══════════════════════════════════════════════════════════════════════════

def create_position_validator(config_path: Optional[str],
                              enabled: bool = True,
                              on_warning=None) -> PositionValidator:
    """Build a validator from a config path. Never raises."""
    try:
        config = PositionConfig.load(config_path)
    except Exception as exc:                                    # noqa: BLE001
        print(f"[PositionValidator] config error: {exc}")
        config = PositionConfig(None, source_path=config_path)
    return PositionValidator(config, on_warning=on_warning, enabled=enabled)


# ═══════════════════════════════════════════════════════════════════════════
# Self-test: run this file directly to check the logic with no camera
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo_cfg = PositionConfig({
        "assembly": "subassembly",
        "camera_elevation_deg": 35,
        "frame_size": [640, 480],
        "backend": "geometric",
        "stable_frames_wrong": 3,
        "stable_frames_correct": 3,
        "seats": {
            "Seat_1": {"anchor": [0.45, 0.60], "class": "bushing",
                       "label": "left hole"},
            "Seat_2": {"anchor": [0.60, 0.64], "class": "bushing",
                       "label": "right hole"},
        },
        "stage_requirements": {
            "Stage_01": [{"seat": "Seat_1", "class": "bushing", "ordinal": 1}],
            "Stage_02": [{"seat": "Seat_1", "class": "bushing", "ordinal": 1},
                         {"seat": "Seat_2", "class": "bushing", "ordinal": 2}],
        },
    })

    class Det:
        def __init__(self, name, cx, cy, conf=0.9):
            self.class_name = name
            self.confidence = conf
            self.bounding_box = (cx - 10, cy - 10, cx + 10, cy + 10)

    v = PositionValidator(demo_cfg)

    def run(label, stage, dets, frames=4):
        for _ in range(frames):
            res = v.update(stage, dets, frame=None)
        print(f"{label:38s} -> {res.verdict:8s} stable={res.stable} "
              f"allow={v.gate(stage)[0]}")
        if res.detail:
            print(f"{'':38s}    {res.detail}")
        if res.message:
            print(f"{'':38s}    screen: {res.message}")

    print("\n--- Stage_01, bushing in the correct hole -------------------")
    run("body_a + bushing @ Seat_1", "Stage_01",
        [Det("body_a", 320, 300), Det("bushing", 288, 288)])

    print("\n--- Stage_01, bushing in the WRONG hole --------------------")
    v.reset()
    run("body_a + bushing @ Seat_2", "Stage_01",
        [Det("body_a", 320, 300), Det("bushing", 384, 307)])

    print("\n--- Stage_01, bushing loose on the bench -------------------")
    v.reset()
    run("bushing far from both seats", "Stage_01",
        [Det("body_a", 320, 300), Det("bushing", 80, 80)])

    print("\n--- Stage_02, both bushings seated correctly ---------------")
    v.reset()
    run("two bushings, one per seat", "Stage_02",
        [Det("body_a", 320, 300), Det("bushing", 288, 288),
         Det("bushing", 384, 307)])

    print("\n--- Stage_02, second bushing in the first hole -------------")
    v.reset()
    run("both bushings near Seat_1", "Stage_02",
        [Det("body_a", 320, 300), Det("bushing", 284, 286),
         Det("bushing", 292, 290)])

    print("\n--- Stage_04, no seat requirement for this stage -----------")
    v.reset()
    run("anything", "Stage_04", [Det("nut", 320, 300)])

    print("\nSelf-test complete. Expected results:")
    print("  correct hole          CORRECT          allow=True")
    print("  wrong hole            WRONG            allow=False  (a mistake)")
    print("  loose on the bench    UNKNOWN          allow=False  (not yet seen"
          " in the seat)")
    print("  both seats filled     CORRECT          allow=True")
    print("  2nd seat never filled UNKNOWN          allow=False")
    print("  no seat requirement   NOT_APPLICABLE   allow=True")
    print("\nNote: 'stable' can read False here simply because each case runs"
          "\nonly a few frames. Stability governs the OPERATOR WARNING, never"
          "\nwhether a stage is allowed to confirm - see"
          " PositionVerdict.blocks_stage.\n")
