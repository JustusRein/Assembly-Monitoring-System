"""
decision/assembly_state_machine.py
================================
SUBSYSTEM: Decision (SS5) — new file, sibling to event_detector.py.

ARCHITECTURE RULE (agreed): YOLO is a replaceable Processing-layer
component. This module never touches YOLO or ObjectDetector directly -
it only consumes a list of class-name strings (one entry per detected
box, duplicates included), so swapping the detector later (YOLO26 ->
YOLO27, a different architecture, anything) requires zero changes here.

event_detector.py already answers "is something moving" (IDLE/ACTIVE).
This module answers a different question: "does what YOLO currently
sees match the correct next step of the assembly, and if not, what's
wrong". It does not replace EventDetector - both run side by side,
exactly as agreed (motion detector stays as a secondary signal).

DEBOUNCE METHOD - majority vote, not consecutive streak:
  Matches scripts/build_detection_timeline.py's already-proven
  majority_filter() pattern (WINDOW_SIZE=5, MINIMUM_POSITIVE_FRAMES=3)
  instead of requiring N-in-a-row. A stage/warning signature is
  confirmed once it's the majority reading within the last WINDOW_SIZE
  frames, tolerating an occasional bad frame inside that window rather
  than resetting on the first miss.

WHY STRICT (exact count) MATCHING:
  A stage is only "reached" when detected part counts match that
  stage's expected components EXACTLY (Stage_02 needs body_a:1 AND
  bushing:2 - not just "a bushing is visible", which is indistinguishable
  from Stage_01's body_a:1 + bushing:1). Deliberately strict and
  deterministic, per the project's decision to use a rule-based state
  machine instead of a learned temporal model. If real-camera testing
  shows too many "no match yet" frames from occlusion, loosen
  _classify() first before adding anything more complex.

OUTPUT FORMAT - reuses AssemblyEvent, not a new logger:
  StageChange and AssemblyWarning below are converted to the existing
  AssemblyEvent dataclass (decision/event_detector.py) before being
  logged, so storage/session_logger.py needs ZERO changes - it already
  knows how to write an AssemblyEvent to CSV. duration_sec is 0.0 for
  these since they're point-in-time events, not durations; this is a
  deliberate reuse of the existing schema rather than a perfect
  semantic fit - noted here rather than hidden.

SOURCE OF TRUTH FOR THE SEQUENCE:
  Loaded from assembly_sequence.json, written by the Blender dataset
  generator, so training data and live state machine can't drift out
  of sync. Falls back to a hardcoded copy if that file isn't present
  on this machine.
"""

import json
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from decision.event_detector import AssemblyEvent


# Fallback sequence - must match the Blender generator's
# STAGE_EXPECTED_SEQUENCE exactly. Only used if assembly_sequence.json
# is not found on disk.
_FALLBACK_STAGE_SEQUENCE = [
    "Stage_01", "Stage_02", "Stage_03", "Stage_04",
    "Stage_05", "Stage_06", "Stage_07",
]
_FALLBACK_EXPECTED_COMPONENTS = {
    "Stage_01": {"body_a": 1, "bushing": 1},
    "Stage_02": {"body_a": 1, "bushing": 2},
    "Stage_03": {"body_a": 1, "body_b": 1, "bushing": 2},
    "Stage_04": {"body_a": 1, "body_b": 1, "bushing": 2, "block_pin": 1},
    "Stage_05": {"body_a": 1, "body_b": 1, "bushing": 2, "block_pin": 1, "nut": 1},
    "Stage_06": {"knob": 1, "threaded_pin": 1},
    "Stage_07": {
        "body_a": 1, "body_b": 1, "bushing": 2, "block_pin": 1,
        "nut": 1, "knob": 1, "threaded_pin": 1,
    },
}

# Matches scripts/build_detection_timeline.py exactly.
WINDOW_SIZE = 5
MINIMUM_POSITIVE_FRAMES = 3

# How many total component units can be missing (occluded) from the
# target set and still count as a match. 1 means "one bushing hidden
# behind a hand while placing the next part" is tolerated; the isolated
# "holding one part up alone" case is still correctly rejected (that's
# missing 2-3 units, not 1) - see _matches() for the full reasoning.
MAX_MISSING_TOLERANCE = 1

# How many recent frames a class stays "trusted" after last being seen
# at its required count, even if the CURRENT frame doesn't show it at
# all. This is what lets an already-confirmed component (e.g. body_a,
# placed several seconds ago) stay "present" in the eyes of the matcher
# while your hand fully occludes it during a later step - without this,
# MAX_MISSING_TOLERANCE alone isn't enough once an older part is fully
# hidden (not just partially), which is the real scenario from the
# "inserting block_pin" screenshot. Tune based on your actual observed
# frame rate - this is expressed in frames, not seconds, because fps
# depends on whether inference is running on GPU or CPU (see the
# earlier CPU/GPU question - resolve that first if this still feels
# too short or too long once tuned).
RECENT_MEMORY_FRAMES = 60


@dataclass
class AssemblyWarning:
    """One detected problem in the assembly sequence."""
    warning_type: str    # "WRONG_ORDER" | "WRONG_PART"
    message:      str
    timestamp:    float
    current_stage: str
    observed:     Dict[str, int]


@dataclass
class StageChange:
    """One confirmed, debounced stage transition."""
    from_stage:   Optional[str]
    to_stage:     str
    timestamp:    float


def _load_sequence(json_path: Optional[str]):
    """Load the stage sequence + expected components from the Blender
    generator's assembly_sequence.json, falling back to the hardcoded
    copy if the file isn't available on this machine."""
    if json_path:
        path = Path(json_path)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sequence = data["sequence"]
                expected = data["expected_components"]
                print(f"[AssemblyStateMachine] Loaded sequence from: {path}")
                return sequence, expected
            except Exception as exc:
                print(f"[AssemblyStateMachine] WARNING: could not parse "
                      f"{path} ({exc}). Using built-in fallback sequence.")
        else:
            print(f"[AssemblyStateMachine] WARNING: {path} not found. "
                  f"Using built-in fallback sequence.")
    return _FALLBACK_STAGE_SEQUENCE, _FALLBACK_EXPECTED_COMPONENTS


def stage_change_to_event(change: StageChange) -> AssemblyEvent:
    """Convert to the existing AssemblyEvent so SessionLogger.log_event()
    works unchanged."""
    return AssemblyEvent(
        event_type=f"STAGE_CHANGE ({change.from_stage} -> {change.to_stage})",
        start_time=change.timestamp,
        end_time=change.timestamp,
        duration_sec=0.0,
        detected_object=change.to_stage,
    )


def warning_to_event(warning: AssemblyWarning) -> AssemblyEvent:
    """Convert to the existing AssemblyEvent so SessionLogger.log_event()
    works unchanged."""
    observed_str = ",".join(f"{k}:{v}" for k, v in sorted(warning.observed.items()))
    return AssemblyEvent(
        event_type=f"WARNING ({warning.warning_type})",
        start_time=warning.timestamp,
        end_time=warning.timestamp,
        duration_sec=0.0,
        detected_object=observed_str,
    )


class AssemblyStateMachine:
    """
    State machine that converts per-frame YOLO detections into confirmed
    assembly-stage transitions and wrong-part/wrong-order warnings.

    Maps to an extension of the Decision subsystem (SS5). Runs alongside
    EventDetector, not instead of it.
    """

    def __init__(self, on_stage_change=None, on_warning=None,
                 sequence_json_path: str = None,
                 window_size: int = WINDOW_SIZE,
                 minimum_positive_frames: int = MINIMUM_POSITIVE_FRAMES,
                 max_missing_tolerance: int = MAX_MISSING_TOLERANCE,
                 recent_memory_frames: int = RECENT_MEMORY_FRAMES):
        """
        Args:
            on_stage_change: callback(StageChange), called whenever a
                             stage transition is confirmed.
            on_warning:      callback(AssemblyWarning), called whenever a
                             wrong-part or wrong-order condition is
                             confirmed.
            sequence_json_path: path to assembly_sequence.json.
            window_size:     frames considered for the majority vote.
            minimum_positive_frames: how many of those frames must agree
                             before a signature is confirmed.
            max_missing_tolerance: how many total component units can be
                             occluded/missing from a target and still
                             count as a match (see MAX_MISSING_TOLERANCE
                             and _matches()).
            recent_memory_frames: how long a class stays "trusted" after
                             last being seen at its required count (see
                             RECENT_MEMORY_FRAMES and _effective_count()).
        """
        self._sequence, self._expected = _load_sequence(sequence_json_path)
        self._stage_index = 0   # Stage_01 is the target, not yet confirmed

        self._on_stage_change = on_stage_change
        self._on_warning      = on_warning

        self._window_size = window_size
        self._min_positive = minimum_positive_frames
        self._max_missing_tolerance = max_missing_tolerance
        self._recent_memory_frames = recent_memory_frames
        self._history: deque = deque(maxlen=window_size)
        self._last_confirmed_signature = None

        # Per-class "how many frames since this class was last seen at
        # (or above) its previous count" and the highest count recently
        # observed - starts empty (never seen). Updated every call to
        # update(), independent of the majority-vote window, so it
        # reflects raw recent history, not confirmed history.
        self._frames_since_seen: Dict[str, int] = {}
        self._recent_max_count: Dict[str, int] = {}

        # current_stage stays None until the FIRST stage's expected
        # components are actually seen and majority-confirmed - it is
        # NOT set to Stage_01 by default. Showing "Stage_01" before
        # anything has been detected would display an assumption as if
        # it were a verified reading, which is exactly the bug this
        # fixes: a frame showing an unrelated part (e.g. body_b alone,
        # which isn't even part of Stage_01) must not display
        # "Stage_01" as if that were confirmed.
        self.current_stage = None
        self.is_confirmed = False

        # status/status_detail: sticky readout for the live display,
        # per GPT's mockup (CURRENT STAGE / EXPECTED NEXT / STATUS).
        # Updated only on CONFIRMED events (majority-voted), not every
        # raw frame, so it reads as a verified state, not a flicker.
        self.status = "WAITING"   # "WAITING" | "OK" | "WRONG_PART" | "WRONG_ORDER"
        self.status_detail = None

    # ------------------------------------------------------------------
    def update(self, class_names_observed: List[str]):
        """
        Feed one frame's detected class names into the state machine.
        Pass one entry per detected box, e.g. ["body_a", "bushing",
        "bushing"] for two bushings - NOT a de-duplicated set, since
        counts matter (see module docstring).

        Returns:
            (StageChange or None, AssemblyWarning or None)
            At most one of these fires per call.
        """
        observed = dict(Counter(class_names_observed))

        # Update per-class recency memory BEFORE classifying, so this
        # frame's own detections can immediately count toward "recently
        # seen" (e.g. the very frame that confirms a count also resets
        # its own memory clock).
        self._update_recent_memory(observed)

        signature, kind = self._classify(observed)

        self._history.append(signature)

        stage_change = None
        warning      = None

        # Majority vote: is `signature` the majority reading within the
        # current window, and have we not already acted on it?
        count_in_window = sum(1 for s in self._history if s == signature)
        window_full = len(self._history) >= min(self._window_size, self._min_positive)

        if (
            window_full
            and count_in_window >= self._min_positive
            and signature != self._last_confirmed_signature
        ):
            if kind == "NEXT_STAGE":
                stage_change = self._advance_stage(observed)
            elif kind == "CURRENT_STAGE":
                if not self.is_confirmed:
                    # First time the STARTING stage's own components are
                    # actually seen - the initial confirmation, not a
                    # no-op. Before this fires, current_stage is None
                    # and the display shows "waiting", not a stage name.
                    stage_change = self._confirm_initial_stage()
                else:
                    # Steady state, OR recovering after a warning: the
                    # assembly currently matches what's expected. Clear
                    # any stale WRONG_PART/WRONG_ORDER status - this is
                    # what makes status correctly go back to "OK" once
                    # a mistake is fixed, instead of staying stuck on
                    # the last warning forever.
                    self.status = "OK"
                    self.status_detail = f"{self.current_stage} confirmed"
            elif kind in ("WRONG_ORDER", "WRONG_PART"):
                warning = self._raise_warning(kind, observed)
            # kind == "UNMATCHED": no action, just mark as seen so we
            # don't re-evaluate the same reading every single frame.
            self._last_confirmed_signature = signature

        return stage_change, warning

    # ------------------------------------------------------------------
    def _update_recent_memory(self, observed: Dict[str, int]):
        """
        Track, per class, how many frames since it was last seen and
        the highest count recently observed. This is what lets an
        already-confirmed component (placed and detected several
        seconds ago) stay "trusted" while later fully occluded by a
        hand working on the NEXT part - MAX_MISSING_TOLERANCE alone
        only covers partial occlusion within a single frame, not a
        component being completely out of view for a while.
        """
        all_tracked = set(self._frames_since_seen.keys()) | set(observed.keys())
        for cls in all_tracked:
            count = observed.get(cls, 0)
            if count > 0:
                self._frames_since_seen[cls] = 0
                self._recent_max_count[cls] = max(
                    self._recent_max_count.get(cls, 0), count
                )
            else:
                age = self._frames_since_seen.get(cls, 0) + 1
                self._frames_since_seen[cls] = age
                if age > self._recent_memory_frames:
                    # Memory expired - forget it, this class must be
                    # freshly, genuinely observed again from here on.
                    self._recent_max_count[cls] = 0

    def _effective_count(self, cls: str, observed: Dict[str, int]) -> int:
        """The count to use for matching purposes: whichever is higher,
        what's genuinely visible RIGHT NOW, or what was recently trusted
        (still within its memory window)."""
        current = observed.get(cls, 0)
        age = self._frames_since_seen.get(cls, self._recent_memory_frames + 1)
        recent = self._recent_max_count.get(cls, 0) if age <= self._recent_memory_frames else 0
        return max(current, recent)

    # ------------------------------------------------------------------
    def _matches(self, observed: Dict[str, int], expected: Dict[str, int]) -> bool:
        """
        Tolerant, memory-aware match: True if `observed` (plus recent
        per-class memory) is "close enough" to `expected`.

        Combines two separate mechanisms, deliberately kept distinct:
          - MAX_MISSING_TOLERANCE: partial occlusion WITHIN one frame
            (e.g. a hand covers one of two bushings, but the rest of
            the scene is still visible this instant).
          - Recent per-class memory (_effective_count): a component
            that isn't visible in THIS frame AT ALL, but was clearly
            seen recently, stays trusted for RECENT_MEMORY_FRAMES. This
            is what fixes "inserting block_pin while body_a is fully
            hidden behind my hand" - body_a doesn't need to be
            re-proven every frame once it was already established.

        Both still respect two hard limits, unchanged from before:
          - ZERO foreign classes tolerated (checked against the RAW
            current frame, not memory - a currently-visible wrong part
            must always surface immediately, memory only helps you
            trust something is STILL there, never helps you ignore
            something wrong that just appeared).
          - Excess is checked against the RAW current frame too, for
            the same reason - it's live evidence, not stale evidence.

        Still deliberately NOT "watch only for the single newest part
        in total isolation": if body_a/bushing were NEVER recently
        confirmed either (e.g. a genuinely disconnected part shown far
        from any real assembly context), their memory is empty too, so
        they still count as fully missing and the match still fails.
        Memory only helps when the rest of the assembly was actually
        there a moment ago - not unconditionally.
        """
        foreign = set(observed.keys()) - set(expected.keys())
        if foreign:
            return False
        missing = sum(
            max(0, expected[cls] - self._effective_count(cls, observed))
            for cls in expected
        )
        excess = sum(
            max(0, observed.get(cls, 0) - expected[cls])
            for cls in expected
        )
        return missing <= self._max_missing_tolerance and excess == 0

    def _classify(self, observed: Dict[str, int]):
        """
        Compare observed component counts against the sequence.

        Returns (signature, kind) where kind is one of:
          "CURRENT_STAGE", "NEXT_STAGE", "WRONG_ORDER", "WRONG_PART",
          "UNMATCHED"

        NOTE ON AN INHERENT LIMIT: Stage_01->02 changes bushing's count
        from 1 to 2 - same class, no new class name to look for. No
        tolerance scheme (this one or any other) can confirm "there are
        now 2" without at least one frame where both are actually
        visible together; that's a physical constraint of count-based
        state, not a gap in this logic. Checking CURRENT before NEXT
        below means an ambiguous occluded frame resolves to "no change
        yet" rather than a premature advance, which is the safe default.
        """
        current_name = self._sequence[self._stage_index]
        current_expected = self._expected[current_name]

        if self._matches(observed, current_expected):
            # Signature includes current_name (not just "CURRENT") so
            # confirming Stage_02-as-current can never collide with a
            # later confirmation of Stage_03-as-current - each stage
            # index gets its own signature, which _last_confirmed_signature
            # depends on to avoid suppressing later, genuinely different
            # transitions (see module notes on this fix).
            return (f"CURRENT:{current_name}", "CURRENT_STAGE")

        if self._stage_index + 1 < len(self._sequence):
            next_name = self._sequence[self._stage_index + 1]
            if self._matches(observed, self._expected[next_name]):
                return (f"NEXT:{next_name}", "NEXT_STAGE")

            # Does it match some stage further ahead than the immediate
            # next one? That means a step was skipped.
            for skip_index in range(self._stage_index + 2, len(self._sequence)):
                skip_name = self._sequence[skip_index]
                if self._matches(observed, self._expected[skip_name]):
                    return (f"SKIP:{skip_name}", "WRONG_ORDER")

        # A class that does not belong to the current stage, the next
        # stage, or any later stage at all is treated as a foreign/wrong
        # part rather than a skip-ahead.
        all_known_classes = set()
        for comp in self._expected.values():
            all_known_classes.update(comp.keys())
        unexpected_classes = set(observed.keys()) - all_known_classes
        if unexpected_classes:
            return (f"UNKNOWN:{sorted(unexpected_classes)}", "WRONG_PART")

        # Anything else (partial detections mid-occlusion, a class from
        # the current OR next stage present in the wrong count, etc.)
        # is left unmatched rather than guessed at.
        return (f"PARTIAL:{sorted(observed.items())}", "UNMATCHED")

    # ------------------------------------------------------------------
    def _confirm_initial_stage(self):
        """First-ever confirmation: the starting stage's own expected
        components were actually detected (majority-voted), not
        assumed. from_stage=None marks this as a start, not a
        transition between two real stages."""
        self.is_confirmed = True
        self.current_stage = self._sequence[self._stage_index]
        self.status = "OK"
        self.status_detail = f"Confirmed {self.current_stage}"

        change = StageChange(
            from_stage=None,
            to_stage=self.current_stage,
            timestamp=time.time(),
        )
        if self._on_stage_change:
            self._on_stage_change(change)
        return change

    def _advance_stage(self, observed):
        old_stage = self._sequence[self._stage_index]
        self._stage_index += 1
        new_stage = self._sequence[self._stage_index]
        self.current_stage = new_stage
        self.is_confirmed = True
        self.status = "OK"
        self.status_detail = f"Advanced {old_stage} -> {new_stage}"

        change = StageChange(
            from_stage=old_stage,
            to_stage=new_stage,
            timestamp=time.time(),
        )
        if self._on_stage_change:
            self._on_stage_change(change)
        return change

    @property
    def waiting_for(self) -> Dict[str, int]:
        """What the CURRENT target stage still needs to see, for display
        while is_confirmed is False (e.g. 'waiting for: body_a, bushing')."""
        name = self._sequence[self._stage_index]
        return dict(self._expected[name])

    @property
    def expected_next(self) -> Optional[Dict[str, int]]:
        """Full expected component set for the NEXT stage, or None if
        not yet confirmed OR already at the final stage. Returns None
        while unconfirmed (not just an empty display) because "expected
        next" is only meaningful relative to a confirmed current stage -
        showing it before Stage_01 itself is confirmed would sit next to
        'waiting_for' and say two different 'next part' things at once.
        For display ('Expected next: ...'). Exposed as the full set
        rather than a single delta part because that's not always true
        here - Stage_06 (knob + threaded_pin) is a disjoint locking
        subassembly, not a one-part addition to Stage_05, so a
        single-part label would misrepresent it."""
        if not self.is_confirmed:
            return None
        if self._stage_index + 1 < len(self._sequence):
            next_name = self._sequence[self._stage_index + 1]
            return dict(self._expected[next_name])
        return None

    @property
    def next_action_text(self) -> Optional[str]:
        """Human-readable operator instruction, e.g. 'Install body_b' or
        'Add another bushing (2 needed)' or, for the disjoint Stage_06
        case, 'Install knob, threaded_pin'. Derived from the DELTA
        between the current and next stage's expected components -
        expected_next itself stays the full raw dict (used for
        matching logic), this is a separate, display-only derivation
        so the underlying logic never depends on how it's phrased.
        Returns None under the same conditions as expected_next."""
        next_set = self.expected_next
        if next_set is None:
            return None
        current_set = self._expected[self._sequence[self._stage_index]]

        new_parts = []
        for cls, needed in next_set.items():
            had = current_set.get(cls, 0)
            if needed > had:
                if had == 0:
                    new_parts.append(cls)
                else:
                    new_parts.append(f"{cls} ({needed} needed, have {had})")

        if not new_parts:
            # Shouldn't normally happen (next always requires something
            # beyond current), but don't leave the display blank if it does.
            return "Continue assembly"
        return "Install " + ", ".join(new_parts)

    @property
    def next_action(self) -> Optional[str]:
        """Operator-friendly instruction: only what's NEW relative to
        the already-confirmed current stage, e.g. 'body_b x1' instead
        of the full 'body_a x1, body_b x1, bushing x2' inventory.

        DISPLAY ONLY - computed as a set-difference on top of
        expected_next, purely for readability. Does NOT change what
        _matches() actually requires to confirm a transition, and must
        never be wired into _classify() as if it did - that would
        reopen the exact false-positive hole _matches() was built to
        close (an isolated part confirming a transition on its own).

        For Stage_01->02 (count-only change): shows 'bushing x1' -
        meaning one MORE bushing, not a first one; reads correctly in
        context next to the stage name.
        For Stage_05->06 (disjoint subassembly): shows the whole
        knob+threaded_pin set, because there genuinely isn't a smaller
        instruction possible for that transition.
        """
        nxt = self.expected_next
        if nxt is None:
            return None
        current_expected = self._expected[self._sequence[self._stage_index]]
        delta = {}
        for cls, count in nxt.items():
            added = count - current_expected.get(cls, 0)
            if added > 0:
                delta[cls] = added
        target = delta if delta else nxt  # fallback, shouldn't normally trigger
        return ", ".join(f"{cls} x{count}" for cls, count in target.items())

    def _raise_warning(self, kind, observed):
        current_name = self._sequence[self._stage_index]
        if kind == "WRONG_ORDER":
            message = (
                f"Wrong order: parts for a later stage were detected "
                f"before completing '{current_name}'."
            )
        else:
            message = (
                f"Wrong/unexpected part detected - does not belong to "
                f"'{current_name}' or the next stage."
            )

        warning = AssemblyWarning(
            warning_type=kind,
            message=message,
            timestamp=time.time(),
            current_stage=current_name,
            observed=observed,
        )
        self.status = kind               # "WRONG_PART" | "WRONG_ORDER" - sticky
        self.status_detail = message     # until the next confirmed event changes it
        if self._on_warning:
            self._on_warning(warning)
        return warning
