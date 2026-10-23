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

# Stricter confirmation used ONLY for count-increase transitions
# (same class, higher count - e.g. bushing 1 -> 2). See __init__.
LONG_WINDOW = 20
LONG_MIN_POSITIVE = 14

# Sustained contradiction needed to revoke a confirmed stage.
CONTRADICTION_FRAMES = 45


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
                 position_gate=None,
                 window_size: int = WINDOW_SIZE,
                 minimum_positive_frames: int = MINIMUM_POSITIVE_FRAMES,
                 max_missing_tolerance: int = MAX_MISSING_TOLERANCE,
                 recent_memory_frames: int = RECENT_MEMORY_FRAMES,
                 long_window: int = LONG_WINDOW,
                 long_min_positive: int = LONG_MIN_POSITIVE,
                 contradiction_frames: int = CONTRADICTION_FRAMES):
        """
        Args:
            on_stage_change: callback(StageChange), called whenever a
                             stage transition is confirmed.
            on_warning:      callback(AssemblyWarning), called whenever a
                             wrong-part or wrong-order condition is
                             confirmed.
            sequence_json_path: path to assembly_sequence.json.
            position_gate:   optional callable(stage_name) -> (allow, detail).
                             Asked ONLY at the moment a stage is about to be
                             confirmed, and only about seat position. See
                             _position_allows() for the full contract and
                             for why this machine remains the sole authority
                             on stages.
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
        # Distinct classes that must be seen together to confirm a
        # stage that introduces no new component (see _delta_observed).
        self._min_live_classes_no_delta = 2
        self._history: deque = deque(maxlen=window_size)
        self._last_confirmed_signature = None

        # A COUNT INCREASE (e.g. bushing 1 -> 2) is the hardest thing for
        # a detector to get right: an empty hole in body_a can read as a
        # second bushing for a few frames. A 3-of-5 majority is far too
        # weak for that - a fraction of a second of double-detection was
        # enough to confirm Stage_02 with only ONE bushing physically
        # installed, and the confirmed baseline then made that error
        # permanent. Count increases therefore need a much longer and
        # stricter agreement window than a transition that introduces a
        # genuinely new class.
        self._long_window = long_window
        self._long_min_positive = long_min_positive
        self._long_history: deque = deque(maxlen=long_window)

        # Frames of sustained contradiction before a confirmed stage is
        # revoked. Without this, one false confirmation is unrecoverable
        # except by manual reset.
        self._contradiction_frames = contradiction_frames
        self._contradiction_count = 0

        # Per-class "how many frames since this class was last seen at
        # (or above) its previous count" and the highest count recently
        # observed - starts empty (never seen). Updated every call to
        # update(), independent of the majority-vote window, so it
        # reflects raw recent history, not confirmed history.
        self._frames_since_seen: Dict[str, int] = {}
        self._recent_max_count: Dict[str, int] = {}

        # CONFIRMED BASELINE - components known to be physically in the
        # assembly because a stage containing them was confirmed.
        #
        # WHY THIS EXISTS: recent memory (above) expires after
        # RECENT_MEMORY_FRAMES. But a bushing pressed into body_a at
        # Stage_02 is hidden PERMANENTLY - it is never detected again.
        # Without a baseline every installed part eventually counts as
        # missing, so the missing-unit total grows with every stage.
        # Stage_04 needs 5 component units and Stage_05 needs 6, so with
        # a tolerance of 1 the later stages became unreachable: the
        # system appeared to work to Stage_04 and then stopped advancing.
        # This baseline is what makes Stage_05-07 reachable at all.
        self._confirmed_components: Dict[str, int] = {}

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
        self.status = "WAITING"   # "WAITING" | "OK" | "WRONG_PART" |
                                  # "WRONG_ORDER" | "WRONG_POSITION"
        self.status_detail = None

        # ---------------- POSITION GATE (new) ----------------
        # YOLO class counts cannot distinguish a bushing in the correct
        # seat from the same bushing in the wrong seat: the two seats are
        # 18 mm apart and both readings are "bushing x1". Without a gate,
        # Stage_01 confirms either way.
        #
        # position_gate is an advisory veto on ONE axis only. This machine
        # still decides everything: it must already be satisfied on class
        # counts, ordering and temporal agreement before it asks, and the
        # gate can then only say "not yet, that part is in the wrong seat".
        # The gate can never confirm, advance, skip or revoke a stage, and
        # a gate that is absent, disabled or uncalibrated returns allow,
        # so behaviour is identical to before it existed.
        self._position_gate = position_gate
        self.position_blocked = False      # read by the display
        self.position_block_detail = None
        self.position_blocked_stage = None

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
        self._long_history.append(signature)

        stage_change = None
        warning      = None

        # ---- revoke a confirmed stage that the evidence contradicts ----
        # A confirmed stage used to be permanent. Combined with the
        # confirmed baseline, a single false confirmation (a detector
        # briefly seeing two bushings where one exists) became
        # unrecoverable. Sustained, positive counter-evidence now walks
        # the stage back instead.
        if self.is_confirmed and self._contradicts_confirmed(observed):
            self._contradiction_count += 1
            if self._contradiction_count >= self._contradiction_frames:
                self._revoke_current_stage()
                return None, None
        else:
            self._contradiction_count = 0

        # Majority vote: is `signature` the majority reading within the
        # current window, and have we not already acted on it?
        count_in_window = sum(1 for s in self._history if s == signature)
        window_full = len(self._history) >= min(self._window_size, self._min_positive)

        if (
            window_full
            and count_in_window >= self._min_positive
            and signature != self._last_confirmed_signature
        ):
            if kind == "NEXT_STAGE" and not self.is_confirmed:
                # A stage may never be skipped. While the CURRENT stage
                # is still unconfirmed, evidence that happens to match
                # the NEXT stage must not advance anything - otherwise
                # showing body_a with both bushings already in view
                # jumped straight to Stage_02 and Stage_01 was never
                # displayed at all. The practical effect was that the
                # very first thing the system appeared to wait for was
                # TWO bushings, contradicting Stage_01 on screen.
                #
                # Each stage must now be confirmed in turn, in order.
                self._last_confirmed_signature = signature
                return None, None

            if kind == "NEXT_STAGE":
                # A count increase needs far stronger agreement than a
                # new class appearing - see _is_count_increase.
                nxt = self._expected[self._sequence[self._stage_index + 1]]
                if self._is_count_increase(nxt):
                    long_count = sum(1 for x in self._long_history if x == signature)
                    if (len(self._long_history) < self._long_window
                            or long_count < self._long_min_positive):
                        return stage_change, warning      # not enough evidence yet

                # Counts, order and timing all agree. The last question is
                # whether the new part went into the seat this stage needs.
                nxt_name = self._sequence[self._stage_index + 1]
                if not self._position_allows(nxt_name):
                    return None, None
                stage_change = self._advance_stage(observed)
            elif kind == "CURRENT_STAGE":
                if not self.is_confirmed:
                    if not self._position_allows(
                            self._sequence[self._stage_index]):
                        return None, None
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

    def _baseline_up_to(self, index: int) -> Dict[str, int]:
        """Union (max per class) of every stage's components up to and
        including `index`.

        Baseline must ACCUMULATE, not be replaced by the latest stage.
        Stage_06 is a disjoint sub-assembly (knob + threaded_pin) that
        shares no components with Stage_05. Replacing the baseline on
        confirmation therefore discarded body_a, body_b, the bushings,
        block_pin and nut - which are still physically on the bench -
        and made Stage_07 unreachable. Taking the union reflects what is
        actually present: parts do not vanish because a separate
        sub-assembly was built."""
        merged: Dict[str, int] = {}
        for i in range(index + 1):
            for cls, count in self._expected[self._sequence[i]].items():
                merged[cls] = max(merged.get(cls, 0), count)
        return merged

    def _observed_or_recent(self, cls: str, observed: Dict[str, int]) -> int:
        """Count backed by ACTUAL recent evidence: visible now, or seen
        within the recent-memory window. Deliberately excludes the
        confirmed baseline - used to verify that a NEW component was
        genuinely observed before a stage is allowed to advance."""
        current = observed.get(cls, 0)
        age = self._frames_since_seen.get(cls, self._recent_memory_frames + 1)
        recent = self._recent_max_count.get(cls, 0) if age <= self._recent_memory_frames else 0
        return max(current, recent)

    def _effective_count(self, cls: str, observed: Dict[str, int]) -> int:
        """The count to use for matching: the highest of what is visible
        now, what was recently seen, and what is already confirmed as
        installed. The baseline term is what lets components hidden
        inside the assembly keep counting as present."""
        return max(self._observed_or_recent(cls, observed),
                   self._confirmed_components.get(cls, 0))

    def _is_count_increase(self, target: Dict[str, int]) -> bool:
        """True if what this target adds beyond the confirmed baseline is
        MORE OF A CLASS ALREADY PRESENT, rather than a new class.

        Stage_01 -> Stage_02 is the example: bushing goes 1 -> 2 with no
        new class appearing. Detectors are unreliable exactly here, so
        these transitions get the stricter confirmation window."""
        for cls, needed in target.items():
            already = self._confirmed_components.get(cls, 0)
            if needed > already and already > 0:
                return True
        return False

    def _contradicts_confirmed(self, observed: Dict[str, int]) -> bool:
        """True if the CURRENT frame positively shows fewer of a class
        than the confirmed stage requires - not merely 'not visible',
        but visible in a smaller quantity.

        Requiring the class to be seen at all avoids treating ordinary
        occlusion as contradiction: a bushing hidden behind a hand is
        absent, which proves nothing. One bushing clearly visible where
        two were confirmed is real counter-evidence."""
        name = self._sequence[self._stage_index]
        for cls, needed in self._expected[name].items():
            seen = observed.get(cls, 0)
            if 0 < seen < needed:
                return True
        return False

    def _delta_observed(self, observed: Dict[str, int],
                        target: Dict[str, int]) -> bool:
        """Guard so the baseline alone can never advance a stage.

        A forward move is allowed only if the components that the target
        stage adds BEYOND what is already confirmed were genuinely
        observed (now, or within the recent-memory window). Without
        this, the baseline could satisfy a later stage on its own and
        the system would march forward with no new evidence. The
        baseline may support what is already installed; it may never
        supply proof of something new."""
        has_delta = False
        for cls, needed in target.items():
            if needed > self._confirmed_components.get(cls, 0):
                has_delta = True
                if self._observed_or_recent(cls, observed) < needed:
                    return False

        if not has_delta:
            # Stage_07 adds no NEW component - it is the union of every
            # earlier stage, so there is no delta to prove. Requiring
            # merely "something visible" made it fire the instant
            # Stage_06 confirmed, skipping straight past Stage_06 in the
            # display. Instead require at least two DISTINCT target
            # classes to be live-observed together, which for the final
            # join means seeing a main-assembly part and a locking
            # sub-assembly part in the same view - real evidence that
            # the two halves are together rather than sitting apart.
            #
            # HONEST LIMIT: a count-based system still cannot verify
            # that the two sub-assemblies were physically JOINED, only
            # that parts from both are present together. Stage_07 is
            # the weakest transition in the sequence and should be
            # reported as such in the evaluation.
            # Counted from the CURRENT frame only - deliberately not
            # from recent memory. Memory would still be holding the
            # knob and threaded_pin seen moments earlier while building
            # the sub-assembly, which let Stage_07 fire immediately
            # after Stage_06 and skip it in the display entirely. The
            # join has to be visible NOW, in one view, to count.
            live = sum(1 for c in target if observed.get(c, 0) > 0)
            return live >= self._min_live_classes_no_delta

        return True

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
            if (self._matches(observed, self._expected[next_name])
                    and self._delta_observed(observed, self._expected[next_name])):
                return (f"NEXT:{next_name}", "NEXT_STAGE")

            # Does it match some stage further ahead than the immediate
            # next one? That means a step was skipped.
            for skip_index in range(self._stage_index + 2, len(self._sequence)):
                skip_name = self._sequence[skip_index]
                if (self._matches(observed, self._expected[skip_name])
                        and self._delta_observed(observed, self._expected[skip_name])):
                    return (f"SKIP:{skip_name}", "WRONG_ORDER")

        # ---- WRONG PART -------------------------------------------------
        #
        # THIS BRANCH WAS UNREACHABLE. It compared the observed classes
        # against the union of classes used by ALL stages and flagged
        # whatever was left over. But the detector only emits its seven
        # classes, and all seven are used somewhere across the seven
        # stages, so that leftover set was empty on every frame that has
        # ever been processed. Putting the knob on the bench during
        # Stage_01 produced no warning: it fell through to UNMATCHED and
        # the system said nothing. No amount of tuning could have made
        # the old condition fire.
        #
        # The real question is not "is this part foreign to the product"
        # but "is this part foreign to WHAT THE OPERATOR SHOULD BE DOING
        # RIGHT NOW". A knob is a valid component of this product and
        # still the wrong thing to be holding at Stage_01.
        #
        # So a part is wrong when it belongs to none of:
        #   * the current stage's requirement   - what is being built
        #   * the next stage's requirement      - preparing ahead is fine
        #   * the confirmed baseline            - already installed
        #
        # Looking one stage ahead is deliberate. Operators reach for the
        # next part before finishing the current one, and warning about
        # that would make the system cry wolf until it was switched off.
        allowed_classes = set(current_expected)
        if self._stage_index + 1 < len(self._sequence):
            allowed_classes |= set(
                self._expected[self._sequence[self._stage_index + 1]])
        allowed_classes |= set(self._confirmed_components)

        foreign = sorted(set(observed) - allowed_classes)
        if foreign:
            return (f"WRONGPART:{foreign}", "WRONG_PART")

        # An EXCESS count is the other half of the same mistake: three
        # bushings when the product never uses more than two means a part
        # came from the wrong bin, even though the class itself belongs.
        # Judged against the highest count any stage calls for, so a part
        # legitimately used twice later is not flagged when the second one
        # appears early.
        excess = []
        for cls, count in observed.items():
            ceiling = max((comp.get(cls, 0) for comp in self._expected.values()),
                          default=0)
            if ceiling and count > ceiling:
                excess.append(f"{cls}({count}>{ceiling})")
        if excess:
            return (f"EXCESS:{sorted(excess)}", "WRONG_PART")

        # Anything else (partial detections mid-occlusion, a class from
        # the current OR next stage present in the wrong count, etc.)
        # is left unmatched rather than guessed at.
        return (f"PARTIAL:{sorted(observed.items())}", "UNMATCHED")

    # ------------------------------------------------------------------
    def _position_allows(self, stage_name: str) -> bool:
        """Ask the position gate whether ``stage_name`` may be confirmed.

        Contract, deliberately narrow:

          * Called ONLY after this machine has already decided, on class
            counts and ordering and temporal agreement, that the stage
            should be confirmed. The gate is the last check, never the
            first, and never the one that proposes a stage.
          * The gate may only VETO. It cannot confirm, advance, skip or
            revoke anything, and it is not consulted about which stage is
            current - that remains this machine's decision alone.
          * No gate, a gate that is disabled, a gate with no calibration,
            or a gate that cannot see the part all return allow. Absence
            of position evidence is not evidence of a wrong position, and
            a check that stalled the whole monitor whenever a hand covered
            the seats would be worse than no check at all.
          * A gate that raises is treated as allow and reported once. A
            fault in the position layer must not be able to freeze the
            assembly monitor.

        When the veto fires, _last_confirmed_signature is deliberately
        left untouched. The operator moving the bushing to the right hole
        does not change the class counts at all, so the signature after
        the correction is the SAME signature. Marking it as acted-on here
        would mean the corrected assembly never re-triggers and the stage
        could never confirm - the fix would appear to do nothing.
        """
        if self._position_gate is None:
            return True

        try:
            result = self._position_gate(stage_name)
        except Exception as exc:                            # noqa: BLE001
            print(f"[StateMachine] position gate error, ignored: {exc}")
            return True

        # Accept (allow, detail, status), (allow, detail) or a bare boolean,
        # so a simpler gate - a lambda in a test, say - still works.
        status_hint = "WRONG_POSITION"
        if isinstance(result, tuple):
            allow = bool(result[0])
            detail = result[1] if len(result) > 1 else None
            if len(result) > 2 and result[2]:
                status_hint = str(result[2])
        else:
            allow, detail = bool(result), None

        if allow:
            if self.position_blocked:
                print(f"[StateMachine] position now acceptable for "
                      f"{stage_name} - confirmation may proceed.")
            self.position_blocked = False
            self.position_block_detail = None
            self.position_blocked_stage = None
            return True

        first_time = not self.position_blocked
        self.position_blocked = True
        self.position_block_detail = detail
        self.position_blocked_stage = stage_name
        # WRONG_POSITION means a part is in a hole it should not be in.
        # WAITING_POSITION means the required hole has not been seen filled
        # yet. They are different facts and the operator needs them worded
        # differently - one is a mistake to fix, the other is the system
        # still looking.
        self.status = status_hint
        self.status_detail = detail or "a part is in the wrong position"
        if first_time:
            print(f"[StateMachine] {stage_name} NOT confirmed - "
                  f"{self.status_detail}")
        return False

    # ------------------------------------------------------------------
    def _clear_position_block(self) -> None:
        """Forget any position veto. Used on reset and manual stage jumps,
        so a veto raised before the jump cannot survive it."""
        self.position_blocked = False
        self.position_block_detail = None
        self.position_blocked_stage = None

    # ------------------------------------------------------------------
    def _confirm_initial_stage(self):
        """First-ever confirmation: the starting stage's own expected
        components were actually detected (majority-voted), not
        assumed. from_stage=None marks this as a start, not a
        transition between two real stages."""
        self.is_confirmed = True
        self.current_stage = self._sequence[self._stage_index]
        self._confirmed_components = self._baseline_up_to(self._stage_index)
        self._clear_position_block()
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

    def _revoke_current_stage(self):
        """Step back one stage after sustained counter-evidence, and
        clear all temporal evidence so the re-confirmation is judged
        purely on what is seen from here on."""
        revoked = self.current_stage
        self._stage_index = max(0, self._stage_index - 1)
        self._history.clear()
        self._long_history.clear()
        self._frames_since_seen.clear()
        self._recent_max_count.clear()
        self._last_confirmed_signature = None
        self._contradiction_count = 0
        self._clear_position_block()

        if self._stage_index == 0 and revoked == self._sequence[0]:
            self.is_confirmed = False
            self.current_stage = None
            self._confirmed_components = {}
        else:
            self.is_confirmed = True
            self.current_stage = self._sequence[self._stage_index]
            self._confirmed_components = self._baseline_up_to(self._stage_index)

        self.status = "WAITING"
        self.status_detail = (
            f"{revoked} withdrawn - the parts seen do not support it")
        print(f"[StateMachine] {revoked} revoked: counter-evidence sustained.")

    def _advance_stage(self, observed):
        old_stage = self._sequence[self._stage_index]
        self._stage_index += 1
        new_stage = self._sequence[self._stage_index]
        self.current_stage = new_stage
        self._confirmed_components = self._baseline_up_to(self._stage_index)
        self.is_confirmed = True
        self._clear_position_block()
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

    # ==================================================================
    # MANUAL CONTROL
    # ------------------------------------------------------------------
    # Operator override. The automatic path is unchanged; these methods
    # exist so a supervised trial can be re-run from a chosen point
    # without restarting the session, and so a mis-confirmation can be
    # corrected on the spot rather than invalidating the whole trial.
    # ==================================================================

    def set_position_gate(self, gate) -> None:
        """Attach (or remove, with None) the position gate after construction.

        Wiring it afterwards rather than in the constructor keeps the
        start-up order sane: the position validator reports its warnings
        through the process log, the process log needs the stage
        requirements from this machine, and this machine needs the gate.
        Something has to be connected last, and a veto is the safest thing
        for it to be - until it is attached, behaviour is exactly the
        pre-position behaviour.
        """
        self._position_gate = gate
        self._clear_position_block()

    def set_stage(self, stage_number: int, confirmed: bool = True) -> bool:
        """Jump to a stage by its 1-based number.

        Args:
            stage_number: 1..total_stages
            confirmed: True treats that stage as already completed, so
                monitoring resumes looking for the NEXT one. False
                treats it as the stage now being worked toward, so the
                system waits to observe it - use this to redo a stage.

        Returns True if applied, False if the number is out of range
        (the caller can then ignore a stray keypress safely).

        All temporal evidence is deliberately cleared: the frame window,
        per-class recent memory and the last-confirmed signature. Keeping
        any of it would let observations from before the jump satisfy a
        transition after it, which is exactly the kind of cross-
        contamination that makes a trial unusable as evidence.
        """
        if not (1 <= stage_number <= len(self._sequence)):
            return False

        index = stage_number - 1
        name = self._sequence[index]

        self._stage_index = index
        self._history.clear()
        self._long_history.clear()
        self._last_confirmed_signature = None
        self._frames_since_seen.clear()
        self._recent_max_count.clear()
        self._contradiction_count = 0
        self._clear_position_block()

        if confirmed:
            self.is_confirmed = True
            self.current_stage = name
            self._confirmed_components = self._baseline_up_to(index)
            self.status = "OK"
            self.status_detail = f"Manually set to {name}"
        else:
            self.is_confirmed = False
            self.current_stage = None
            # Baseline = everything confirmed by the PREVIOUS stage, so
            # parts already installed still count while this stage is
            # re-attempted. At stage 1 there is no previous stage.
            if index > 0:
                self._confirmed_components = self._baseline_up_to(index - 1)
            else:
                self._confirmed_components = {}
            self.status = "WAITING"
            self.status_detail = f"Waiting to confirm {name}"

        return True

    def reset(self) -> None:
        """Return to the very start: nothing confirmed, no baseline, no
        temporal evidence. Equivalent to a fresh state machine."""
        self._stage_index = 0
        self._history.clear()
        self._long_history.clear()
        self._last_confirmed_signature = None
        self._frames_since_seen.clear()
        self._recent_max_count.clear()
        self._contradiction_count = 0
        self._confirmed_components = {}
        self.current_stage = None
        self.is_confirmed = False
        self._clear_position_block()
        self.status = "WAITING"
        self.status_detail = None

    @property
    def waiting_for(self) -> Dict[str, int]:
        """What the CURRENT target stage still needs to see, for display
        while is_confirmed is False (e.g. 'waiting for: body_a, bushing')."""
        name = self._sequence[self._stage_index]
        return dict(self._expected[name])

    # ------------------------------------------------------------------
    # Look-ahead properties - display only, no effect on any logic.
    # Added so the operator panel can show BOTH what to do now and what
    # is coming next, letting the operator prepare the following parts
    # while still completing the current stage.
    # ------------------------------------------------------------------

    @property
    def current_requirement(self) -> Dict[str, int]:
        """Exactly what the system is checking for RIGHT NOW.

        Exposed so the panel can print the live requirement instead of a
        separately-worded hint. Display and logic then read from the
        same source and cannot disagree - previously the panel showed a
        phrase while the state machine tested a different dictionary,
        and any mismatch between the two was invisible on screen."""
        name = self._sequence[self._stage_index]
        return dict(self._expected[name])

    @property
    def all_stage_requirements(self):
        """{stage_name: {class: count}} for every stage in the sequence.

        The process log needs this to recognise which observed part sets
        are stages the system was taught and which are combinations it has
        never been told about - the ones worth learning from."""
        return {name: dict(self._expected[name]) for name in self._sequence}

    @property
    def goal_requirement(self):
        """Component set of the finished product (the last stage)."""
        return dict(self._expected[self._sequence[-1]])

    @property
    def target_stage_name(self) -> str:
        """Name of the stage the operator is building RIGHT NOW.

        Distinct from current_stage, which is the last CONFIRMED stage.
        The operator panel and the 360-degree preview must show this one:
        after Stage_01 is confirmed the instruction is already Stage_02,
        and a preview of the finished step would be showing the past.
        Pairs with current_requirement, which is the matching part set.

        NOTE on a known quirk, left as it is on purpose: once a stage is
        confirmed, _stage_index still points AT that stage, so this returns
        the confirmed stage until the next one is confirmed. The display
        compensates - it appends "(done)" when this equals current_stage and
        shows next_stage_name alongside - and the preview behaviour built on
        that has been reviewed and accepted. Changing it here would silently
        alter the panel.

        For "which stage would be confirmed NEXT", which is a different
        question and the one the position gate needs, use
        pending_stage_name below."""
        return self._sequence[self._stage_index]

    @property
    def pending_stage_name(self) -> Optional[str]:
        """The stage that a confirmation would confirm next.

        Unlike target_stage_name this never names an already-confirmed
        stage, which is what the position layer has to reason about: the
        gate is asked "may THIS stage be confirmed", so the validator must
        have been evaluating that same stage, not the one just finished.
        Returns None when every stage is confirmed."""
        if not self.is_confirmed:
            return self._sequence[self._stage_index]
        if self._stage_index + 1 < len(self._sequence):
            return self._sequence[self._stage_index + 1]
        return None

    @property
    def stage_number(self) -> int:
        """1-based position of the current target stage, for 'Stage 3 of 7'.
        Note this is the stage being WORKED TOWARD when unconfirmed, and
        the stage CONFIRMED once is_confirmed is True."""
        return self._stage_index + 1

    @property
    def total_stages(self) -> int:
        return len(self._sequence)

    @property
    def next_stage_name(self) -> Optional[str]:
        """Name of the stage that follows the current one, or None at
        the end of the sequence."""
        if self._stage_index + 1 < len(self._sequence):
            return self._sequence[self._stage_index + 1]
        return None

    @property
    def upcoming_action_text(self) -> Optional[str]:
        """The action needed AFTER the next one - a preparation hint so
        the operator can have the following part ready. Returns None if
        there is no stage that far ahead.

        Deliberately separate from next_action_text: that one is the
        instruction to act on NOW, this is advance notice only. Keeping
        them apart in the API stops the panel from ever presenting a
        preparation hint as if it were the current instruction."""
        i = self._stage_index
        if i + 2 >= len(self._sequence):
            return None
        after_next = self._expected[self._sequence[i + 2]]
        next_set = self._expected[self._sequence[i + 1]]

        new_parts = []
        for cls, needed in after_next.items():
            had = next_set.get(cls, 0)
            if needed > had:
                if had == 0:
                    new_parts.append(cls)
                else:
                    new_parts.append(f"{cls} (x{needed})")
        if not new_parts:
            return None
        return ", ".join(new_parts)

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
                    # Phrased as the ACTION, not the target total.
                    # "Install bushing (2 needed, have 1)" sat under the
                    # heading "Stage_01" and read as though Stage_01
                    # itself required two bushings. It does not -
                    # Stage_01 is body_a + ONE bushing. Say how many
                    # more to add instead.
                    more = needed - had
                    unit = "one more" if more == 1 else f"{more} more"
                    new_parts.append(f"{unit} {cls}")

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

    def _offending_parts(self, observed: Dict[str, int]):
        """Which detected parts actually caused a WRONG_PART verdict.

        Recomputed here rather than carried through the signature string,
        so the wording on screen can never drift away from the rule in
        _classify that produced the verdict. The two must agree, and the
        cheapest way to guarantee that is to derive both from the same
        expression.
        """
        current_expected = self._expected[self._sequence[self._stage_index]]
        allowed = set(current_expected)
        if self._stage_index + 1 < len(self._sequence):
            allowed |= set(self._expected[self._sequence[self._stage_index + 1]])
        allowed |= set(self._confirmed_components)

        foreign = [cls for cls in sorted(observed) if cls not in allowed]
        excess = []
        for cls in sorted(observed):
            if cls in allowed:
                ceiling = max((comp.get(cls, 0) for comp in self._expected.values()),
                              default=0)
                if ceiling and observed[cls] > ceiling:
                    excess.append((cls, observed[cls], ceiling))
        return foreign, excess

    def _raise_warning(self, kind, observed):
        current_name = self._sequence[self._stage_index]
        if kind == "WRONG_ORDER":
            message = (
                f"Wrong order: parts for a later stage were detected "
                f"before completing '{current_name}'."
            )
        else:
            # NAME THE PART. The old wording said only that something
            # unexpected had been seen, leaving the operator to work out
            # what, among everything on the bench, the system meant. A
            # warning that cannot be acted on is noise, and noise gets
            # switched off.
            foreign, excess = self._offending_parts(observed)
            needed = ", ".join(f"{k} x{v}"
                               for k, v in self._expected[current_name].items())
            if foreign:
                message = (
                    f"Wrong part: {', '.join(foreign)} does not belong to "
                    f"'{current_name}'. This stage needs {needed}."
                )
            elif excess:
                cls, seen, ceiling = excess[0]
                message = (
                    f"Too many parts: {seen} x {cls} detected, but this "
                    f"product never uses more than {ceiling}. Remove the "
                    f"extra one."
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
