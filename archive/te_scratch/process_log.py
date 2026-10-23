"""
storage/process_log.py
======================
SUBSYSTEM: Storage (SS8)

Records WHAT was on the bench, WHEN, and FOR HOW LONG - as a plain file a
human can read and a program can analyse.

WHY A SEPARATE LOG FROM session_logger.py:
  session_logger records EVENTS (active / idle / stage confirmed). This one
  records STATES: the exact set of parts observed, how long that set lasted,
  and whether it is a stage the system knows or a combination it has never
  been told about. That is what makes two things possible later:

    1. an operator or engineer can open one file and see the whole build
    2. the learner (process_learner.py) can compare different ways of
       reaching the same finished product and find the quicker one

WHAT COUNTS AS A STATE CHANGE:
  The set of component COUNTS changes and stays changed for
  min_dwell_sec. The dwell time matters: detections flicker, and without
  it every dropped frame would be logged as a new state and the timings
  would be meaningless.

FORMAT: JSON Lines - one self-contained JSON object per line. Readable in
a text editor, loadable by pandas, and not corrupted if the program is
killed mid-session, which a single big JSON object would be.
"""

import json
import time
from datetime import datetime
from pathlib import Path


def combination_id(counts):
    """Stable name for a set of parts: 'block_pin1-body_a1-bushing2'.

    Sorted by class name so the same physical state always produces the
    same id, whatever order the detector happened to report the parts in.
    """
    if not counts:
        return "empty"
    return "-".join(f"{k}{v}" for k, v in sorted(counts.items()) if v > 0)


class ProcessLogger:

    def __init__(self, log_dir, session_id=None, known_stages=None,
                 min_dwell_sec=1.0):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.dir / f"process_{self.session_id}.jsonl"

        # {combination_id: stage_name} for states the system already knows
        self.known_stages = dict(known_stages or {})
        self.min_dwell_sec = float(min_dwell_sec)

        self._t0 = time.time()
        self._current = None          # the state currently being timed
        self._candidate = None        # a change not yet held long enough
        self._candidate_since = None
        self._states = []             # closed states, in order

        self._write({
            "type": "session_start",
            "session_id": self.session_id,
            "time": datetime.now().isoformat(timespec="seconds"),
            "known_stages": self.known_stages,
        })
        print(f"[ProcessLog] Recording to: {self.path}")

    # ------------------------------------------------------------------

    def _write(self, record):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _close_current(self, now):
        if self._current is None:
            return
        self._current["ended_s"] = round(now - self._t0, 2)
        self._current["duration_s"] = round(
            self._current["ended_s"] - self._current["started_s"], 2)
        self._states.append(self._current)
        self._write({"type": "state", **self._current})
        self._current = None

    # ------------------------------------------------------------------

    def update(self, component_counts, target_stage=None,
               confirmed_stage=None, status=None, now=None):
        """Call once per frame with the parts currently detected."""
        now = now if now is not None else time.time()
        counts = {k: int(v) for k, v in (component_counts or {}).items() if v > 0}
        cid = combination_id(counts)

        if self._current is not None and self._current["combination_id"] == cid:
            self._candidate = None
            return

        # A different set: only accept it once it has persisted, so that
        # detector flicker does not create phantom states.
        if self._candidate != cid:
            self._candidate = cid
            self._candidate_since = now
            return
        if now - self._candidate_since < self.min_dwell_sec:
            return

        start = self._candidate_since
        self._close_current(start)
        self._current = {
            "combination_id": cid,
            "components": counts,
            "known_stage": self.known_stages.get(cid),
            "is_known": cid in self.known_stages,
            "target_stage": target_stage,
            "confirmed_stage": confirmed_stage,
            "status": status,
            "started_s": round(start - self._t0, 2),
        }
        self._candidate = None

    # ------------------------------------------------------------------

    def close(self, goal_combination_id=None, note=None):
        now = time.time()
        self._close_current(now)

        known = [s for s in self._states if s["is_known"]]
        unknown = [s for s in self._states if not s["is_known"]]
        reached_goal = bool(goal_combination_id) and any(
            s["combination_id"] == goal_combination_id for s in self._states)

        summary = {
            "type": "session_end",
            "session_id": self.session_id,
            "total_time_s": round(now - self._t0, 2),
            "states_observed": len(self._states),
            "known_states": len(known),
            "unknown_states": len(unknown),
            "reached_goal": reached_goal,
            "goal_combination_id": goal_combination_id,
            "path": [s["combination_id"] for s in self._states],
            "note": note,
        }
        self._write(summary)
        print(f"[ProcessLog] {len(self._states)} states "
              f"({len(unknown)} not in the defined sequence), "
              f"{summary['total_time_s']:.1f}s, "
              f"goal reached: {reached_goal}")
        print(f"[ProcessLog] Saved: {self.path}")
        return summary
