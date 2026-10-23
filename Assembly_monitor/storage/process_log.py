"""
storage/process_log.py
======================
PROCESS HISTORY / TRACE LOG (V2.3)

This module records the PROCESS, not only the short ACTIVE/IDLE events.
It is deliberately separate from session_logger.py.

It records:
  - the component state on the bench and its dwell time
  - whether that state is a taught stage, an unknown combination, or an
    invalid/wrong attempt reported by the state machine
  - structured stage transitions
  - structured wrong-part / wrong-order warnings
  - operator verification of the final product
  - a machine-readable route through the observed states

Important design rule:
  An unknown combination is NOT automatically called "wrong".
  "Unknown" means: the current assembly profile has not defined that
  combination as a taught state.  The later learning/profile subsystem can
  decide whether such a state is a valid alternative route.

A wrong part / wrong order is different: it is explicitly reported by the
known-assembly state machine and is stored as INVALID_ATTEMPT.

FORMAT: JSON Lines. Every record is independently readable, so a partial
session remains useful even if the program is stopped unexpectedly.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


SCHEMA_VERSION = 2


def combination_id(counts: Dict[str, int] | None) -> str:
    """Return a stable ID for a counted component combination.

    The original V2 format is intentionally preserved so older analysis
    scripts can still recognise IDs such as:
        body_a1-bushing2
    """
    if not counts:
        return "empty"
    return "-".join(
        f"{k}{int(v)}"
        for k, v in sorted(counts.items())
        if int(v) > 0
    )


def normalize_counts(component_counts: Dict[str, int] | None) -> Dict[str, int]:
    """Clean detector counts before they enter the history log."""
    out: Dict[str, int] = {}
    for name, value in (component_counts or {}).items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            out[str(name)] = count
    return dict(sorted(out.items()))


class ProcessLogger:
    """Persistent process history for one assembly-monitoring session."""

    def __init__(
        self,
        log_dir: str | Path,
        session_id: Optional[str] = None,
        known_stages: Optional[Dict[str, str]] = None,
        min_dwell_sec: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.dir / f"process_{self.session_id}.jsonl"

        # {combination_id: stage_name}
        self.known_stages = dict(known_stages or {})
        self.min_dwell_sec = max(0.0, float(min_dwell_sec))
        self.metadata = dict(metadata or {})

        self._t0 = time.time()
        self._closed = False

        self._current: Optional[Dict[str, Any]] = None
        self._candidate: Optional[str] = None
        self._candidate_counts: Dict[str, int] = {}
        self._candidate_context: Dict[str, Any] = {}
        self._candidate_since: Optional[float] = None

        self._states: list[Dict[str, Any]] = []
        self._warnings: list[Dict[str, Any]] = []
        self._transitions: list[Dict[str, Any]] = []
        self._status_changes: list[Dict[str, Any]] = []

        self._last_status: Optional[str] = None
        self._last_status_detail: Optional[str] = None

        self._final_product_verified = False
        self._verification: Optional[Dict[str, Any]] = None

        self._write({
            "type": "session_start",
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "time": datetime.now().isoformat(timespec="seconds"),
            "known_stages": self.known_stages,
            "metadata": self.metadata,
        })
        print(f"[ProcessLog] Recording to: {self.path}")

    # ------------------------------------------------------------------
    # Low-level output
    # ------------------------------------------------------------------

    def _write(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _elapsed(self, timestamp: float) -> float:
        return round(max(0.0, timestamp - self._t0), 3)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _state_type(self, cid: str, status: Optional[str]) -> str:
        status_u = str(status or "").upper()
        if status_u in {"WRONG_PART", "WRONG_ORDER"}:
            return "INVALID_ATTEMPT"
        if cid in self.known_stages:
            return "STANDARD_STAGE"
        return "UNKNOWN_COMBINATION"

    def _close_current(self, now: float) -> None:
        if self._current is None:
            return

        started = float(self._current["started_s"])
        ended = self._elapsed(now)
        self._current["ended_s"] = ended
        self._current["duration_s"] = round(max(0.0, ended - started), 3)

        self._states.append(dict(self._current))
        self._write({"type": "state", **self._current})
        self._current = None

    # ------------------------------------------------------------------
    # Per-frame state observation
    # ------------------------------------------------------------------

    def update(
        self,
        component_counts: Dict[str, int] | None,
        target_stage: Optional[str] = None,
        confirmed_stage: Optional[str] = None,
        status: Optional[str] = None,
        status_detail: Optional[str] = None,
        now: Optional[float] = None,
    ) -> None:
        """Feed the currently observed component counts once per frame.

        The same combination is kept as one state until the observed set
        has changed and the new set has survived ``min_dwell_sec``.
        """
        if self._closed:
            return

        now = time.time() if now is None else float(now)
        counts = normalize_counts(component_counts)
        cid = combination_id(counts)

        # Status changes are useful evidence too, especially when an
        # operator puts the wrong part down and then corrects it.
        if status != self._last_status or status_detail != self._last_status_detail:
            if self._last_status is not None or status is not None:
                record = {
                    "type": "status_change",
                    "time": datetime.now().isoformat(timespec="milliseconds"),
                    "elapsed_s": self._elapsed(now),
                    "from_status": self._last_status,
                    "to_status": status,
                    "status_detail": status_detail,
                    "target_stage": target_stage,
                    "confirmed_stage": confirmed_stage,
                    "combination_id": cid,
                    "components": counts,
                }
                self._status_changes.append(record)
                self._write(record)
            self._last_status = status
            self._last_status_detail = status_detail

        context = {
            "target_stage": target_stage,
            "confirmed_stage": confirmed_stage,
            "status": status,
            "status_detail": status_detail,
        }

        if self._current is not None and self._current["combination_id"] == cid:
            # Keep the latest live decision context available if this state
            # later closes. Do not rewrite the already-written state yet.
            self._current["last_status"] = status
            self._current["last_status_detail"] = status_detail
            self._current["target_stage_last"] = target_stage
            self._current["confirmed_stage_last"] = confirmed_stage
            return

        # New candidate combination. It must stay present long enough to be
        # considered a real process state.
        if self._candidate != cid:
            self._candidate = cid
            self._candidate_counts = counts
            self._candidate_context = context
            self._candidate_since = now
            return

        if self._candidate_since is None or now - self._candidate_since < self.min_dwell_sec:
            return

        start = self._candidate_since
        self._close_current(start)

        state = {
            "combination_id": cid,
            "components": counts,
            "state_type": self._state_type(cid, status),
            "known_stage": self.known_stages.get(cid),
            "is_known_stage": cid in self.known_stages,
            "target_stage": target_stage,
            "confirmed_stage": confirmed_stage,
            "status_at_start": status,
            "status_detail_at_start": status_detail,
            "started_s": self._elapsed(start),
        }
        self._current = state

        self._candidate = None
        self._candidate_counts = {}
        self._candidate_context = {}
        self._candidate_since = None

    # ------------------------------------------------------------------
    # Explicit decision-history records
    # ------------------------------------------------------------------

    def record_stage_change(self, change: Any, transition_type: str = "STANDARD") -> None:
        """Store a structured stage transition alongside the state path."""
        if self._closed or change is None:
            return

        timestamp = float(getattr(change, "timestamp", time.time()))
        record = {
            "type": "stage_transition",
            "schema_version": SCHEMA_VERSION,
            "time": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_s": self._elapsed(timestamp),
            "from_stage": getattr(change, "from_stage", None),
            "to_stage": getattr(change, "to_stage", None),
            "transition_type": str(transition_type),
        }
        self._transitions.append(record)
        self._write(record)

    def record_warning(self, warning: Any) -> None:
        """Store a structured WRONG_PART / WRONG_ORDER decision."""
        if self._closed or warning is None:
            return

        timestamp = float(getattr(warning, "timestamp", time.time()))
        observed = normalize_counts(getattr(warning, "observed", None))
        record = {
            "type": "warning",
            "schema_version": SCHEMA_VERSION,
            "time": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_s": self._elapsed(timestamp),
            "warning_type": getattr(warning, "warning_type", "UNKNOWN_WARNING"),
            "message": getattr(warning, "message", ""),
            "current_stage": getattr(warning, "current_stage", None),
            "observed": observed,
            "combination_id": combination_id(observed),
        }
        self._warnings.append(record)
        self._write(record)

    def mark_final_product_verified(
        self,
        verified: bool = True,
        verifier: str = "operator",
        note: Optional[str] = None,
    ) -> None:
        """Record human verification of the finished product.

        This must remain a separate fact from ``goal_observed``. YOLO/state
        logic can establish that the expected classes were observed; it does
        not by itself establish that the final parts are seated correctly.
        """
        if self._closed:
            return

        self._final_product_verified = bool(verified)
        self._verification = {
            "verified": self._final_product_verified,
            "verifier": verifier,
            "note": note,
            "time": datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": self._elapsed(time.time()),
        }
        self._write({
            "type": "final_product_verification",
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            **self._verification,
        })
        print(
            "[ProcessLog] Final-product verification: "
            + ("VERIFIED" if self._final_product_verified else "NOT VERIFIED")
        )

    # Backwards/short-name convenience for future UI code.
    set_final_product_verified = mark_final_product_verified

    # ------------------------------------------------------------------
    # Session close
    # ------------------------------------------------------------------

    def close(
        self,
        goal_combination_id: Optional[str] = None,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Close the session and write one complete machine-readable summary."""
        if self._closed:
            return getattr(self, "_summary", {})

        now = time.time()
        self._close_current(now)
        self._closed = True

        known = [s for s in self._states if s.get("is_known_stage")]
        unknown = [s for s in self._states if s.get("state_type") == "UNKNOWN_COMBINATION"]
        invalid = [s for s in self._states if s.get("state_type") == "INVALID_ATTEMPT"]

        path = [s["combination_id"] for s in self._states]
        goal_observed = bool(
            goal_combination_id
            and any(s["combination_id"] == goal_combination_id for s in self._states)
        )

        first_build = next(
            (s for s in self._states if s["combination_id"] != "empty"), None
        )
        goal_state = next(
            (s for s in reversed(self._states)
             if goal_combination_id and s["combination_id"] == goal_combination_id),
            None,
        )
        if first_build and goal_state:
            build_start_s = float(first_build["started_s"])
            build_end_s = float(goal_state["ended_s"])
            build_time_s = round(max(0.0, build_end_s - build_start_s), 3)
        else:
            build_start_s = None
            build_end_s = None
            build_time_s = None

        eligible_for_learning = bool(
            goal_observed
            and self._final_product_verified
            and not self._warnings
        )

        summary = {
            "type": "session_end",
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "time": datetime.now().isoformat(timespec="seconds"),
            "total_time_s": round(now - self._t0, 3),
            "build_start_s": build_start_s,
            "build_end_s": build_end_s,
            "build_time_s": build_time_s,
            "states_observed": len(self._states),
            "known_states": len(known),
            "unknown_states": len(unknown),
            "invalid_states": len(invalid),
            "warning_count": len(self._warnings),
            "stage_transition_count": len(self._transitions),
            "goal_combination_id": goal_combination_id,
            "goal_observed": goal_observed,
            "final_product_verified": self._final_product_verified,
            "verification": self._verification,
            "eligible_for_learning": eligible_for_learning,
            "path": path,
            "note": note,
        }
        self._summary = summary
        self._write(summary)

        print(
            f"[ProcessLog] {len(self._states)} states | "
            f"unknown={len(unknown)} | invalid={len(self._warnings)} | "
            f"goal observed={goal_observed} | "
            f"final verified={self._final_product_verified} | "
            f"learning eligible={eligible_for_learning}"
        )
        print(f"[ProcessLog] Saved: {self.path}")
        return summary
