"""
process_learner.py
==================
SAFE ROUTE LEARNER

Compares completed assembly sessions and proposes faster routes.

Important safety rule:
  It does NOT learn from a session merely because the last class set looked
  complete. The finished product must have been explicitly verified by a
  human, and the run must contain no recorded WRONG_PART / WRONG_ORDER
  warning.

Unknown combinations are retained as observations. They can become valid
alternatives later, after the assembly/profile layer defines or approves
that route. This script never silently converts an unknown state into an
approved manufacturing instruction.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


def _read_jsonl(path: Path) -> tuple[list[dict], Optional[dict]]:
    states: list[dict] = []
    end: Optional[dict] = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("type") == "state":
                states.append(rec)
            elif rec.get("type") == "session_end":
                end = rec
    except (ValueError, OSError) as exc:
        print(f"  skipped {path.name}: {exc}")
        return [], None
    return states, end


def load_sessions(log_dir: str | Path) -> list[dict]:
    """Load all complete process sessions."""
    sessions: list[dict] = []
    for path in sorted(Path(log_dir).glob("process_*.jsonl")):
        states, end = _read_jsonl(path)
        if end is None:
            print(f"  skipped {path.name}: no session_end (interrupted run?)")
            continue
        sessions.append({
            "file": path.name,
            "session_id": end.get("session_id"),
            "total_time_s": end.get("total_time_s"),
            "build_time_s": end.get("build_time_s"),
            "goal_observed": bool(end.get("goal_reached",
                                  end.get("goal_observed",
                                  end.get("reached_goal", False)))),
            "stage_route": end.get("stage_route", []),
            "final_product_verified": bool(end.get("final_product_verified", False)),
            "eligible_for_learning": bool(end.get("eligible_for_learning", False)),
            "warning_count": int(end.get("warning_count", 0) or 0),
            "path": end.get("path", []),
            "states": states,
            "unknown_states": int(end.get("unknown_states", 0) or 0),
        })
    return sessions


def route_key(path: Sequence[str]) -> Tuple[str, ...]:
    """Collapse consecutive duplicates in a state path."""
    out: list[str] = []
    for state in path:
        if not out or out[-1] != state:
            out.append(state)
    return tuple(out)


def collapse_dropouts(seq: Sequence[str], max_passes: int = 4) -> list[str]:
    """Remove A -> B -> A patterns, where B is a momentary detection loss.

    This is what a dropped detection looks like in the log: the assembly
    does not change, but one part is briefly unseen, so the recorded set
    shrinks and then returns. The operator did nothing - the detector
    blinked. Left in, every run gets a slightly different route and no two
    runs can ever be compared.

    Only the sandwiched entry is removed; a genuine A -> B -> C stays.
    """
    out = list(seq)
    for _ in range(max_passes):
        changed = False
        i = 1
        while i < len(out) - 1:
            if out[i - 1] == out[i + 1] and out[i] != out[i - 1]:
                del out[i]
                changed = True
            else:
                i += 1
        if not changed:
            break
    return out


def _is_strict_subset(counts_a: Dict[str, int], counts_b: Dict[str, int]) -> bool:
    """True if A contains only parts of B, in equal or smaller numbers,
    and strictly fewer parts overall."""
    if not counts_a or not counts_b:
        return False
    for cls, n in counts_a.items():
        if n > counts_b.get(cls, 0):
            return False
    return sum(counts_a.values()) < sum(counts_b.values())


def session_route(session: dict, min_state_sec: float = 0.0,
                  stages_only: bool = False) -> Tuple[str, ...]:
    """The route actually taken, built from the recorded STATES.

    Why not simply use the stored "path": a detection that drops out for a
    second inserts a phantom state (a bushing briefly unseen turns
    body_a1-bushing2 into body_a1-bushing1 and back). Two operators doing
    exactly the same thing then produce different routes, every route is
    seen once, nothing ever reaches min_runs, and the learner silently
    concludes there is nothing to learn.

    min_state_sec drops states too short to be a real assembly step.
    stages_only reduces the route to the stages the system knows, which
    compares the meaningful skeleton and ignores intermediate handling.
    """
    # The stage route comes from the state machine, which already applies
    # temporal filtering and strict confirmation. When it is present it is
    # the honest record of what was completed, and it is immune to the
    # detector losing sight of a part behind the operator's hands.
    stage_route = session.get("stage_route") or []
    if stages_only and stage_route:
        return route_key(stage_route)

    states = session.get("states") or []
    if not states:
        return route_key(session.get("path", []))

    seq: list[str] = []
    last_counts: Dict[str, int] = {}
    for st in states:
        cid = st.get("combination_id")
        if not cid or cid == "empty":
            continue
        try:
            duration = float(st.get("duration_s") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        if duration < min_state_sec:
            continue
        # A state that only LOSES parts compared with the previous one is a
        # detection dropout, not an assembly step: during a build parts are
        # added, not removed. Keeping these makes every run look like a
        # different route, and then nothing can ever be compared.
        counts = st.get("components") or {}
        if last_counts and _is_strict_subset(counts, last_counts):
            continue
        last_counts = counts

        if stages_only:
            stage = st.get("known_stage")
            if not stage:
                continue
            seq.append(stage)
        else:
            seq.append(cid)
    return route_key(collapse_dropouts(route_key(seq)))


def stage_timing_table(sessions: Sequence[dict]) -> list[dict]:
    """Median time spent on each known stage, across eligible runs.

    Shows WHERE the time goes, which is what makes a faster route
    explainable instead of just shorter."""
    per_stage: defaultdict[str, list[float]] = defaultdict(list)
    for s in sessions:
        for st in s.get("states") or []:
            stage = st.get("known_stage")
            if not stage:
                continue
            try:
                per_stage[stage].append(float(st.get("duration_s") or 0.0))
            except (TypeError, ValueError):
                continue
    rows = []
    for stage, times in sorted(per_stage.items()):
        rows.append({
            "stage": stage,
            "observations": len(times),
            "median_s": round(statistics.median(times), 1),
            "fastest_s": round(min(times), 1),
            "slowest_s": round(max(times), 1),
        })
    return rows


def _time_value(session: dict) -> Optional[float]:
    """Prefer process build time; fall back to total time for old logs."""
    value = session.get("build_time_s")
    if value is None:
        value = session.get("total_time_s")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def analyse(
    sessions: Sequence[dict],
    reference_route: Optional[Sequence[str]] = None,
    min_runs: int = 3,
    min_gain: float = 10.0,
    min_state_sec: float = 0.0,
    stages_only: bool = False,
) -> tuple[Optional[dict], list[dict]]:
    """Return the observed baseline route and safe alternative candidates."""
    usable = [
        s for s in sessions
        if s.get("goal_observed")
        and s.get("final_product_verified")
        and s.get("warning_count", 0) == 0
        and _time_value(s) is not None
    ]
    if not usable:
        return None, []

    routes: defaultdict[Tuple[str, ...], list[dict]] = defaultdict(list)
    for session in usable:
        routes[session_route(session, min_state_sec, stages_only)].append(session)

    rows: list[dict] = []
    for route, runs in routes.items():
        times = [_time_value(r) for r in runs]
        times = [float(t) for t in times if t is not None]
        med = statistics.median(times)
        mad = (
            statistics.median([abs(t - med) for t in times])
            if len(times) > 1 else 0.0
        )
        rows.append({
            "route": route,
            "runs": len(runs),
            "median_s": round(med, 3),
            "fastest_s": round(min(times), 3),
            "slowest_s": round(max(times), 3),
            "spread_s": round(mad, 3),
            "steps": len(route),
            "sessions": [r["session_id"] for r in runs],
        })

    rows.sort(key=lambda r: (r["median_s"], -r["runs"]))

    reference = None
    if reference_route:
        reference = next(
            (r for r in rows if r["route"] == tuple(reference_route)),
            None,
        )
    if reference is None:
        reference = max(rows, key=lambda r: r["runs"])

    candidates: list[dict] = []
    for row in rows:
        if row["route"] == reference["route"]:
            continue
        if reference["median_s"] <= 0:
            gain = 0.0
        else:
            gain = (
                (reference["median_s"] - row["median_s"])
                / reference["median_s"] * 100.0
            )
        candidate = dict(row)
        candidate["gain_percent"] = round(gain, 1)
        candidate["enough_runs"] = row["runs"] >= min_runs
        candidate["enough_gain"] = gain >= min_gain
        candidate["qualifies"] = bool(candidate["enough_runs"] and candidate["enough_gain"])
        candidates.append(candidate)

    return reference, candidates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="folder with process_*.jsonl")
    ap.add_argument("--min-runs", type=int, default=3)
    ap.add_argument("--min-state-sec", type=float, default=2.0,
                    help="ignore states shorter than this when comparing "
                         "routes; they are usually detection flicker, not "
                         "assembly steps")
    ap.add_argument("--stages-only", action="store_true",
                    help="compare routes by the known stages only, ignoring "
                         "intermediate part handling")
    ap.add_argument("--min-gain", type=float, default=10.0)
    ap.add_argument("--reference", help="JSON file containing {\"route\": [...]}")
    ap.add_argument("--out", help="proposal output JSON")
    args = ap.parse_args()

    sessions = load_sessions(args.logs)
    print("=" * 76)
    print(f"PROCESS LEARNER   {len(sessions)} complete session files")
    print("=" * 76)

    verified = [s for s in sessions if s.get("final_product_verified")]
    clean_verified = [
        s for s in verified
        if s.get("goal_observed") and s.get("warning_count", 0) == 0
    ]
    print(f"  final product verified       : {len(verified)}")
    print(f"  verified + complete + clean  : {len(clean_verified)}")

    reference_route = None
    if args.reference and Path(args.reference).exists():
        try:
            reference_route = json.loads(
                Path(args.reference).read_text(encoding="utf-8")
            ).get("route")
        except (ValueError, OSError) as exc:
            print(f"  reference file could not be read: {exc}")

    reference, candidates = analyse(
        sessions,
        reference_route=reference_route,
        min_runs=max(1, args.min_runs),
        min_gain=args.min_gain,
        min_state_sec=args.min_state_sec,
        stages_only=args.stages_only,
    )

    mode = "known stages only" if args.stages_only else "full part combinations"
    print(f"  route comparison      : {mode}, "
          f"ignoring states under {args.min_state_sec:g}s")

    eligible_sessions = [s for s in sessions if s.get("eligible_for_learning")]
    timing = stage_timing_table(eligible_sessions)
    if timing:
        print()
        print("  TIME PER STAGE (median over usable runs)")
        print(f"      {'stage':<12}{'seen':>6}{'median':>9}{'fastest':>9}{'slowest':>9}")
        print("      " + "-" * 45)
        for row in timing:
            print(f"      {row['stage']:<12}{row['observations']:>6}"
                  f"{row['median_s']:>9}{row['fastest_s']:>9}{row['slowest_s']:>9}")
        slowest = max(timing, key=lambda r: r["median_s"])
        print(f"      -> most time is spent on {slowest['stage']} "
              f"({slowest['median_s']}s median)")

    if reference is None:
        print()
        print("  No learning-safe route exists yet.")
        print("  Finish an assembly, verify the final product with [v], and")
        print("  record at least a few clean runs before proposing a route.")
        return

    print()
    print(
        f"  BASELINE ROUTE   {reference['runs']} verified clean runs, "
        f"median {reference['median_s']}s"
    )
    for step in reference["route"]:
        print(f"      {step}")

    print()
    print("  OTHER CLEAN VERIFIED ROUTES")
    if not candidates:
        print("      none")
    for row in candidates:
        if row["qualifies"]:
            verdict = "PROPOSE"
        elif not row["enough_runs"]:
            verdict = f"too few runs (need {args.min_runs})"
        else:
            verdict = "not enough gain"
        print(
            f"      runs={row['runs']:>2}  median={row['median_s']:>8.3f}s "
            f"spread={row['spread_s']:>7.3f}s  faster={row['gain_percent']:>6.1f}%  "
            f"{verdict}"
        )

    winners = [r for r in candidates if r["qualifies"]]
    if not winners:
        print()
        print("  No route qualifies yet. Keep recording verified sessions.")
        return

    best = winners[0]
    print()
    print("=" * 76)
    print(
        f"  PROPOSAL: route is {best['gain_percent']}% faster "
        f"across {best['runs']} verified clean runs"
    )
    print("=" * 76)
    for step in best["route"]:
        print(f"      {step}")
    print()
    print("  This remains a PROPOSAL. Human/process-engineer approval is required")
    print("  before the route becomes the official instruction sequence.")

    proposal = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "status": "PROPOSED - awaiting human approval",
        "learning_policy": {
            "requires_final_product_verification": True,
            "rejects_sessions_with_warnings": True,
            "min_runs": args.min_runs,
            "min_gain_percent": args.min_gain,
        },
        "current_route_steps": list(reference["route"]),
        "current_route_evidence": {
            k: v for k, v in reference.items() if k != "route"
        },
        "proposed_route_steps": list(best["route"]),
        "proposed_route_evidence": {
            k: v for k, v in best.items() if k != "route"
        },
        "evidence_sessions": best["sessions"],
    }
    out = Path(args.out) if args.out else Path(args.logs) / "learned_route_proposal.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(proposal, indent=2), encoding="utf-8")
    print(f"\n  Written: {out}")


if __name__ == "__main__":
    main()
