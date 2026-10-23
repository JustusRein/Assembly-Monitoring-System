"""
===============================================================================
process_learner.py - learn a faster assembly order from recorded sessions
===============================================================================

Reads every process_*.jsonl written by storage/process_log.py and answers
one question:

    has anyone reached the SAME finished product by a different route,
    reliably, and faster than the order we currently instruct?

If yes it writes a PROPOSAL. It never changes the instructions by itself.

WHY APPROVAL AND NOT AUTOMATIC ADOPTION:
  A shorter route is not automatically a correct one. The system sees which
  parts are present, not whether each one is seated correctly, so a "faster"
  path may be one where a step was skipped or a part left loose. Until the
  placement check can confirm the result is geometrically correct, a human
  has to approve. That is also the normal rule in manufacturing: a process
  change is approved by a process engineer, not adopted silently.

HOW A ROUTE IS JUDGED (all four must hold):
  1. it reached the goal state (the complete product)
  2. it was observed at least MIN_RUNS times - one fast run can be luck
  3. its MEDIAN time is faster than the current order by MIN_GAIN_PERCENT
  4. its spread is not wild (median absolute deviation reported, so a route
     that is sometimes fast and sometimes terrible is visible)

USAGE
    python process_learner.py --logs "D:\\...\\Assembly_monitor\\logs\\process"
    python process_learner.py --logs ... --min-runs 3 --min-gain 10
===============================================================================
"""

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def load_sessions(log_dir):
    """One record per session file: path taken, total time, goal reached."""
    sessions = []
    for path in sorted(Path(log_dir).glob("process_*.jsonl")):
        states, end = [], None
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
            continue
        if end is None:
            print(f"  skipped {path.name}: no session_end (interrupted run?)")
            continue
        sessions.append({
            "file": path.name,
            "session_id": end.get("session_id"),
            "total_time_s": end.get("total_time_s"),
            "reached_goal": end.get("reached_goal"),
            "goal": end.get("goal_combination_id"),
            "path": end.get("path", []),
            "states": states,
        })
    return sessions


def route_key(path):
    """The route, ignoring repeats caused by a part being briefly lost."""
    out = []
    for state in path:
        if not out or out[-1] != state:
            out.append(state)
    return tuple(out)


def analyse(sessions, reference_route=None, min_runs=3, min_gain=10.0):
    usable = [s for s in sessions if s["reached_goal"] and s["total_time_s"]]
    if not usable:
        return None, []

    routes = defaultdict(list)
    for s in usable:
        routes[route_key(s["path"])].append(s)

    rows = []
    for route, runs in routes.items():
        times = [r["total_time_s"] for r in runs]
        med = statistics.median(times)
        mad = (statistics.median([abs(t - med) for t in times])
               if len(times) > 1 else 0.0)
        rows.append({
            "route": route,
            "runs": len(runs),
            "median_s": round(med, 1),
            "fastest_s": round(min(times), 1),
            "slowest_s": round(max(times), 1),
            "spread_s": round(mad, 1),
            "steps": len(route),
            "sessions": [r["session_id"] for r in runs],
        })
    rows.sort(key=lambda r: r["median_s"])

    # The reference is the route we currently instruct, if it was observed;
    # otherwise the most frequently used route.
    reference = None
    if reference_route:
        reference = next((r for r in rows if r["route"] == tuple(reference_route)), None)
    if reference is None:
        reference = max(rows, key=lambda r: r["runs"])

    candidates = []
    for row in rows:
        if row["route"] == reference["route"]:
            continue
        gain = (reference["median_s"] - row["median_s"]) / reference["median_s"] * 100.0
        row["gain_percent"] = round(gain, 1)
        row["enough_runs"] = row["runs"] >= min_runs
        row["enough_gain"] = gain >= min_gain
        row["qualifies"] = row["enough_runs"] and row["enough_gain"]
        candidates.append(row)

    return reference, candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="folder with process_*.jsonl")
    ap.add_argument("--min-runs", type=int, default=3)
    ap.add_argument("--min-gain", type=float, default=10.0,
                    help="percent faster before a route is proposed")
    ap.add_argument("--reference", help="JSON file with the current route")
    ap.add_argument("--out", help="where to write the proposal")
    args = ap.parse_args()

    sessions = load_sessions(args.logs)
    print("=" * 72)
    print(f"PROCESS LEARNER   {len(sessions)} sessions from {args.logs}")
    print("=" * 72)
    if not sessions:
        return

    completed = [s for s in sessions if s["reached_goal"]]
    print(f"  completed the product : {len(completed)} / {len(sessions)}")

    reference_route = None
    if args.reference and Path(args.reference).exists():
        reference_route = json.loads(
            Path(args.reference).read_text(encoding="utf-8")).get("route")

    reference, candidates = analyse(sessions, reference_route,
                                    args.min_runs, args.min_gain)
    if reference is None:
        print("  no session reached the finished product yet - nothing to learn")
        return

    print()
    print(f"  CURRENT ROUTE   {reference['runs']} runs, "
          f"median {reference['median_s']}s "
          f"(fastest {reference['fastest_s']}s, slowest {reference['slowest_s']}s)")
    for step in reference["route"]:
        print(f"      {step}")

    print()
    print("  OTHER ROUTES OBSERVED")
    if not candidates:
        print("      none - everyone built it the same way")
    hdr = f"      {'runs':>5}{'median':>9}{'spread':>8}{'faster by':>11}   verdict"
    print(hdr)
    print("      " + "-" * (len(hdr) - 6))
    for row in candidates:
        if row["qualifies"]:
            verdict = "PROPOSE"
        elif not row["enough_runs"]:
            verdict = f"too few runs (need {args.min_runs})"
        else:
            verdict = "not enough gain"
        print(f"      {row['runs']:>5}{row['median_s']:>9}{row['spread_s']:>8}"
              f"{row['gain_percent']:>10}%   {verdict}")

    winners = [r for r in candidates if r["qualifies"]]
    if not winners:
        print()
        print("  No route qualifies yet. Keep recording sessions.")
        return

    best = winners[0]
    print()
    print("=" * 72)
    print(f"  PROPOSAL: a route {best['gain_percent']}% faster, "
          f"seen {best['runs']} times")
    print("=" * 72)
    for step in best["route"]:
        print(f"      {step}")
    print()
    print("  This is a PROPOSAL, not a change. Before adopting it:")
    print("    1. watch one build following this route")
    print("    2. confirm the finished product is correct, not just complete")
    print("    3. have the process engineer approve it")

    proposal = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "status": "PROPOSED - awaiting human approval",
        "current_route": {k: v for k, v in reference.items() if k != "route"},
        "current_route_steps": list(reference["route"]),
        "proposed_route": {k: v for k, v in best.items() if k != "route"},
        "proposed_route_steps": list(best["route"]),
        "evidence_sessions": best["sessions"],
        "criteria": {"min_runs": args.min_runs,
                     "min_gain_percent": args.min_gain},
    }
    out = Path(args.out) if args.out else Path(args.logs) / "learned_route_proposal.json"
    out.write_text(json.dumps(proposal, indent=2), encoding="utf-8")
    print(f"\n  Written: {out}")


if __name__ == "__main__":
    main()
