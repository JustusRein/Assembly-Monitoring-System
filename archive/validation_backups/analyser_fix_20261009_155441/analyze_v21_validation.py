r"""
===============================================================================
analyze_v21_validation.py
===============================================================================
Turns the V2.1 process logs into thesis material: tables, figures and LaTeX
you can paste straight into the Validation chapter.

WHAT IT READS
    logs/process/*.jsonl        expected process-log output from main.py; verify
                                the live logger schema before trusting metrics
    trials.csv                  a small ground-truth file you fill in by hand
                                (the script writes a template if missing)

WHAT IT WRITES  (into validation_output/)
    v21_session_summary.csv     one row per trial
    v21_stage_timing.csv        every stage transition with its duration
    v21_error_events.csv        every warning the system raised
    v21_metrics.json            everything, machine-readable
    v21_validation_report.txt   the written-up result
    tab_*.tex                   LaTeX tables, ready to \input
    fig_*.pdf / fig_*.png       figures, PDF for LaTeX and PNG to look at

WHY A GROUND-TRUTH FILE IS NEEDED
    The V1 chapter had to admit that 49 detected events could not be turned
    into an accuracy figure, because nothing recorded what the TRUE events
    were. That is the gap this file closes. You write down, before or just
    after each run, what you actually did - which stages you completed, and
    whether you deliberately introduced an error. The script then compares
    what the system reported against what really happened, and that
    comparison is what makes this an accuracy experiment rather than a
    functional demonstration.

    Filling in trials.csv is a required part of the measurement, but it is not
    sufficient on its own: the actual log record types, timestamps, and warning
    names must correspond to the intended event being scored.

USAGE  (Windows)
    cd "D:\\Thesis\\SOFTWARE V2.1 AI Final\\Assembly_monitor"
    python analyze_v21_validation.py

    First run writes validation_output\\trials_TEMPLATE.csv listing every log
    file it found. Fill it in, save it as trials.csv next to it, run again.

===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ═══════════════════════════════════════════════════════════════════════════
# Appearance - muted, print-safe, colour-blind safe
# ═══════════════════════════════════════════════════════════════════════════

C_OK = "#2b6cb0"        # blue    - normal / confirmed
C_WARN = "#d97706"      # amber   - warning raised
C_BAD = "#c0392b"       # red     - error / missed
C_GREY = "#8a8a8a"
C_LIGHT = "#d9d9d9"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 150,
})

EXPECTED_SEQUENCE = [f"Stage_{i:02d}" for i in range(1, 8)]

# Conditions you can write in trials.csv. Anything else is treated as
# "other" and still counted, just not grouped.
CONDITION_NOMINAL = "nominal"
CONDITION_WRONG_PART = "induced_wrong_part"
CONDITION_WRONG_ORDER = "induced_wrong_order"
CONDITION_WRONG_POSITION = "induced_wrong_position"

EXPECTED_WARNING = {
    CONDITION_WRONG_PART: "WRONG_PART",
    CONDITION_WRONG_ORDER: "WRONG_ORDER",
    CONDITION_WRONG_POSITION: "WRONG_POSITION",
}


def log(msg):
    print(f"[v21-validation] {msg}")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════
# Reading the logs
# ═══════════════════════════════════════════════════════════════════════════

def read_jsonl(path: Path) -> list:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                log(f"  {path.name}: line {n} is not valid JSON, skipped")
    return out


def parse_session(path: Path) -> dict:
    """Pull one session's records into a flat structure."""
    records = read_jsonl(path)

    states, transitions, warnings, status_changes = [], [], [], []
    summary, verification, session_start = None, None, None

    for r in records:
        t = r.get("type")
        if t == "session_start":
            session_start = r
        elif t == "state":
            states.append(r)
        elif t == "stage_transition":
            transitions.append(r)
        elif t == "warning":
            warnings.append(r)
        elif t == "status_change":
            status_changes.append(r)
        elif t == "session_end":
            summary = r
        elif t == "final_product_verification":
            verification = r

    # The deployed ProcessLogger schema stores the stage route as individual
    # stage_transition records. It does NOT write summary fields named
    # ``stage_route`` or ``stages_completed``. Derive these fields from the
    # structured transitions instead of silently reporting zero stages.
    transitions = sorted(
        transitions,
        key=lambda r: float(r.get("elapsed_s") or 0.0),
    )
    route = [str(tr["to_stage"]) for tr in transitions if tr.get("to_stage")]

    # Older logs may not contain structured transitions. As a fallback, try
    # the state-level stage fields, then the session's combination->stage map.
    if not route:
        route = []
        known_stages = (session_start or {}).get("known_stages") or {}
        for state in states:
            stage = (state.get("known_stage") or state.get("confirmed_stage")
                     or known_stages.get(state.get("combination_id")))
            if stage and (not route or route[-1] != stage):
                route.append(str(stage))

    # A session killed with the window X rather than 'q' has no session_end.
    # Reconstruct what can be reconstructed rather than discarding the run -
    # but mark it, because an unterminated log is weaker evidence and the
    # reader should be told which rows those are.
    reconstructed = summary is None
    if reconstructed:
        last_t = max([s.get("ended_s") or s.get("started_s") or 0.0
                      for s in states] + [0.0])
        summary = {
            "session_id": path.stem,
            "total_time_s": last_t,
            "build_time_s": None,
            "states_observed": len(states),
            "warning_count": len(warnings),
            "stage_transition_count": len(transitions),
            "stage_route": route,
            "stages_completed": len(route),
            "goal_reached": bool(route and route[-1] == EXPECTED_SEQUENCE[-1]),
            "final_product_verified": bool(verification
                                           and verification.get("verified")),
        }
    else:
        # Normalize the different logger versions to one analysis schema.
        # Keep each logged transition in order: duplicates or skipped stages
        # must remain visible rather than being silently deduplicated.
        summary.setdefault("stage_route", route)
        summary.setdefault("stages_completed", len(route))
        # Stage_07 transition is evidence of a reported stage transition, not
        # proof that the final physical component combination was observed.
        if route and "Stage_07" in route:
            summary["goal_reached"] = True
        elif "goal_reached" not in summary:
            summary["goal_reached"] = bool(summary.get("reached_goal", False))
        # Preserve the logger's separate final-combination evidence where it
        # exists. This matters because a Stage_07 transition and goal-combo
        # observation may disagree in a real session.
        summary.setdefault(
            "goal_combination_observed",
            summary.get("goal_observed", summary.get("reached_goal")),
        )
        summary.setdefault(
            "final_product_verified",
            bool(verification and verification.get("verified")),
        )

    return {
        "file": path.name,
        "path": str(path),
        "states": states,
        "transitions": transitions,
        "warnings": warnings,
        "status_changes": status_changes,
        "summary": summary,
        "verification": verification,
        "session_start": session_start,
        "reconstructed": reconstructed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Ground truth
# ═══════════════════════════════════════════════════════════════════════════

TRIAL_COLUMNS = [
    "log_file", "trial_id", "date", "operator", "lighting",
    "condition", "true_stages_completed", "true_stage_route",
    "error_introduced_at_stage", "error_elapsed_s", "notes",
]

TRIAL_HELP = """# trials.csv - INDEPENDENT GROUND TRUTH. Complete one row per validation run.
#
# log_file                   JSONL filename (pre-filled; do not change)
# trial_id                   e.g. T2-01
# date                       DD.MM.YYYY
# operator                   initials or anonymised code
# lighting                   describe actual lighting (normal/bright/dim + note)
# condition                  nominal | induced_wrong_part | induced_wrong_order | induced_wrong_position
# true_stages_completed      number of stages actually completed correctly, 0-7
# true_stage_route           actual stage route, e.g. Stage_01>Stage_02>Stage_03
#                            leave blank only if you did not record the route
# error_introduced_at_stage  stage at which the deliberate error was introduced
# error_elapsed_s            seconds from session start when the error was introduced;
#                            required to calculate response latency and to score a
#                            warning as occurring after the introduced error
# notes                      deviations, camera movement, obstruction, etc.
#
# IMPORTANT: condition and true_stages_completed are intentionally BLANK in the
# template. Do not report accuracy until you fill them from your own observation.
# Do not label wrong-position trials as successful if position validation was disabled
# or if the system displayed BODY NOT TRACKED / POSITION CHECK PAUSED.
"""


def write_trial_template(out_dir: Path, sessions: list) -> Path:
    path = out_dir / "trials_TEMPLATE.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        fh.write(TRIAL_HELP + "\n")
        writer = csv.writer(fh)
        writer.writerow(TRIAL_COLUMNS)
        for i, session in enumerate(sessions, 1):
            # Never pre-fill ground truth: doing so can silently create false accuracy.
            writer.writerow([session["file"], f"T2-{i:02d}", "", "", "", "", "", "", "", "", ""])
    return path


def read_trials(path: Path) -> dict:
    """Read CSV safely, including quoted notes containing commas."""
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8-sig").splitlines()
    content = [line for line in content if line.strip() and not line.lstrip().startswith("#")]
    if not content:
        return {}
    reader = csv.DictReader(io.StringIO("\n".join(content)))
    rows = {}
    for raw in reader:
        row = {str(k).strip(): (v.strip() if isinstance(v, str) else "")
               for k, v in raw.items() if k is not None}
        filename = row.get("log_file", "")
        if filename:
            rows[filename] = row
    return rows



# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def stage_durations(session: dict) -> list:
    """Time spent working on each stage = gap between confirmations.

    The first stage is measured from the first non-empty observed state,
    because the clock on Stage_01 starts when the operator first puts
    something on the bench, not when the program launched.
    """
    out = []
    trs = sorted(session["transitions"], key=lambda r: r.get("elapsed_s") or 0.0)
    if not trs:
        return out

    first_build = next((s for s in session["states"]
                        if s.get("combination_id") != "empty"), None)
    prev_t = float(first_build["started_s"]) if first_build else 0.0

    for tr in trs:
        to_stage = tr.get("to_stage")
        t = float(tr.get("elapsed_s") or 0.0)
        if not to_stage:
            continue
        out.append({
            "stage": to_stage,
            "started_at_s": round(prev_t, 2),
            "confirmed_at_s": round(t, 2),
            "duration_s": round(max(0.0, t - prev_t), 2),
            "from_stage": tr.get("from_stage"),
            "transition_type": tr.get("transition_type", "STANDARD"),
        })
        prev_t = t
    return out


def sequence_is_correct(route: list) -> bool:
    """Did a nominal trial report the complete taught sequence in order?

    A correct prefix alone is not a successful full-build sequence. Stage-count
    agreement is reported separately, so partial runs cannot earn a sequence
    success merely because their observed prefix was in order.
    """
    return route == EXPECTED_SEQUENCE


def parse_optional_float(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (TypeError, ValueError):
        return None


def parse_stage_route(value):
    """Parse an operator-recorded route separated by >, ;, or |."""
    if not value or not str(value).strip():
        return None
    import re
    parts = [part.strip() for part in re.split(r">|;|\|", str(value)) if part.strip()]
    return parts or None


def error_response_time(session: dict, warning_type: str, error_elapsed_s):
    """Elapsed time from recorded error injection to the first matching later warning.

    Returns None if the injection time was not independently recorded. It deliberately
    does not infer injection time from a status change because that would understate
    or otherwise misrepresent the actual response latency.
    """
    if error_elapsed_s is None:
        return None
    matching = [
        parse_optional_float(w.get("elapsed_s"))
        for w in session["warnings"]
        if w.get("warning_type") == warning_type
    ]
    later = [t for t in matching if t is not None and t >= error_elapsed_s]
    if not later:
        return None
    return round(min(later) - error_elapsed_s, 2)



def analyse(sessions: list, trials: dict) -> dict:
    rows = []
    all_stage_rows = []
    all_error_rows = []

    for s in sessions:
        summ = s["summary"]
        trial = trials.get(s["file"], {})
        route = summ.get("stage_route") or []
        durations = stage_durations(s)

        wcounts = Counter(w.get("warning_type", "UNKNOWN")
                          for w in s["warnings"])

        condition = (trial.get("condition") or "unclassified").strip()
        true_stages_raw = trial.get("true_stages_completed")
        try:
            true_stages = int(true_stages_raw) if true_stages_raw not in (None, "") else None
            if true_stages is not None and not 0 <= true_stages <= 7:
                log(f"WARNING: {s['file']}: true_stages_completed must be 0..7; excluded from stage-count scoring")
                true_stages = None
        except (TypeError, ValueError):
            true_stages = None
        true_route = parse_stage_route(trial.get("true_stage_route"))
        error_elapsed_s = parse_optional_float(trial.get("error_elapsed_s"))

        reported = int(summ.get("stages_completed") or 0)
        stage_agreement = (None if true_stages is None
                           else reported == true_stages)
        over_report = (None if true_stages is None
                       else max(0, reported - true_stages))
        under_report = (None if true_stages is None
                        else max(0, true_stages - reported))

        expected_w = EXPECTED_WARNING.get(condition)
        expected_warning_times = []
        if expected_w:
            expected_warning_times = [
                parse_optional_float(w.get("elapsed_s"))
                for w in s["warnings"] if w.get("warning_type") == expected_w
            ]
            expected_warning_times = [t for t in expected_warning_times if t is not None]
        # Score error detection only if the injection time is recorded, so a warning
        # emitted before the deliberate error cannot be counted as detecting it.
        error_detected = (
            None if expected_w is None or error_elapsed_s is None
            else any(t >= error_elapsed_s for t in expected_warning_times)
        )
        response_s = error_response_time(s, expected_w, error_elapsed_s) if expected_w else None

        if true_route is not None:
            sequence_correct = route == true_route
        elif condition == CONDITION_NOMINAL:
            sequence_correct = sequence_is_correct(route)
        else:
            sequence_correct = None

        row = {
            "trial_id": trial.get("trial_id") or s["file"],
            "log_file": s["file"],
            "date": trial.get("date", ""),
            "operator": trial.get("operator", ""),
            "lighting": trial.get("lighting", ""),
            "condition": condition,
            "total_time_s": round(float(summ.get("total_time_s") or 0.0), 2),
            "build_time_s": (round(float(summ["build_time_s"]), 2)
                             if summ.get("build_time_s") is not None and summ.get("build_time_s") != ""
                             else None),
            "states_observed": summ.get("states_observed"),
            "stages_reported": reported,
            "stages_true": true_stages,
            "true_stage_route": " > ".join(true_route) if true_route else "",
            "stage_count_agrees": stage_agreement,
            "over_reported": over_report,
            "under_reported": under_report,
            "sequence_correct": sequence_correct,
            "route": " > ".join(route),
            "error_elapsed_s": error_elapsed_s,
            "goal_reached": bool(summ.get("goal_reached")),
            "goal_combination_observed": summ.get("goal_combination_observed"),
            "final_verified": bool(summ.get("final_product_verified")),
            "warnings_total": len(s["warnings"]),
            "wrong_part": wcounts.get("WRONG_PART", 0),
            "wrong_order": wcounts.get("WRONG_ORDER", 0),
            "wrong_position": wcounts.get("WRONG_POSITION", 0),
            "expected_warning": expected_w or "",
            "induced_error_detected": error_detected,
            "response_time_s": response_s,
            "log_complete": not s["reconstructed"],
            "notes": trial.get("notes", ""),
        }
        rows.append(row)

        for d in durations:
            all_stage_rows.append({"trial_id": row["trial_id"], **d})
        for w in s["warnings"]:
            all_error_rows.append({
                "trial_id": row["trial_id"],
                "elapsed_s": round(float(w.get("elapsed_s") or 0.0), 2),
                "warning_type": w.get("warning_type"),
                "current_stage": w.get("current_stage"),
                "message": (w.get("message") or "").replace(",", ";"),
                "observed": combination_str(w.get("observed")),
            })

    return {
        "sessions": rows,
        "stages": all_stage_rows,
        "errors": all_error_rows,
        "aggregate": aggregate(rows, all_stage_rows),
    }


def combination_str(counts) -> str:
    if not counts:
        return ""
    return "; ".join(f"{k} x{v}" for k, v in sorted(counts.items()))


def aggregate(rows: list, stage_rows: list) -> dict:
    # Incomplete sessions can be reconstructed for inspection, but they are
    # excluded from aggregate performance metrics because their termination
    # and final state were not logged normally (for example, window closed via X).
    nominal_all = [r for r in rows if r["condition"] == CONDITION_NOMINAL]
    induced_all = [r for r in rows if r["condition"] in EXPECTED_WARNING]
    nominal = [r for r in nominal_all if r["log_complete"]]
    induced = [r for r in induced_all if r["log_complete"]]

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.mean(vals), 2) if vals else None

    def median(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.median(vals), 2) if vals else None

    def sd(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.stdev(vals), 2) if len(vals) > 1 else None

    # Per-stage timing across the nominal runs only. Including a run where
    # the operator deliberately did the wrong thing would inflate the
    # timings with the time spent correcting it.
    nominal_ids = {r["trial_id"] for r in nominal}
    by_stage = defaultdict(list)
    for s in stage_rows:
        if s["trial_id"] in nominal_ids:
            by_stage[s["stage"]].append(s["duration_s"])

    stage_stats = {}
    for stage in EXPECTED_SEQUENCE:
        vals = by_stage.get(stage, [])
        stage_stats[stage] = {
            "n": len(vals),
            "mean_s": mean(vals),
            "median_s": median(vals),
            "sd_s": sd(vals),
            "min_s": round(min(vals), 2) if vals else None,
            "max_s": round(max(vals), 2) if vals else None,
        }

    checked = [r for r in nominal if r["stage_count_agrees"] is not None]
    agree = [r for r in checked if r["stage_count_agrees"]]
    seq_ok = [r for r in nominal if r["sequence_correct"]]

    det = [r for r in induced if r["induced_error_detected"] is not None]
    det_ok = [r for r in det if r["induced_error_detected"]]

    by_condition = {}
    for cond in EXPECTED_WARNING:
        sub = [r for r in induced if r["condition"] == cond]
        scored = [r for r in sub if r["induced_error_detected"] is not None]
        hit = [r for r in scored if r["induced_error_detected"]]
        by_condition[cond] = {
            "trials": len(sub),
            "scored_trials": len(scored),
            "unscored_missing_injection_time": len(sub) - len(scored),
            "detected": len(hit),
            "detection_rate": (round(len(hit) / len(scored), 4) if scored else None),
            "mean_response_s": mean([r["response_time_s"] for r in scored]),
        }

    # This is a raw count of warning records in nominal trials. It is not
    # automatically equal to false-alarm trials unless every warning type is
    # defined as an error warning in the logger's schema.
    nominal_warning_events = sum(r["warnings_total"] for r in nominal)
    nominal_trials_with_warnings = sum(1 for r in nominal if r["warnings_total"] > 0)
    labelled_gt_trials = sum(1 for r in rows
                             if r["condition"] != "unclassified"
                             and r["stages_true"] is not None
                             and r["log_complete"])

    return {
        "trials_total": len(rows),
        "trials_nominal": len(nominal_all),
        "trials_induced": len(induced_all),
        "trials_nominal_complete": len(nominal),
        "trials_induced_complete": len(induced),
        "logs_incomplete": sum(1 for r in rows if not r["log_complete"]),
        "ground_truth_rows_scored": labelled_gt_trials,

        "stage_count_checked": len(checked),
        "stage_count_agree": len(agree),
        "stage_count_agreement_rate": (round(len(agree) / len(checked), 4)
                                       if checked else None),
        "sequence_trials_scored": len([r for r in nominal if r["sequence_correct"] is not None]),
        "sequence_correct_rate": (round(sum(1 for r in nominal if r["sequence_correct"] is True) /
                                         len([r for r in nominal if r["sequence_correct"] is not None]), 4)
                                  if any(r["sequence_correct"] is not None for r in nominal) else None),
        "goal_reached_rate": (round(sum(1 for r in nominal if r["goal_reached"])
                                    / len(nominal), 4) if nominal else None),

        "build_time_mean_s": mean([r["build_time_s"] for r in nominal]),
        "build_time_median_s": median([r["build_time_s"] for r in nominal]),
        "build_time_sd_s": sd([r["build_time_s"] for r in nominal]),
        "build_time_min_s": (min([r["build_time_s"] for r in nominal
                                  if r["build_time_s"] is not None], default=None)),
        "build_time_max_s": (max([r["build_time_s"] for r in nominal
                                  if r["build_time_s"] is not None], default=None)),

        "stage_stats": stage_stats,

        "induced_checked": len(det),
        "induced_detected": len(det_ok),
        "error_detection_rate": (round(len(det_ok) / len(det), 4)
                                 if det else None),
        "mean_response_s": mean([r["response_time_s"] for r in det_ok]),
        "by_condition": by_condition,
        "nominal_warning_events": nominal_warning_events,
        "nominal_trials_with_warnings": nominal_trials_with_warnings,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def save(fig, out_dir: Path, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    return out_dir / f"{name}.pdf"


def fig_stage_timeline(data, out_dir):
    """When each stage was confirmed, per trial. Shows the process shape."""
    rows = [r for r in data["sessions"] if r["condition"] == CONDITION_NOMINAL]
    if not rows:
        return None
    by_trial = defaultdict(list)
    for s in data["stages"]:
        by_trial[s["trial_id"]].append(s)

    trials = [r["trial_id"] for r in rows if by_trial.get(r["trial_id"])]
    if not trials:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 0.5 * len(trials) + 1.6))
    cmap = plt.get_cmap("Blues")

    for y, tid in enumerate(trials):
        stages = sorted(by_trial[tid], key=lambda s: s["confirmed_at_s"])
        valid_starts = [s.get("started_at_s") for s in stages
                        if s.get("started_at_s") is not None]
        origin = min(valid_starts) if valid_starts else 0.0
        for s in stages:
            idx = (EXPECTED_SEQUENCE.index(s["stage"])
                   if s["stage"] in EXPECTED_SEQUENCE else 0)
            left = max(0.0, float(s.get("started_at_s") or 0.0) - origin)
            ax.barh(y, s["duration_s"], left=left, height=0.6,
                    color=cmap(0.30 + 0.09 * idx), edgecolor="white",
                    linewidth=0.8)
            if s["duration_s"] > 4:
                ax.text(left + s["duration_s"] / 2, y, str(idx + 1),
                        ha="center", va="center", fontsize=7, color="white")

    ax.set_yticks(range(len(trials)))
    ax.set_yticklabels(trials)
    ax.invert_yaxis()
    ax.set_xlabel("time from first stable non-empty state in the log (s)")
    ax.set_title("Stage-confirmation timeline, nominal trials")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(facecolor=cmap(0.55), label="one assembly stage")],
              loc="lower right", frameon=False, fontsize=8)
    return save(fig, out_dir, "fig_stage_timeline")


def fig_stage_durations(data, out_dir):
    """Mean time per stage with spread. The core timing result."""
    st = data["aggregate"]["stage_stats"]
    stages = [s for s in EXPECTED_SEQUENCE if st[s]["n"] > 0]
    if not stages:
        return None

    means = [st[s]["mean_s"] for s in stages]
    sds = [st[s]["sd_s"] or 0.0 for s in stages]
    ns = [st[s]["n"] for s in stages]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = range(len(stages))
    ax.bar(x, means, yerr=sds, capsize=3, color=C_OK, width=0.62,
           error_kw={"ecolor": C_GREY, "elinewidth": 1})
    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m + (max(means) * 0.03), f"{m:.1f}",
                ha="center", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels([s.replace("Stage_", "") for s in stages])
    ax.set_xlabel(f"assembly stage   (n = {ns[0]} trials)"
                  if len(set(ns)) == 1 else "assembly stage")
    ax.set_ylabel("time to complete (s)")
    ax.set_title("Mean time per assembly stage, nominal trials")
    ax.grid(axis="x", visible=False)
    return save(fig, out_dir, "fig_stage_durations")


def fig_error_detection(data, out_dir):
    """Induced errors against errors detected, by error type."""
    bc = data["aggregate"]["by_condition"]
    conds = [c for c in EXPECTED_WARNING if bc[c]["trials"] > 0]
    if not conds:
        return None

    labels = [c.replace("induced_", "").replace("_", " ") for c in conds]
    induced = [bc[c]["scored_trials"] for c in conds]
    detected = [bc[c]["detected"] for c in conds]
    unscored = [bc[c]["unscored_missing_injection_time"] for c in conds]

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    x = range(len(conds))
    w = 0.36
    ax.bar([i - w / 2 for i in x], induced, w, label="scored trials",
           color=C_LIGHT, edgecolor=C_GREY, linewidth=0.8)
    ax.bar([i + w / 2 for i in x], detected, w, label="detected",
           color=C_OK)
    for i, (a, b, u) in enumerate(zip(induced, detected, unscored)):
        ax.text(i - w / 2, a + 0.08, str(a), ha="center", fontsize=8,
                color=C_GREY)
        ax.text(i + w / 2, b + 0.08, str(b), ha="center", fontsize=8,
                color=C_OK)
        if b < a:
            ax.text(i, max(a, b) + 0.5, f"{a - b} missed", ha="center",
                    fontsize=7.5, color=C_BAD)
        if u:
            ax.text(i, max(a, b) + (0.95 if b < a else 0.45),
                    f"{u} unscored", ha="center", fontsize=7.5,
                    color=C_WARN)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("number of trials")
    ax.set_title("Error-warning detection among scored trials")
    ax.set_ylim(0, max(max(induced, default=0), max(detected, default=0), 1) * 1.8 + 0.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="x", visible=False)
    return save(fig, out_dir, "fig_error_detection")


def fig_build_times(data, out_dir):
    """Total build time per nominal trial, with the mean marked."""
    rows = [r for r in data["sessions"]
            if r["condition"] == CONDITION_NOMINAL
            and r["build_time_s"] is not None]
    if len(rows) < 2:
        return None

    ids = [r["trial_id"] for r in rows]
    vals = [r["build_time_s"] for r in rows]
    m = statistics.mean(vals)

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.bar(range(len(vals)), vals, color=C_OK, width=0.6)
    ax.axhline(m, color=C_WARN, linewidth=1.2, linestyle="--")
    ax.text(len(vals) - 0.4, m, f"  mean {m:.1f} s", va="center",
            fontsize=8, color=C_WARN)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"{v:.0f}", ha="center", fontsize=8)

    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=0)
    ax.set_ylabel("build time (s)")
    ax.set_title("Complete assembly build time per nominal trial")
    ax.grid(axis="x", visible=False)
    return save(fig, out_dir, "fig_build_times")


# ═══════════════════════════════════════════════════════════════════════════
# LaTeX
# ═══════════════════════════════════════════════════════════════════════════

def tex_escape(s) -> str:
    s = "" if s is None else str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def num(v, dp=2, dash="--"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return dash
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{dp}f}"
    return str(v)


def pct(v, dash="--"):
    return dash if v is None else f"{v * 100:.1f}\\,\\%"


def write_tex_tables(data, out_dir: Path) -> list:
    """Write four LaTeX tables; all files are placed directly in out_dir."""
    agg = data["aggregate"]
    written = []
    endrow = chr(92) * 2
    hline = chr(92) + "hline"

    # Table 1: summary
    p = out_dir / "tab_v21_summary.tex"
    L = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Summary of the V2.1 validation trials.}",
        r"\label{tab:v21_summary}",
        r"\renewcommand{\arraystretch}{1.4}",
        r"\setlength{\tabcolsep}{8pt}", "",
        r"\begin{tabular}{|p{7.5cm}|p{4.0cm}|}", hline,
        r"\textbf{Measure} & \textbf{Recorded value} " + endrow, hline,
        f"Trials recorded & {agg['trials_total']} " + endrow, hline,
        f"Nominal trials recorded & {agg['trials_nominal']} " + endrow, hline,
        f"Complete nominal trials used for metrics & {agg['trials_nominal_complete']} " + endrow, hline,
        f"Error-injection trials recorded & {agg['trials_induced']} " + endrow, hline,
        f"Complete error-injection trials used for metrics & {agg['trials_induced_complete']} " + endrow, hline,
        (f"Nominal stage-count agreement & {agg['stage_count_agree']} of "
         f"{agg['stage_count_checked']} ({pct(agg['stage_count_agreement_rate'])}) " + endrow), hline,
        (f"Nominal stage-sequence agreement & {agg['sequence_trials_scored']} scored; "
         f"{pct(agg['sequence_correct_rate'])} correct " + endrow), hline,
        f"Stage\\_07 transition observed in nominal trials & {pct(agg['goal_reached_rate'])} " + endrow, hline,
        f"Mean nominal build time & {num(agg['build_time_mean_s'])}\\,s " + endrow, hline,
        f"Median nominal build time & {num(agg['build_time_median_s'])}\\,s " + endrow, hline,
        (f"Introduced errors detected (scored trials) & {agg['induced_detected']} of "
         f"{agg['induced_checked']} ({pct(agg['error_detection_rate'])}) " + endrow), hline,
        f"Warning records in nominal trials & {agg['nominal_warning_events']} " + endrow, hline,
        f"Nominal trials with at least one warning & {agg['nominal_trials_with_warnings']} " + endrow, hline,
        r"\end{tabular}", r"\end{table}",
    ]
    p.write_text("\n".join(L), encoding="utf-8")
    written.append(p)

    # Table 2: individual trials
    p = out_dir / "tab_v21_trials.tex"
    L = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Individual validation-trial outcomes.}",
        r"\label{tab:v21_trials}",
        r"\renewcommand{\arraystretch}{1.3}",
        r"\setlength{\tabcolsep}{5pt}", r"\footnotesize", "",
        r"\begin{tabular}{|l|l|c|c|c|c|c|}", hline,
        (r"\textbf{Trial} & \textbf{Condition} & \textbf{Reported} & "
         r"\textbf{True} & \textbf{Seq.} & \textbf{Build (s)} & "
         r"\textbf{Warnings} " + endrow), hline,
    ]
    for row in data["sessions"]:
        seq = "--" if row["sequence_correct"] is None else ("yes" if row["sequence_correct"] else "no")
        L.append(
            f"{tex_escape(row['trial_id'])} & "
            f"{tex_escape(row['condition'].replace('induced_', ''))} & "
            f"{row['stages_reported']} & {num(row['stages_true'], 0)} & "
            f"{seq} & {num(row['build_time_s'], 1)} & {row['warnings_total']} " + endrow)
        L.append(hline)
    L += [r"\end{tabular}", r"\end{table}"]
    p.write_text("\n".join(L), encoding="utf-8")
    written.append(p)

    # Table 3: stage timing (descriptive system-log timings, not timing accuracy)
    p = out_dir / "tab_v21_stage_timing.tex"
    L = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Stage-transition interval durations in nominal trials.}",
        r"\label{tab:v21_stage_timing}",
        r"\renewcommand{\arraystretch}{1.3}",
        r"\setlength{\tabcolsep}{8pt}", "",
        r"\begin{tabular}{|l|c|c|c|c|c|}", hline,
        (r"\textbf{Stage} & \textbf{n} & \textbf{Mean (s)} & "
         r"\textbf{Median (s)} & \textbf{Min (s)} & \textbf{Max (s)} " + endrow),
        hline,
    ]
    for stage in EXPECTED_SEQUENCE:
        st = agg["stage_stats"][stage]
        if not st["n"]:
            continue
        L.append(
            f"{tex_escape(stage)} & {st['n']} & {num(st['mean_s'])} & "
            f"{num(st['median_s'])} & {num(st['min_s'])} & {num(st['max_s'])} " + endrow)
        L.append(hline)
    L += [r"\end{tabular}", r"\end{table}"]
    p.write_text("\n".join(L), encoding="utf-8")
    written.append(p)

    # Table 4: error handling. Denominator includes only trials with an injection timestamp.
    p = out_dir / "tab_v21_error_handling.tex"
    L = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Detection of deliberately introduced assembly errors.}",
        r"\label{tab:v21_error_handling}",
        r"\renewcommand{\arraystretch}{1.3}",
        r"\setlength{\tabcolsep}{6pt}", "",
        r"\footnotesize",
        r"\begin{tabular}{|l|c|c|c|c|c|}", hline,
        (r"\textbf{Error type} & \textbf{Total} & \textbf{Scored} & "
         r"\textbf{Detected} & \textbf{Rate (scored)} & "
         r"\textbf{Mean latency (s)} " + endrow), hline,
    ]
    for condition, stats in agg["by_condition"].items():
        if not stats["trials"]:
            continue
        L.append(
            f"{tex_escape(condition.replace('induced_', '').replace('_', ' '))} & "
            f"{stats['trials']} & {stats['scored_trials']} & {stats['detected']} & "
            f"{pct(stats['detection_rate'])} & {num(stats['mean_response_s'])} " + endrow)
        L.append(hline)
    L += [r"\end{tabular}", r"\end{table}"]
    p.write_text("\n".join(L), encoding="utf-8")
    written.append(p)
    return written



# ═══════════════════════════════════════════════════════════════════════════
# CSV + report
# ═══════════════════════════════════════════════════════════════════════════

def write_csv(path: Path, rows: list, columns=None):
    columns = columns or (list(rows[0].keys()) if rows else [])
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})



def write_report(data, out_dir: Path, figures: list) -> Path:
    agg = data["aggregate"]
    p = out_dir / "v21_validation_report.txt"
    L = []
    A = L.append
    A("=" * 78)
    A("V2.1 SYSTEM VALIDATION  -  ANALYSIS OUTPUT")
    A("=" * 78)
    A("")
    A(f"  Generated      {datetime.now().isoformat(timespec='seconds')}")
    A(f"  Trials         {agg['trials_total']} "
      f"({agg['trials_nominal']} nominal, {agg['trials_induced']} error-injection)")
    if agg["logs_incomplete"]:
        A(f"  WARNING        {agg['logs_incomplete']} log(s) have no session_end "
          f"record. Those runs were closed without pressing 'q'; they remain in "
          f"the per-trial CSV for inspection but are excluded from aggregate metrics. "
          f"Re-run them if possible.")
    A(f"  Complete logs used for nominal metrics: {agg['trials_nominal_complete']} of {agg['trials_nominal']} nominal trials.")
    A(f"  Complete logs used for error metrics: {agg['trials_induced_complete']} of {agg['trials_induced']} error-injection trials.")
    A("")
    A("WHAT IS BEING MEASURED, AND AGAINST WHAT")
    A(f"  Independent ground-truth rows scored: {agg['ground_truth_rows_scored']} of {agg['trials_total']} logs.")
    if agg["ground_truth_rows_scored"] == agg["trials_total"] and agg["trials_total"] > 0:
        A("  Stage counts and induced-error outcomes are compared with the")
        A("  operator-recorded ground truth. Metrics below are limited to the")
        A("  outcomes for which required ground-truth fields were recorded.")
    else:
        A("  WARNING: ground truth is missing or incomplete for one or more logs.")
        A("  Do not describe aggregate outputs as accuracy until the trials.csv")
        A("  fields are completed and the report is regenerated.")
    A("")
    A("1  STAGE RECOGNITION")
    A(f"  stage count agreement    {agg['stage_count_agree']} of "
      f"{agg['stage_count_checked']}  ({_p(agg['stage_count_agreement_rate'])})")
    A(f"  correct stage sequence   {_p(agg['sequence_correct_rate'])}")
    A(f"  Stage_07 transition seen  {_p(agg['goal_reached_rate'])}")
    combo_scored = [r for r in data["sessions"]
                    if r.get("goal_combination_observed") is not None]
    combo_seen = sum(1 for r in combo_scored
                     if r.get("goal_combination_observed") is True)
    A(f"  final component combination observed {combo_seen} of {len(combo_scored)} logs with this field")
    over = sum(r["over_reported"] or 0 for r in data["sessions"])
    under = sum(r["under_reported"] or 0 for r in data["sessions"])
    A(f"  stages over-reported     {over}   (system claimed a stage that was "
      f"not completed)")
    A(f"  stages under-reported    {under}   (operator completed a stage the "
      f"system did not confirm)")
    A("")
    A("  These two errors are not equivalent and should not be averaged.")
    A("  An over-report is a false confirmation: the system certifies work")
    A("  that was not done, and the confirmed baseline carries that mistake")
    A("  into every later stage. An under-report only delays the operator.")
    A("")
    A("2  PROCESS TIMING")
    A(f"  build time  mean {_n(agg['build_time_mean_s'])} s   "
      f"median {_n(agg['build_time_median_s'])} s   "
      f"sd {_n(agg['build_time_sd_s'])} s")
    A(f"              min  {_n(agg['build_time_min_s'])} s   "
      f"max {_n(agg['build_time_max_s'])} s")
    A("")
    A("  per stage (nominal trials only):")
    A(f"    {'stage':<10} {'n':>3} {'mean':>8} {'median':>8} {'min':>8} {'max':>8}")
    for stage in EXPECTED_SEQUENCE:
        st = agg["stage_stats"][stage]
        if not st["n"]:
            continue
        A(f"    {stage:<10} {st['n']:>3} {_n(st['mean_s']):>8} "
          f"{_n(st['median_s']):>8} {_n(st['min_s']):>8} {_n(st['max_s']):>8}")
    A("")
    A("3  ERROR HANDLING")
    A(f"  introduced errors detected   {agg['induced_detected']} of "
      f"{agg['induced_checked']}  ({_p(agg['error_detection_rate'])})")
    A(f"  mean warning latency among detected, time-stamped trials {_n(agg['mean_response_s'])} s")
    A("  (trials without error_elapsed_s or without the expected warning are not in the latency mean)")
    A(f"  warning records in nominal runs {agg['nominal_warning_events']}")
    A(f"  nominal trials with >=1 warning {agg['nominal_trials_with_warnings']}")
    A("  (warning records are not automatically false alarms; inspect warning types)")
    A("")
    for cond, st in agg["by_condition"].items():
        if not st["trials"]:
            continue
        A(f"    {cond.replace('induced_', ''):<16} "
          f"{st['detected']}/{st['scored_trials']} scored "
          f"(total {st['trials']}, unscored {st['unscored_missing_injection_time']}); "
          f"{_p(st['detection_rate'])}; latency {_n(st['mean_response_s'])} s")
    A("")
    A("4  WHAT THESE NUMBERS DO AND DO NOT COVER")
    A("  They describe this system, on this assembly, on this bench, under")
    A("  the lighting recorded in trials.csv, operated by the people listed")
    A("  there. They are a measurement of the complete pipeline end to end -")
    A("  detection, state machine, and the operator-facing output together -")
    A("  not of the YOLO detector in isolation.")
    A("")
    A("  The trial count is small. With single-digit trials a rate such as")
    A("  100 % means 'no failure was observed in n attempts', not 'failure")
    A("  cannot occur'. Report n alongside every percentage.")
    A("")
    A("  Position validation is a separate claim. Real-bench wrong-hole trials")
    A("  can only be scored if position validation was enabled, body tracking")
    A("  remained valid, and each actual hole placement was independently")
    A("  recorded. Trials run while validation was disabled or paused do not")
    A("  validate wrong-hole detection. Synthetic detector metrics and the")
    A("  synthetic POS-FEAS-01 observability study are separate from real-bench")
    A("  end-to-end performance.")
    A("")
    A("FILES")
    A("  v21_session_summary.csv   one row per trial")
    A("  v21_stage_timing.csv      every stage transition and its duration")
    A("  v21_error_events.csv      every warning raised")
    A("  v21_metrics.json          everything, machine-readable")
    for f in figures:
        if f:
            A(f"  {f.name}")
    for t in sorted(out_dir.glob("tab_*.tex")):
        A(f"  {t.name}")
    A("=" * 78)
    p.write_text("\n".join(L), encoding="utf-8")
    return p


def _n(v, dp=2):
    return "--" if v is None else f"{v:.{dp}f}"


def _p(v):
    return "--" if v is None else f"{v * 100:.1f}%"


# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/process",
                    help="folder holding the *.jsonl process logs")
    ap.add_argument("--out", default="validation_output")
    ap.add_argument("--trials", default=None,
                    help="ground-truth csv (default <out>/trials.csv)")
    args = ap.parse_args()

    log_dir = Path(args.logs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = Path(args.trials) if args.trials else out_dir / "trials.csv"

    log("=" * 60)
    log("V2.1 VALIDATION ANALYSIS")
    log("=" * 60)

    if not log_dir.exists():
        log(f"ERROR: no log folder at {log_dir.resolve()}")
        log("Run main.py at least once, then try again.")
        return 1

    files = sorted(log_dir.glob("*.jsonl"))
    if not files:
        log(f"ERROR: no .jsonl files in {log_dir.resolve()}")
        return 1
    log(f"found {len(files)} log file(s)")
    log("NOTE: every JSONL directly inside the selected log folder is analyzed.")
    log("Archive older/demo logs outside this folder before collecting validation trials.")

    sessions = [parse_session(f) for f in files]
    for s in sessions:
        flag = "  (no session_end - closed without 'q')" if s["reconstructed"] else ""
        log(f"  {s['file']}: {len(s['states'])} states, "
            f"{len(s['transitions'])} transitions, "
            f"{len(s['warnings'])} warnings{flag}")

    trials = read_trials(trials_path)
    if not trials:
        tmpl = write_trial_template(out_dir, sessions)
        log("")
        log("NO GROUND TRUTH YET.")
        log(f"  A template has been written to: {tmpl}")
        log("  Fill in the condition and true_stages_completed columns,")
        log(f"  save it as {trials_path}, and run this script again.")
        log("")
        log("  Without it the script can only describe what the system")
        log("  reported. With it, the system can be checked against what")
        log("  actually happened - which is the whole point of a validation")
        log("  chapter. The V1 chapter had to admit it could not do this.")
        log("")
        log("  Continuing anyway with timing-only output.")
    else:
        log(f"ground truth: {len(trials)} trial(s) from {trials_path.name}")
        missing = [s["file"] for s in sessions if s["file"] not in trials]
        if missing:
            log(f"  NOTE: {len(missing)} log(s) have no ground-truth row and "
                f"are reported as 'unclassified':")
            for m in missing[:5]:
                log(f"    {m}")

    data = analyse(sessions, trials)

    write_csv(out_dir / "v21_session_summary.csv", data["sessions"])
    write_csv(out_dir / "v21_stage_timing.csv", data["stages"])
    write_csv(out_dir / "v21_error_events.csv", data["errors"])
    (out_dir / "v21_metrics.json").write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8")

    figures = [
        fig_stage_timeline(data, out_dir),
        fig_stage_durations(data, out_dir),
        fig_build_times(data, out_dir),
        fig_error_detection(data, out_dir),
    ]
    tables = write_tex_tables(data, out_dir)
    report = write_report(data, out_dir, figures)

    agg = data["aggregate"]
    log("")
    log("=" * 60)
    log("RESULTS")
    log("=" * 60)
    log(f"  trials                  {agg['trials_total']} "
        f"({agg['trials_nominal']} nominal, {agg['trials_induced']} induced)")
    log(f"  stage count agreement   {_p(agg['stage_count_agreement_rate'])}")
    log(f"  correct sequence        {_p(agg['sequence_correct_rate'])}")
    log(f"  build time mean         {_n(agg['build_time_mean_s'])} s")
    log(f"  errors detected         {_p(agg['error_detection_rate'])}")
    log(f"  warnings in nominal     {agg['nominal_warning_events']} records across {agg['nominal_trials_with_warnings']} trials")
    log("")
    log(f"  report   {report}")
    log(f"  figures  {sum(1 for f in figures if f)} written")
    log(f"  tables   {len(tables)} .tex files")
    log(f"  folder   {out_dir.resolve()}")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
