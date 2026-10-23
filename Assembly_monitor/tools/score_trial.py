"""
score_trial.py
========================================================================
SCORE A LIVE ASSEMBLY TRIAL AGAINST AN INDEPENDENT GROUND TRUTH.

WHY THIS MATTERS FOR THE THESIS
-------------------------------
Table 6.10 says "unscored" nine times. The reason is always the same: the
system's own output was the only record, and Section 5.10 correctly refuses
to let the system be its own ground truth.

This script closes that gap. You write down what ACTUALLY happened, by hand,
from the recorded video. The script then compares the software's log against
your independent record and calculates the metrics the thesis already
defines:

    workflow accuracy        Eq. (3.3)   target > 90 %, interim > 80 %
    stage agreement          Eq. (5.7)
    transition latency       Eq. (5.8)   signed, seconds
    relative timing error    Eq. (5.9)   target < 10 %
    wrong-part / wrong-order detection, per ERROR EPISODE, not per record

The episode rule matters and the thesis already states it (Section 5.11):
several warnings during one continuous episode are repeated observations of
one condition, not several detections. This script groups them accordingly.

WHAT YOU NEED
-------------
  1. A process log from a trial:  logs\\process\\process_YYYYMMDD_HHMMSS.jsonl
  2. A ground-truth CSV that YOU fill in from the video.

     Create the template with:
         python tools\\score_trial.py --template my_trial_gt.csv

     It looks like this. Fill in the seconds from the video, counted from
     the moment the software started:

         event_type,label,true_time_s,notes
         stage,Stage_01,8.4,body_a and bushing seated
         stage,Stage_02,21.0,
         error,WRONG_PART,45.2,block_pin placed during Stage_01

     event_type is either 'stage' (a real completed assembly step) or
     'error' (a deliberately introduced mistake).

HOW TO RUN
----------
    python tools\\score_trial.py --log logs\\process\\process_...jsonl ^
                                --truth my_trial_gt.csv ^
                                --trial T-10

OUTPUT
------
  thesis_results\\trials\\<trial>\\
      trial_summary.tex      Table: the headline scored metrics
      trial_transitions.tex  Table: per-stage reference vs system vs latency
      trial_errors.tex       Table: introduced error episodes and response
      trial_timeline.png     Reference vs system stage timeline
      trial_latency.png      Per-stage transition latency
      trial_results.json     Raw numbers
      REPORT.txt             Plain-language summary

HONESTY RULES BUILT IN
----------------------
  * A metric with no reference data is printed as "unscored", never
    estimated from the system's own records.
  * A transition counts as correct only if the system reported the right
    stage within the matching tolerance (default 3.0 s, set with
    --tolerance). Late beyond tolerance counts as missed, not correct.
  * Warnings outside any declared error episode are counted as FALSE
    warnings and reported separately. They are not quietly dropped.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEMPLATE = """event_type,label,true_time_s,notes
# ---------------------------------------------------------------------
# Fill this in from the RECORDED VIDEO, not from the software output.
# true_time_s = seconds from the moment main.py started.
#
# event_type = stage  ->  a real assembly step you actually completed
# event_type = error  ->  a mistake you deliberately introduced
#                         (label it WRONG_PART, WRONG_ORDER or UNKNOWN_OBJECT)
#
# Delete the example rows below and enter your own.
# Lines starting with # are ignored.
# ---------------------------------------------------------------------
stage,Stage_01,0.0,
stage,Stage_02,0.0,
stage,Stage_03,0.0,
stage,Stage_04,0.0,
stage,Stage_05,0.0,
stage,Stage_06,0.0,
stage,Stage_07,0.0,
"""


# ══════════════════════════════════════════════════════════════════════
# Reading

def read_log(path):
    """Pull transitions, warnings and the session summary out of the JSONL."""
    transitions, warnings, summary, start = [], [], None, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get("type")
            if t == "stage_transition":
                transitions.append(rec)
            elif t == "warning":
                warnings.append(rec)
            elif t == "session_end":
                summary = rec
            elif t == "session_start":
                start = rec
    return transitions, warnings, summary, start


def read_truth(path):
    """Read the hand-written reference. Blank and # lines are skipped."""
    stages, errors = [], []
    with open(path, encoding="utf-8-sig") as f:
        rows = [r for r in f if r.strip() and not r.lstrip().startswith("#")]
    for row in csv.DictReader(rows):
        kind = (row.get("event_type") or "").strip().lower()
        label = (row.get("label") or "").strip()
        raw = (row.get("true_time_s") or "").strip()
        note = (row.get("notes") or "").strip()
        if not kind or not label or not raw:
            continue
        try:
            t = float(raw)
        except ValueError:
            continue
        entry = {"label": label, "true_time_s": t, "notes": note}
        if kind == "stage":
            stages.append(entry)
        elif kind == "error":
            errors.append(entry)
    stages.sort(key=lambda e: e["true_time_s"])
    errors.sort(key=lambda e: e["true_time_s"])
    return stages, errors


# ══════════════════════════════════════════════════════════════════════
# Scoring

def score_transitions(truth_stages, transitions, tolerance):
    """Match each reference stage to the system's report of that stage.

    A system transition counts for a reference stage only when it names the
    same stage AND lands within `tolerance` of the reference time. The
    nearest qualifying report wins, and each system record may be used once,
    so a burst of repeated confirmations cannot satisfy several references.
    """
    used = set()
    rows = []
    for ref in truth_stages:
        best, best_dt = None, None
        for i, tr in enumerate(transitions):
            if i in used:
                continue
            if (tr.get("to_stage") or "") != ref["label"]:
                continue
            sys_t = tr.get("elapsed_s")
            if sys_t is None:
                continue
            dt = float(sys_t) - ref["true_time_s"]
            if abs(dt) <= tolerance and (best_dt is None or abs(dt) < abs(best_dt)):
                best, best_dt = i, dt
        if best is not None:
            used.add(best)
            rows.append({
                "stage": ref["label"],
                "reference_s": ref["true_time_s"],
                "system_s": float(transitions[best]["elapsed_s"]),
                "latency_s": best_dt,
                "status": "correct",
            })
        else:
            # Did the system report this stage at all, just too late?
            #
            # If it did, that record is CLAIMED by this reference even
            # though it falls outside the tolerance. Without claiming it,
            # the same late report would be counted twice - once here as
            # "out of tolerance" and again below as "spurious" - which
            # would describe one late confirmation as two separate
            # faults and overstate the error rate.
            late = [(i, float(t["elapsed_s"]))
                    for i, t in enumerate(transitions)
                    if i not in used
                    and (t.get("to_stage") or "") == ref["label"]
                    and t.get("elapsed_s") is not None]
            if late:
                # nearest in time, so a repeated confirmation does not
                # arbitrarily attach to the wrong reference
                i_late, t_late = min(
                    late, key=lambda p: abs(p[1] - ref["true_time_s"]))
                used.add(i_late)
                rows.append({
                    "stage": ref["label"],
                    "reference_s": ref["true_time_s"],
                    "system_s": t_late,
                    "latency_s": t_late - ref["true_time_s"],
                    "status": "out of tolerance",
                })
            else:
                rows.append({
                    "stage": ref["label"],
                    "reference_s": ref["true_time_s"],
                    "system_s": None,
                    "latency_s": None,
                    "status": "missed",
                })

    # Genuinely spurious: a system transition that no reference stage
    # accounts for at all, on time or late.
    spurious = [transitions[i] for i in range(len(transitions)) if i not in used]
    return rows, spurious


def score_durations(truth_stages, rows):
    """Relative timing error per step, Eq. (5.9).

    A step duration is the gap between consecutive reference stages. It is
    scored only when BOTH bounding transitions were matched, because an
    unmatched boundary leaves the system duration undefined.
    """
    out = []
    for i in range(1, len(truth_stages)):
        ref_dur = truth_stages[i]["true_time_s"] - truth_stages[i - 1]["true_time_s"]
        a, b = rows[i - 1], rows[i]
        if ref_dur <= 0 or a["system_s"] is None or b["system_s"] is None:
            out.append({"step": f"{truth_stages[i-1]['label']} to "
                                f"{truth_stages[i]['label']}",
                        "reference_s": ref_dur, "system_s": None,
                        "error_pct": None})
            continue
        sys_dur = b["system_s"] - a["system_s"]
        out.append({
            "step": f"{truth_stages[i-1]['label']} to {truth_stages[i]['label']}",
            "reference_s": ref_dur,
            "system_s": sys_dur,
            "error_pct": abs(sys_dur - ref_dur) / ref_dur * 100.0,
        })
    return out


def score_errors(truth_errors, warnings, window):
    """One row per INTRODUCED ERROR EPISODE, per Section 5.11.

    Several warnings inside one episode are repeated observations of one
    condition. Counting them individually would inflate the result, which is
    exactly the mistake Section 6.2.5 warns against.
    """
    used, rows = set(), []
    for err in truth_errors:
        t0 = err["true_time_s"]
        hits = []
        for i, w in enumerate(warnings):
            wt = w.get("elapsed_s")
            if wt is None or i in used:
                continue
            if t0 <= float(wt) <= t0 + window:
                hits.append((i, float(wt)))
        if hits:
            for i, _ in hits:
                used.add(i)
            first = min(h[1] for h in hits)
            rows.append({
                "label": err["label"],
                "introduced_s": t0,
                "first_warning_s": first,
                "response_s": first - t0,
                "warning_records": len(hits),
                "warning_types": sorted({
                    warnings[i].get("warning_type", "?") for i, _ in hits}),
                "detected": True,
                "notes": err["notes"],
            })
        else:
            rows.append({
                "label": err["label"], "introduced_s": t0,
                "first_warning_s": None, "response_s": None,
                "warning_records": 0, "warning_types": [],
                "detected": False, "notes": err["notes"],
            })

    false_warnings = [w for i, w in enumerate(warnings) if i not in used]
    return rows, false_warnings


# ══════════════════════════════════════════════════════════════════════
# Output

def pctf(x, nd=2):
    return "unscored" if x is None else f"{x:.{nd}f}"


def write_outputs(out_dir, trial, data, tolerance, window):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = data["transitions"]
    durs = data["durations"]
    errs = data["errors"]
    s = data["summary"]

    (out_dir / "trial_results.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8")

    esc = lambda x: str(x).replace("_", r"\_")                  # noqa: E731

    # ---- summary table ----
    tex = [
        r"\begin{table}[htbp]", r"\centering",
        rf"\caption{{Scored results for trial {esc(trial)}, measured against "
        r"an independently annotated reference recorded from the trial video. "
        rf"A transition counts as correct only within a {tolerance:.1f}\,s "
        r"matching tolerance.}",
        rf"\label{{tab:trial-{esc(trial)}-summary}}",
        r"\begin{tabular}{|l|r|l|}", r"\hline",
        r"\textbf{Measure} & \textbf{Result} & \textbf{Target} \\", r"\hline",
    ]
    wf = s["workflow_accuracy_pct"]
    tex.append(f"Workflow accuracy (Eq.~3.3) & {pctf(wf)}\\,\\% & "
               r"$>90\,\%$ goal; $>80\,\%$ interim \\")
    tex.append(r"\hline")
    tex.append(f"Correct transitions & {s['correct']} of {s['reference_stages']} & --- \\\\")
    tex.append(r"\hline")
    tex.append(f"Missed transitions & {s['missed']} & --- \\\\")
    tex.append(r"\hline")
    tex.append(f"Transitions outside tolerance & {s['out_of_tolerance']} & --- \\\\")
    tex.append(r"\hline")
    tex.append(f"Spurious transitions & {s['spurious']} & --- \\\\")
    tex.append(r"\hline")
    tex.append(f"Mean absolute latency (Eq.~5.8) & "
               f"{pctf(s['mean_abs_latency_s'])}\\,s & --- \\\\")
    tex.append(r"\hline")
    tex.append(f"Mean relative timing error (Eq.~5.9) & "
               f"{pctf(s['mean_timing_error_pct'])}\\,\\% & $<10\\,\\%$ \\\\")
    tex.append(r"\hline")
    if s["error_episodes"]:
        tex.append(f"Error episodes detected & "
                   f"{s['errors_detected']} of {s['error_episodes']} & --- \\\\")
        tex.append(r"\hline")
        tex.append(f"Mean warning response time & "
                   f"{pctf(s['mean_response_s'])}\\,s & --- \\\\")
        tex.append(r"\hline")
    tex.append(f"False warnings (outside any episode) & "
               f"{s['false_warnings']} & --- \\\\")
    tex += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (out_dir / "trial_summary.tex").write_text("\n".join(tex), encoding="utf-8")

    # ---- transitions table ----
    tex = [
        r"\begin{table}[htbp]", r"\centering",
        rf"\caption{{Reference and system transition times for trial "
        rf"{esc(trial)}. A positive latency means the system reported the "
        r"transition after the reference time.}",
        rf"\label{{tab:trial-{esc(trial)}-transitions}}",
        r"\begin{tabular}{|l|r|r|r|l|}", r"\hline",
        r"\textbf{Stage} & \textbf{Reference (s)} & \textbf{System (s)} & "
        r"\textbf{Latency (s)} & \textbf{Status} \\", r"\hline",
    ]
    for r in rows:
        sys_s = "---" if r["system_s"] is None else f"{r['system_s']:.2f}"
        lat_s = "---" if r["latency_s"] is None else f"{r['latency_s']:+.2f}"
        tex.append(
            f"\\texttt{{{esc(r['stage'])}}} & {r['reference_s']:.2f} & "
            f"{sys_s} & {lat_s} & {r['status']} \\\\")
        tex.append(r"\hline")
    tex += [r"\end{tabular}", r"\end{table}", ""]
    (out_dir / "trial_transitions.tex").write_text("\n".join(tex),
                                                   encoding="utf-8")

    # ---- errors table ----
    if errs:
        tex = [
            r"\begin{table}[htbp]", r"\centering",
            rf"\caption{{Deliberately introduced error episodes in trial "
            rf"{esc(trial)} and the corresponding system response. Repeated "
            r"warning records inside one episode are counted as one episode, "
            r"following the scoring rule in Section~5.11.}",
            rf"\label{{tab:trial-{esc(trial)}-errors}}",
            r"\begin{tabular}{|l|r|r|r|r|l|}", r"\hline",
            r"\textbf{Introduced} & \textbf{At (s)} & "
            r"\textbf{1st warning (s)} & \textbf{Response (s)} & "
            r"\textbf{Records} & \textbf{Detected} \\", r"\hline",
        ]
        for e in errs:
            fw = "---" if e["first_warning_s"] is None else f"{e['first_warning_s']:.2f}"
            rs = "---" if e["response_s"] is None else f"{e['response_s']:.2f}"
            tex.append(
                f"\\texttt{{{esc(e['label'])}}} & {e['introduced_s']:.2f} & "
                f"{fw} & {rs} & {e['warning_records']} & "
                f"{'yes' if e['detected'] else 'NO'} \\\\")
            tex.append(r"\hline")
        tex += [r"\end{tabular}", r"\end{table}", ""]
        (out_dir / "trial_errors.tex").write_text("\n".join(tex),
                                                  encoding="utf-8")

    # ---- figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [r["stage"] for r in rows]
        idx = list(range(len(labels)))

        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot([r["reference_s"] for r in rows], idx, "o-",
                label="Reference (annotated video)", color="#2a2d6b")

        # NaN at an unreported stage so the line BREAKS there. Joining
        # across a gap would draw a system report that never happened,
        # which is exactly the kind of implied result Section 6.5 warns
        # against.
        sv = [float("nan") if r["system_s"] is None else r["system_s"]
              for r in rows]
        ax.plot(sv, idx, "s--", label="System report", color="#9c4a4a")

        missed = [(r["reference_s"], i) for i, r in zip(idx, rows)
                  if r["system_s"] is None]
        if missed:
            ax.plot([m[0] for m in missed], [m[1] for m in missed], "x",
                    markersize=12, markeredgewidth=2.5, color="#c00000",
                    label="Not reported by the system")
        ax.set_yticks(idx)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Time from session start (s)")
        ax.set_ylabel("Assembly stage")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(out_dir / "trial_timeline.png", dpi=200)
        plt.close(fig)

        lat = [(r["stage"], r["latency_s"]) for r in rows
               if r["latency_s"] is not None]
        if lat:
            fig, ax = plt.subplots(figsize=(8, 3.8))
            names = [l[0] for l in lat]
            vals = [l[1] for l in lat]
            ax.bar(names, vals,
                   color=["#8f93e0" if abs(v) <= tolerance else "#e8a0a0"
                          for v in vals],
                   edgecolor="#4a4d99")
            ax.axhline(0, color="#333", lw=1)
            ax.axhline(tolerance, ls="--", color="#9c4a4a", lw=1,
                       label=f"tolerance {tolerance:.1f} s")
            ax.axhline(-tolerance, ls="--", color="#9c4a4a", lw=1)
            ax.set_ylabel("Transition latency (s)")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
            fig.tight_layout()
            fig.savefig(out_dir / "trial_latency.png", dpi=200)
            plt.close(fig)
    except Exception as exc:                                # noqa: BLE001
        print(f"  [warn] could not draw figures: {exc}")

    # ---- plain report ----
    L = [f"SCORED TRIAL  {trial}", "=" * 70, "",
         f"  matching tolerance   {tolerance:.1f} s",
         f"  error response window {window:.1f} s",
         f"  time-base offset     {data.get('time_offset_note', 'none')}", "",
         "WORKFLOW", "-" * 70,
         f"  Reference stages       {s['reference_stages']}",
         f"  Correct transitions    {s['correct']}",
         f"  Out of tolerance       {s['out_of_tolerance']}",
         f"  Missed                 {s['missed']}",
         f"  Spurious               {s['spurious']}",
         f"  Workflow accuracy      {pctf(s['workflow_accuracy_pct'])} %   "
         f"(target >90 %, interim >80 %)",
         "", "TIMING", "-" * 70,
         f"  Mean |latency|         {pctf(s['mean_abs_latency_s'])} s",
         f"  Mean timing error      {pctf(s['mean_timing_error_pct'])} %   "
         f"(target <10 %)", ""]

    if durs:
        L += ["  per step:"]
        for d in durs:
            sys_txt = "---" if d["system_s"] is None else f"{d['system_s']:.2f}"
            L.append(f"    {d['step']:<26} ref {d['reference_s']:6.2f} s   "
                     f"sys {sys_txt:>6} s   "
                     f"err {pctf(d['error_pct'])} %")
        L.append("")

    if errs:
        L += ["ERROR EPISODES", "-" * 70]
        for e in errs:
            if e["detected"]:
                L.append(f"  {e['label']:<16} introduced {e['introduced_s']:6.2f} s "
                         f"-> warned after {e['response_s']:.2f} s "
                         f"({e['warning_records']} records: "
                         f"{', '.join(e['warning_types'])})")
            else:
                L.append(f"  {e['label']:<16} introduced {e['introduced_s']:6.2f} s "
                         f"-> NOT DETECTED")
        L += [f"  detected {s['errors_detected']} of {s['error_episodes']} episodes",
              f"  false warnings outside any episode: {s['false_warnings']}", ""]
    else:
        L += ["ERROR EPISODES", "-" * 70,
              "  none declared in the ground truth, so error detection is",
              "  unscored for this trial.", ""]

    L += ["WHERE THIS GOES IN THE THESIS", "-" * 70,
          "  New subsection 6.2.x 'Scored live trials'.",
          "  Table 6.10 rows that this trial moves off 'unscored':",
          "    - Assembly-stage recognition",
          "    - Step timing and transition latency",
          "    - Wrong-order handling      (if a WRONG_ORDER episode was run)",
          "    - Wrong-part handling       (if a WRONG_PART episode was run)",
          "", "FILES WRITTEN", "-" * 70]
    for f in sorted(out_dir.glob("*")):
        if f.is_file():
            L.append(f"  {f.name}")

    rep = "\n".join(L)
    (out_dir / "REPORT.txt").write_text(rep, encoding="utf-8")
    print(rep)


# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="process_*.jsonl from the trial")
    ap.add_argument("--truth", help="your hand-written ground-truth CSV")
    ap.add_argument("--trial", default="T-10", help="trial identifier")
    ap.add_argument("--tolerance", type=float, default=3.0,
                    help="seconds a transition may differ and still count")
    ap.add_argument("--window", type=float, default=15.0,
                    help="seconds after an introduced error in which a "
                         "warning counts as a response")
    ap.add_argument("--template", help="write a blank ground-truth CSV here "
                                       "and exit")
    ap.add_argument("--show", help="print the transitions and warnings in a "
                                   "log and exit, so you can see what the "
                                   "system reported before annotating")
    ap.add_argument("--offset", type=float, default=None,
                    help="seconds to ADD to every ground-truth time. Use "
                         "when your times are measured from the start of "
                         "the recorded video rather than from the start of "
                         "the program.")
    ap.add_argument("--auto-align", action="store_true",
                    help="derive the offset by aligning the FIRST reference "
                         "stage to the system's report of that stage. Step "
                         "durations and relative timing error are unaffected "
                         "by any constant offset; absolute latency for the "
                         "first stage becomes zero by construction and is "
                         "reported as such.")
    args = ap.parse_args()

    if args.show:
        transitions, warnings, summary, _ = read_log(args.show)
        print()
        print("WHAT THE SYSTEM REPORTED")
        print("=" * 62)
        print("  Times are seconds from when the program started,")
        print("  NOT from the start of the recorded video.")
        print()
        print("  STAGE TRANSITIONS")
        print("  " + "-" * 58)
        if transitions:
            for t in transitions:
                print(f"    {float(t.get('elapsed_s', 0)):8.2f} s   "
                      f"{str(t.get('from_stage')):>10} -> "
                      f"{t.get('to_stage')}")
        else:
            print("    none")
        print()
        print("  WARNINGS")
        print("  " + "-" * 58)
        if warnings:
            for w in warnings:
                print(f"    {float(w.get('elapsed_s', 0)):8.2f} s   "
                      f"{w.get('warning_type'):<16} "
                      f"stage {w.get('current_stage')}")
        else:
            print("    none")
        if summary:
            print()
            print(f"  session length {summary.get('total_time_s')} s")
        print()
        return

    if args.template:
        p = Path(args.template)
        p.write_text(TEMPLATE, encoding="utf-8")
        print(f"\nGround-truth template written to: {p.resolve()}\n")
        print("Open it, watch your trial video, and fill in true_time_s for")
        print("each stage you actually completed. Add 'error' rows for any")
        print("mistake you deliberately introduced. Then run:\n")
        print(f'    python tools\\score_trial.py --log <your log> '
              f'--truth "{p}" --trial T-10\n')
        return

    if not args.log or not args.truth:
        ap.error("--log and --truth are both required (or use --template)")

    transitions, warnings, summary, _ = read_log(args.log)
    truth_stages, truth_errors = read_truth(args.truth)

    if not truth_stages:
        raise SystemExit(
            "\nNo 'stage' rows with a time were found in the ground-truth "
            "file.\nFill in true_time_s for at least two stages, then run "
            "again.\n")

    # ---- time-base alignment -------------------------------------------
    # The recorded video starts when [r] was pressed, which is after the
    # program started, so hand-measured video times and the log's
    # elapsed_s do not share a zero. A single constant offset reconciles
    # them. Step DURATIONS, and therefore the relative timing error of
    # Eq. (5.9), are unaffected by any constant offset; only the absolute
    # latency of Eq. (5.8) depends on it.
    offset = 0.0
    offset_note = "none (ground-truth times already in program time)"

    if args.offset is not None:
        offset = float(args.offset)
        offset_note = f"{offset:+.2f} s, supplied with --offset"
    elif args.auto_align:
        first = truth_stages[0]
        cand = [float(t["elapsed_s"]) for t in transitions
                if (t.get("to_stage") or "") == first["label"]
                and t.get("elapsed_s") is not None]
        if cand:
            offset = min(cand) - first["true_time_s"]
            offset_note = (f"{offset:+.2f} s, derived by aligning "
                           f"{first['label']} (--auto-align). The absolute "
                           f"latency of {first['label']} is zero by "
                           f"construction and is not an independent result.")
        else:
            offset_note = ("--auto-align requested but the system never "
                           "reported the first reference stage; no offset "
                           "applied")

    if offset:
        for e in truth_stages:
            e["true_time_s"] += offset
        for e in truth_errors:
            e["true_time_s"] += offset

    print()
    print(f"  time-base offset applied: {offset_note}")
    print()

    rows, spurious = score_transitions(truth_stages, transitions,
                                       args.tolerance)
    durs = score_durations(truth_stages, rows)
    errs, false_warnings = score_errors(truth_errors, warnings, args.window)

    correct = sum(1 for r in rows if r["status"] == "correct")
    oot = sum(1 for r in rows if r["status"] == "out of tolerance")
    missed = sum(1 for r in rows if r["status"] == "missed")
    lats = [abs(r["latency_s"]) for r in rows
            if r["status"] == "correct" and r["latency_s"] is not None]
    errps = [d["error_pct"] for d in durs if d["error_pct"] is not None]
    resps = [e["response_s"] for e in errs if e["response_s"] is not None]

    s = {
        "reference_stages": len(truth_stages),
        "correct": correct,
        "out_of_tolerance": oot,
        "missed": missed,
        "spurious": len(spurious),
        "workflow_accuracy_pct": 100.0 * correct / len(truth_stages),
        "mean_abs_latency_s": (sum(lats) / len(lats)) if lats else None,
        "mean_timing_error_pct": (sum(errps) / len(errps)) if errps else None,
        "error_episodes": len(errs),
        "errors_detected": sum(1 for e in errs if e["detected"]),
        "mean_response_s": (sum(resps) / len(resps)) if resps else None,
        "false_warnings": len(false_warnings),
    }

    data = {
        "trial": args.trial,
        "log": str(args.log),
        "truth": str(args.truth),
        "tolerance_s": args.tolerance,
        "response_window_s": args.window,
        "time_offset_s": offset,
        "time_offset_note": offset_note,
        "transitions": rows,
        "durations": durs,
        "errors": errs,
        "spurious_transitions": spurious,
        "false_warnings": false_warnings,
        "summary": s,
        "session_summary": summary,
    }

    out_dir = PROJECT_ROOT / "thesis_results" / "trials" / str(args.trial)
    write_outputs(out_dir, args.trial, data, args.tolerance, args.window)


if __name__ == "__main__":
    main()
