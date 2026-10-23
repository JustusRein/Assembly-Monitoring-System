#!/usr/bin/env python3
"""Generate V2.1 assembly-monitor validation summaries and figures from real logs.

Default input:  <project-root>/logs/process/*.jsonl
Optional manifest: <project-root>/validation_output/trials.csv
Optional dataset inventory: pass --dataset-root D:\\Thesis\\Blender\\synthetic_dataset

This script does not fabricate ground truth. Without a completed trials.csv it reports
observed log behaviour only and explicitly does not calculate accuracy.
"""
from __future__ import annotations
import argparse, csv, json, math, re, statistics, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

STAGE_RE = re.compile(r"Stage_0?([1-7])", re.I)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
EXPECTED_WARNING = {
    "induced_wrong_order": {"WRONG_ORDER"},
    "induced_wrong_part": {"WRONG_PART", "UNEXPECTED_PART"},
    "induced_wrong_position": {"WRONG_POSITION", "POSITION_ERROR"},
}

def stage_num(s: Any) -> int | None:
    if not s: return None
    m = STAGE_RE.search(str(s))
    return int(m.group(1)) if m else None

def csv_float(v: Any) -> float | None:
    try:
        if v is None or str(v).strip() == "": return None
        x = float(v)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError): return None

def read_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    records, issues = [], []
    with path.open("r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try:
                item = json.loads(line)
                if isinstance(item, dict): records.append(item)
                else: issues.append(f"line {lineno}: JSON value is not an object")
            except json.JSONDecodeError as e:
                issues.append(f"line {lineno}: invalid JSON ({e.msg})")
    return records, issues

def route_text(stages: list[str]) -> str:
    return ">".join(stages)

def summarize_process_log(path: Path) -> tuple[dict, list[dict], list[dict], list[str]]:
    records, issues = read_jsonl(path)
    start = next((r for r in records if r.get("type") == "session_start"), {})
    end = next((r for r in reversed(records) if r.get("type") == "session_end"), {})
    transitions = [r for r in records if r.get("type") == "stage_transition"]
    warnings = [r for r in records if r.get("type") == "warning"]
    statuses = [r for r in records if r.get("type") == "status_change"]
    states = [r for r in records if r.get("type") == "state"]
    verifications = [r for r in records if r.get("type") == "final_product_verification"]
    stages = [str(r.get("to_stage")) for r in transitions if r.get("to_stage")]
    stage_elapsed = [csv_float(r.get("elapsed_s")) for r in transitions]
    stage_elapsed = [x for x in stage_elapsed if x is not None]
    duration = csv_float(end.get("total_time_s"))
    if duration is None:
        candidates = []
        for r in records:
            for k in ("elapsed_s", "ended_s", "started_s"):
                v = csv_float(r.get(k))
                if v is not None: candidates.append(v)
        duration = max(candidates) if candidates else None
    verified = end.get("final_product_verified")
    if verified is None:
        verified = any(r.get("verified") is True for r in verifications)
    goal_observed = end.get("goal_observed")
    session_id = start.get("session_id") or end.get("session_id") or path.stem.replace("process_", "")
    meta = start.get("metadata", {}) if isinstance(start.get("metadata"), dict) else {}
    summary = {
        "log_file": path.name,
        "session_id": session_id,
        "start_time": start.get("time", ""),
        "duration_s": duration,
        "state_records": int(end.get("states_observed", len(states)) or 0),
        "known_states": end.get("known_states", ""),
        "unknown_states": end.get("unknown_states", ""),
        "invalid_states": end.get("invalid_states", ""),
        "stage_transition_records": int(end.get("stage_transition_count", len(transitions)) or 0),
        "stage_route_observed": route_text(stages),
        "highest_stage_observed": max((stage_num(s) or 0 for s in stages), default=0) or "",
        "warning_records": int(end.get("warning_count", len(warnings)) or 0),
        "warning_types": ";".join(sorted(Counter(str(w.get("warning_type", "UNKNOWN")) for w in warnings).keys())),
        "goal_observed": goal_observed if goal_observed is not None else "",
        "final_product_verified": verified if verified is not None else "",
        "eligible_for_learning": end.get("eligible_for_learning", ""),
        "model_path": meta.get("model_path", ""),
        "camera_fps": meta.get("camera_fps", ""),
        "camera_resolution": "x".join(map(str, meta.get("camera_resolution", []))) if isinstance(meta.get("camera_resolution"), list) else "",
        "json_records": len(records),
        "json_parse_issues": len(issues),
    }
    trans_rows = []
    for i, r in enumerate(transitions, 1):
        trans_rows.append({"log_file": path.name, "session_id": session_id, "transition_index": i,
            "elapsed_s": csv_float(r.get("elapsed_s")), "from_stage": r.get("from_stage", ""),
            "to_stage": r.get("to_stage", ""), "transition_type": r.get("transition_type", "")})
    warn_rows = []
    for i, r in enumerate(warnings, 1):
        obs = r.get("observed", "")
        if isinstance(obs, (dict, list)): obs = json.dumps(obs, ensure_ascii=False)
        warn_rows.append({"log_file": path.name, "session_id": session_id, "warning_index": i,
            "elapsed_s": csv_float(r.get("elapsed_s")), "warning_type": r.get("warning_type", "UNKNOWN"),
            "current_stage": r.get("current_stage", ""), "message": r.get("message", ""),
            "observed": obs, "combination_id": r.get("combination_id", "")})
    # If session_end counts exist but warning records were not retained, preserve the count in summary;
    # warning details / warning type charts remain based only on explicit warning events.
    return summary, trans_rows, warn_rows, issues

def read_manifest(path: Path) -> list[dict]:
    if not path.exists(): return []
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines: return []
    return list(csv.DictReader(lines))

def split_route(value: Any) -> list[str]:
    if value is None: return []
    return [x.strip() for x in re.split(r"[>,;|]+", str(value)) if x.strip()]

def enrich_ground_truth(summaries: list[dict], transitions: list[dict], warnings: list[dict], manifest: list[dict]) -> list[dict]:
    by_log = {s["log_file"]: s for s in summaries}
    trans_map: dict[str,list[dict]] = defaultdict(list)
    warn_map: dict[str,list[dict]] = defaultdict(list)
    for t in transitions: trans_map[t["log_file"]].append(t)
    for w in warnings: warn_map[w["log_file"]].append(w)
    rows = []
    for m in manifest:
        log = (m.get("log_file") or "").strip()
        if not log or log not in by_log: continue
        cond = (m.get("condition") or "").strip().lower()
        truth_raw = (m.get("true_stages_completed") or "").strip()
        true_n = csv_float(truth_raw)
        route_truth = split_route(m.get("true_stage_route", ""))
        actual = [str(t.get("to_stage")) for t in trans_map.get(log, []) if t.get("to_stage")]
        expected_warnings = EXPECTED_WARNING.get(cond, set())
        error_s = csv_float(m.get("error_elapsed_s"))
        matching = [w for w in warn_map.get(log, []) if str(w.get("warning_type", "")).upper() in expected_warnings]
        after = [w for w in matching if error_s is not None and csv_float(w.get("elapsed_s")) is not None and csv_float(w.get("elapsed_s")) >= error_s]
        latency = min((csv_float(w.get("elapsed_s")) - error_s for w in after), default=None) if error_s is not None else None
        # Position detection is explicitly not scored if position validation was disabled.
        note = (m.get("notes") or "").lower()
        pos_disabled = cond == "induced_wrong_position" and any(x in note for x in ("disabled", "paused", "not tracked", "not active"))
        row = {"log_file": log, "trial_id": m.get("trial_id", ""), "condition": cond,
            "true_stages_completed": int(true_n) if true_n is not None and true_n.is_integer() else (true_n if true_n is not None else ""),
            "observed_transition_records": len(actual), "observed_stage_route": route_text(actual),
            "stage_count_match": (len(actual) == int(true_n)) if true_n is not None and true_n.is_integer() else "",
            "true_route": route_text(route_truth),
            "route_exact_match": (actual == route_truth) if route_truth else "",
            "expected_warning_type": ";".join(sorted(expected_warnings)),
            "expected_warning_seen": (bool(matching) if expected_warnings else ""),
            "expected_warning_after_injection": (bool(after) if expected_warnings and error_s is not None else ""),
            "warning_latency_s": latency if latency is not None else "",
            "position_result_scored": not pos_disabled if cond == "induced_wrong_position" else "",
            "notes": m.get("notes", "")}
        rows.append(row)
    return rows

def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        if not fields: f.write("\n"); return
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows: w.writerow(row)

def tex_escape(x: Any) -> str:
    return str(x).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#").replace("$", r"\$")

def write_tex(path: Path, rows: list[dict], columns: list[tuple[str,str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated from recorded logs. Check before inclusion.\n")
        f.write("\\begin{tabular}{" + "l"*len(columns) + "}\n\\hline\n")
        f.write(" & ".join(tex_escape(label) for _,label in columns) + r" \\ \hline" + "\n")
        for row in rows:
            vals=[]
            for key,_ in columns:
                v=row.get(key, "")
                if isinstance(v,float): v=f"{v:.3f}"
                vals.append(tex_escape(v if v not in (None, "") else "--"))
            f.write(" & ".join(vals) + r" \\" + "\n")
        f.write("\\hline\n\\end{tabular}\n")

def create_manifest_template(path: Path, summaries: list[dict]) -> None:
    if path.exists(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["log_file","trial_id","date","operator","lighting","condition","true_stages_completed","true_stage_route","error_introduced_at_stage","error_elapsed_s","notes"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        f.write("# Independent ground truth. Fill from your own observation/video; never infer it from system output.\n")
        f.write("# condition: nominal | induced_wrong_part | induced_wrong_order | induced_wrong_position\n")
        f.write("# true_stages_completed: 0-7; true_stage_route: Stage_01>Stage_02>...; error_elapsed_s is seconds from session_start.\n")
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for s in summaries:
            w.writerow({"log_file": s["log_file"]})

def plot_all(out: Path, summaries: list[dict], transitions: list[dict], warnings: list[dict], gt_rows: list[dict], dataset_root: Path | None) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is required for plots. Install into the same Python environment with: python -m pip install matplotlib")
    made=[]
    def save(fig, stem):
        fig.tight_layout()
        fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
        fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig); made.extend([f"{stem}.png", f"{stem}.pdf"])
    # Stage-transition timeline plus warnings, using elapsed_s from process log.
    valid_sessions=[s for s in summaries if any(t["log_file"]==s["log_file"] and t.get("elapsed_s") is not None for t in transitions)]
    if valid_sessions:
        fig, ax=plt.subplots(figsize=(10.5,max(3.2,0.7*len(valid_sessions))))
        ymap={s["log_file"]: i for i,s in enumerate(valid_sessions)}
        for s in valid_sessions:
            fn=s["log_file"]; y=ymap[fn]
            ts=[t for t in transitions if t["log_file"]==fn and t.get("elapsed_s") is not None]
            ws=[w for w in warnings if w["log_file"]==fn and w.get("elapsed_s") is not None]
            if ts:
                xs=[t["elapsed_s"] for t in ts]
                ax.scatter(xs,[y]*len(xs),s=54,marker="o",label="Stage transition" if y==0 else None)
                for t in ts: ax.annotate(str(t.get("to_stage","")),(t["elapsed_s"],y),xytext=(0,8),textcoords="offset points",ha="center",fontsize=8,rotation=25)
            if ws:
                xs=[w["elapsed_s"] for w in ws]
                ax.scatter(xs,[y]*len(xs),s=90,marker="x",linewidths=2,label="Warning" if y==0 else None)
                for w in ws: ax.annotate(str(w.get("warning_type","WARNING")),(w["elapsed_s"],y),xytext=(0,-15),textcoords="offset points",ha="center",fontsize=8)
        ax.set_yticks(list(ymap.values()),[s["log_file"].replace("process_","").replace(".jsonl","") for s in valid_sessions])
        ax.set_xlabel("Elapsed time from session start (s)"); ax.set_title("V2.1 logged stage transitions and warnings")
        ax.grid(axis="x",alpha=.25); ax.legend(loc="best"); save(fig,"fig_v21_transition_timeline")
    # Raw counts by session (not correctness).
    if summaries:
        labels=[s["log_file"].replace("process_","").replace(".jsonl","") for s in summaries]
        x=list(range(len(labels))); width=.36
        fig,ax=plt.subplots(figsize=(max(8,1.2*len(labels)),4.8))
        ax.bar([i-width/2 for i in x],[s["stage_transition_records"] for s in summaries],width,label="Recorded stage transitions")
        ax.bar([i+width/2 for i in x],[s["warning_records"] for s in summaries],width,label="Recorded warnings")
        ax.set_xticks(x,labels,rotation=35,ha="right"); ax.set_ylabel("Record count")
        ax.set_title("Recorded process events per session (descriptive only)"); ax.legend(); ax.grid(axis="y",alpha=.25)
        save(fig,"fig_v21_session_event_counts")
    # Warning type distribution.
    counts=Counter(str(w.get("warning_type") or "UNKNOWN") for w in warnings)
    if counts:
        fig,ax=plt.subplots(figsize=(7,4.3)); names=list(counts); vals=[counts[n] for n in names]
        bars=ax.bar(names,vals); ax.bar_label(bars); ax.set_ylabel("Recorded warning events")
        ax.set_title("Warning types recorded in process logs"); ax.grid(axis="y",alpha=.25)
        save(fig,"fig_v21_warning_type_counts")
    # Duration per session.
    durations=[(s["log_file"].replace("process_","").replace(".jsonl",""),s["duration_s"]) for s in summaries if isinstance(s.get("duration_s"),(int,float))]
    if durations:
        fig,ax=plt.subplots(figsize=(max(7,1.1*len(durations)),4.3)); labels=[a for a,b in durations]; vals=[b for a,b in durations]
        bars=ax.bar(labels,vals); ax.bar_label(bars,fmt="%.1f",padding=3,fontsize=8)
        ax.set_ylabel("Session duration (s)"); ax.set_title("Recorded process-log session duration")
        ax.tick_params(axis="x",rotation=35); ax.grid(axis="y",alpha=.25); save(fig,"fig_v21_session_durations")
    # Ground-truth comparisons only when the required information is entered.
    comparable=[r for r in gt_rows if isinstance(r.get("true_stages_completed"),(int,float))]
    if comparable:
        labels=[str(r.get("trial_id") or r["log_file"].replace("process_", "")) for r in comparable]
        x=list(range(len(comparable))); width=.36
        fig,ax=plt.subplots(figsize=(max(8,1.2*len(labels)),4.8))
        ax.bar([i-width/2 for i in x],[r["true_stages_completed"] for r in comparable],width,label="Ground-truth stages completed")
        ax.bar([i+width/2 for i in x],[r["observed_transition_records"] for r in comparable],width,label="System transition records")
        ax.set_xticks(x,labels,rotation=35,ha="right"); ax.set_ylabel("Count"); ax.set_title("Ground truth compared with logged transitions (review meaning before use)")
        ax.legend(); ax.grid(axis="y",alpha=.25); save(fig,"fig_v21_ground_truth_stage_counts")
    # Dataset inventory. This is a file inventory, not a training/test split or accuracy result.
    if dataset_root and dataset_root.exists():
        inv=[]
        children=[p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
        direct_imgs=sum(1 for p in dataset_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if direct_imgs: inv.append(("_root",direct_imgs))
        for d in children:
            n=sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            inv.append((d.name,n))
        inv=[r for r in inv if r[1]>0]
        if inv:
            inv.sort(key=lambda x:x[1],reverse=True)
            write_csv(out/"dataset_asset_inventory.csv",[{"top_level_folder":a,"image_files_recursive":b} for a,b in inv])
            top=inv[:18][::-1]
            fig,ax=plt.subplots(figsize=(9.5,max(4.5,.32*len(top))))
            bars=ax.barh([a for a,b in top],[b for a,b in top]); ax.bar_label(bars,fmt="%d",padding=3,fontsize=8)
            ax.set_xlabel("Image files found recursively"); ax.set_title("Synthetic-dataset folder inventory (not the train/val/test split)")
            ax.grid(axis="x",alpha=.25); save(fig,"fig_synthetic_dataset_asset_inventory")
    return made

def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent,help="Assembly_monitor project root")
    ap.add_argument("--dataset-root",type=Path,default=None,help="Optional Blender synthetic_dataset folder")
    ap.add_argument("--process-dir",type=Path,default=None,help="Optional explicit process-log folder")
    ap.add_argument("--manifest",type=Path,default=None,help="Optional trials.csv; default <root>/validation_output/trials.csv")
    args=ap.parse_args()
    root=args.root.resolve(); process_dir=(args.process_dir or root/"logs"/"process").resolve()
    outbase=root/"validation_output"
    if not process_dir.exists():
        print(f"ERROR: process-log folder not found: {process_dir}",file=sys.stderr); return 2
    paths=sorted(process_dir.glob("*.jsonl"))
    if not paths:
        print(f"ERROR: no .jsonl process logs found directly inside {process_dir}",file=sys.stderr); return 2
    summaries=[]; trans=[]; warns=[]; issues=[]
    for p in paths:
        s,t,w,errs=summarize_process_log(p); summaries.append(s); trans.extend(t); warns.extend(w)
        issues.extend([f"{p.name}: {e}" for e in errs])
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    out=outbase/f"v21_evidence_{stamp}"; out.mkdir(parents=True,exist_ok=False)
    write_csv(out/"v21_session_summary.csv",summaries)
    write_csv(out/"v21_transition_events.csv",trans)
    write_csv(out/"v21_warning_events.csv",warns)
    (out/"v21_log_parse_issues.txt").write_text("\n".join(issues) if issues else "No malformed JSONL records found.\n",encoding="utf-8")
    manifest_path=(args.manifest or outbase/"trials.csv").resolve()
    create_manifest_template(outbase/"trials_TEMPLATE.csv",summaries)
    manifest=read_manifest(manifest_path)
    gt_rows=enrich_ground_truth(summaries,trans,warns,manifest)
    if gt_rows:
        write_csv(out/"v21_ground_truth_comparison.csv",gt_rows)
    # compact LaTeX tables
    write_tex(out/"tab_v21_session_summary.tex",summaries,[
        ("session_id","Session"),("state_records","States"),("stage_transition_records","Transitions"),("warning_records","Warnings"),("duration_s","Duration (s)"),("final_product_verified","Final verified")])
    if warns:
        write_tex(out/"tab_v21_warning_events.tex",warns,[
            ("session_id","Session"),("elapsed_s","Time (s)"),("warning_type","Warning"),("current_stage","Stage"),("message","Message")])
    figs=plot_all(out,summaries,trans,warns,gt_rows,args.dataset_root.resolve() if args.dataset_root else None)
    total_trans=sum(int(s.get("stage_transition_records") or 0) for s in summaries)
    total_warn=sum(int(s.get("warning_records") or 0) for s in summaries)
    total_states=sum(int(s.get("state_records") or 0) for s in summaries)
    report=[]
    report += ["# V2.1 validation evidence report", "", f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Project root: `{root}`", f"Process-log directory: `{process_dir}`", f"Number of JSONL logs analysed: **{len(summaries)}**", "",
        "## Descriptive log evidence", "",
        f"- State records (using session-end counters where present): **{total_states}**",
        f"- Stage-transition records: **{total_trans}**",
        f"- Warning records (using session-end counters where present): **{total_warn}**",
        f"- Explicit warning-event rows available for plotting: **{len(warns)}**", "",
        "These are counts from software logs, not accuracy percentages. A transition or warning is not by itself proof that the physical assembly stage or error was correct.", "",
        "## Ground truth", ""]
    if gt_rows:
        valid=[r for r in gt_rows if r.get("stage_count_match") in (True,False)]
        matched=sum(r["stage_count_match"] is True for r in valid)
        report.append(f"Ground-truth rows matched to logs: **{len(gt_rows)}**. Rows with a numeric true-stage count: **{len(valid)}**.")
        if valid: report.append(f"Transition-count agreements: **{matched}/{len(valid)}** (descriptive count agreement, not full workflow accuracy).")
        report.append("Inspect `v21_ground_truth_comparison.csv` before reporting any rate. Route comparison is only scored where the true route was entered.")
    else:
        report += [f"No usable ground-truth manifest was found at `{manifest_path}` or its required fields are blank.",
            "No stage accuracy, sequence accuracy, error-detection rate, or latency is claimed. Fill `validation_output/trials.csv` from an independent observation/video, then rerun the script."]
    report += ["", "## Per-session records", "", "| Log | States | Transition records | Warning records | Highest stage logged | Final verified |", "|---|---:|---:|---:|---|---|"]
    for s in summaries:
        report.append(f"| {s['log_file']} | {s['state_records']} | {s['stage_transition_records']} | {s['warning_records']} | {s['highest_stage_observed']} | {s['final_product_verified']} |")
    report += ["", "## Generated figures", ""] + [f"- `{f}`" for f in figs]
    report += ["", "## Interpretation boundaries", "",
        "1. Sessions in the selected process directory are included even if they were pilots. Move older/demo logs outside `logs/process` before a formal trial series; do not remove raw data, archive them elsewhere.",
        "2. Position validation was disabled in the reported runs. Do not interpret these logs as evidence of wrong-position detection.",
        "3. An unknown object outside the seven YOLO classes may not be detected. Report a missed unknown-object test as a negative observation.",
        "4. The optional dataset folder inventory is only a count of image files by top-level directory. It is not a dataset split, annotation-quality result, or detector-performance score.",
        "5. Keep system logs and the independent physical ground truth separate. Never use the system's own predicted stage as the ground-truth answer."]
    (out/"v21_validation_evidence_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    # JSON summary for scripts and reproducibility.
    (out/"v21_metrics.json").write_text(json.dumps({"generated_at":datetime.now().isoformat(timespec="seconds"),"root":str(root),"process_dir":str(process_dir),"logs_analysed":len(summaries),"state_records":total_states,"stage_transition_records":total_trans,"warning_records":total_warn,"explicit_warning_events":len(warns),"ground_truth_rows":len(gt_rows),"figures":figs},indent=2),encoding="utf-8")
    print("="*64); print("V2.1 EVIDENCE REPORT GENERATED"); print("="*64)
    print(f"Logs analysed: {len(summaries)} | states: {total_states} | stage-transition records: {total_trans} | warning records: {total_warn}")
    if not gt_rows: print("Ground truth: NOT COMPLETE; no accuracy percentage calculated.")
    print(f"Output folder: {out}")
    print("Main report:  " + str(out/"v21_validation_evidence_report.md"))
    print("Session CSV:  " + str(out/"v21_session_summary.csv"))
    print(f"Figures: {len(figs)} file(s), PNG + PDF variants where applicable")
    return 0

if __name__ == "__main__":
    try: raise SystemExit(main())
    except RuntimeError as e:
        print(f"ERROR: {e}",file=sys.stderr); raise SystemExit(2)
