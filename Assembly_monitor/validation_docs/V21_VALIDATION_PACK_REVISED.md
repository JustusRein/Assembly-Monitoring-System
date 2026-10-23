# V2.1 Validation Pack - Audited and Corrected

This revised pack is intended for the 09 October 2026 validation session. It keeps
Claude's useful structure and thesis-ready outputs, but fixes methodological and
workflow problems found in the original files.

## Read this first: critical audit findings

- `EXAMPLE_report.txt` is **a synthetic software-test example**, not a real-bench result. The shared Claude conversation says the script was tested against synthetic JSONL fixtures. Do not copy its 100% results or its timings into the thesis.
- The old run sheet lists 5 nominal + 2 wrong-part + 2 wrong-order + 2 wrong-position runs, which totals **11**, not 10. The example report's 10-trial count and its 2/1/2 error mix do not correspond to that run sheet.
- The original script silently pre-filled every new template row as `nominal` with `true_stages_completed=7`. That could create false accuracy results if the operator did not edit every row. The corrected script leaves ground-truth fields blank.
- The original response-time calculation inferred the error start from a prior status-change record. That is not necessarily when the operator actually introduced the error. The corrected script requires `error_elapsed_s` to score detection latency and will leave that result unscored when the timestamp is absent.
- The original script counted every warning record in nominal sessions as a “false alarm”. The corrected output reports raw warning records and trials with warnings separately; the warning types still need to be reviewed before calling them false alarms.
- The original template parser split CSV rows on commas, which breaks valid quoted notes containing commas. The corrected script uses Python's CSV parser.
- The original LaTeX skeleton points to `tables/` and `figures/`, while the script writes the generated `.tex` and figure files directly into `validation_output/`. The skeleton below uses paths consistent with the script.
- The position-tracking screenshots from the actual workstation showed `BODY NOT TRACKED – POSITION CHECK PAUSED` and misaligned hole markers. You previously disabled `POSITION_VALIDATION_ENABLED`. **Do not run or report wrong-position detection as a successful validation while the feature is disabled or tracking is paused.** Treat it as an outstanding limitation unless you later establish valid tracking and get correctly aligned markers on real frames.
- The corrected script has been compiled, tested against six analyser smoke tests, and checked against a sample JSONL from the provided `Assembly_monitor.zip`. The deployed `ProcessLogger` records stages as individual `stage_transition` rows rather than writing `stage_route` and `stages_completed` directly in `session_end`; the analyser now derives the route from those transitions. Still inspect a fresh pilot log from the exact folder you will use for validation, because the packaged archive may not be byte-for-byte identical to the running application.
- The error table and graph distinguish total error-injection trials from trials that can actually be scored. An error trial with no independently recorded `error_elapsed_s` is marked unscored rather than being counted as a detection failure. The detection-rate denominator is therefore the number of scored trials, and the generated table shows total and scored counts separately.
- The sample archived session recorded all seven stage transitions but had `goal_observed=false` and `build_time_s=null` in its `session_end` record. The analyser therefore reports the Stage_07 transition separately from whether the final component combination was actually observed. Do not treat the transition alone as proof that the final physical assembly was visually confirmed.

The analysis is designed to produce thesis material from your actual logs; it does not manufacture physical-test results. Leave missing or unmeasured results blank and describe them as limitations. The uploaded `EXAMPLE_report.txt` must remain an example only: do not use its percentages, trial count, build times, or response times in the thesis because it was generated from synthetic test fixtures.

---

## PART 1 - What the V1 chapter could not do, and what this fixes

Your V1 section ends with an honest admission:

> "No independently annotated reference sequence was available for this
> recording... the 49 detected events cannot be converted into an accuracy
> percentage."

That is the single difference between a *functional demonstration* and a
*validation*. V1 had no ground truth, so it could only report what the system
said, never whether the system was right.

For V2.1 you can close that gap cheaply, because the ground truth for an
assembly task is trivially knowable: **you are the one building it.** You know
how many stages you completed and whether you deliberately put the wrong part
down. Writing that into a small CSV before/after each run is what converts
your logs into measurable accuracy.

The ground-truth file is necessary, but not sufficient by itself. The JSONL fields
must actually represent the events the analyser expects, each ground-truth row
must match the correct log file, and the warning being scored must occur after
the deliberately introduced error. Do not skip these checks.

---

## PART 2 - The trials to record (prioritise the working system first)

### Before recording: isolate the validation logs

1. Close the monitor. **Copy** the contents of `Assembly_monitor\logs\process` to a safe backup folder.
2. Move older/demo `.jsonl` files out of `logs\process` into a subfolder such as `logs\process_archive_before_validation` (the analyser scans only the top level of the selected folder). Do not delete them.
3. Start the monitor and confirm the camera, YOLO model, stage panel and logging work. Record the software/model configuration used. Keep the camera position, resolution, assembly fixture and lighting fixed for the five nominal trials.
4. At the end of every run, press `q` to write the `session_end` record. Press `v` only if the final product was actually completed and the application's verification action is intended for that trial. Do not press the final-product verification key on an incomplete or deliberately incorrect run.

### Recommended trial plan for today's deadline

| Trial IDs | Condition | Procedure | What can be assessed |
|---|---|---|---|
| T2-01 to T2-05 | `nominal` | Build Stage 01 → Stage 07 correctly under the same camera and lighting setup | Stage-count/sequence agreement and descriptive build/stage timing |
| T2-06 to T2-07 | `induced_wrong_part` | Reach a known stage, then present a part that is not required by the next stage | Wrong-part warning, if this warning is implemented in the live logger |
| T2-08 to T2-09 | `induced_wrong_order` | Present a later-stage part before it is required, without actually completing the skipped operation | Wrong-order/skip warning, if this is the system's actual warning type |
| Not scored today | `induced_wrong_position` | Do not count as a successful position trial while the validator is disabled, body pose is not tracked, or seat markers are misaligned | Record the failure as a limitation with screenshot/log evidence; repeat only after the real-frame position check is demonstrably active |

This is **nine planned core trials**, not eleven. A further two position trials may be added only if the position feature first passes a real-camera setup check. The earlier example report (10 trials) is not a report of this plan.

Five nominal repetitions provide a small descriptive sample, not a statistically strong reliability demonstration. Keep their conditions as similar as practical so that the timings are comparable. If you change the lighting, camera angle, resolution, or operator, record it and analyse it as a separate condition rather than mixing it silently into the nominal timing average.

For each deliberate error, record the approximate time in seconds from the start of the application session at which you introduced it. A phone stopwatch started with the application is sufficient if the application's own elapsed time is not visible. Without this timestamp, the script can list warnings but cannot reliably calculate a response latency or prove that the warning occurred after the introduced error.

---

## PART 3 - Produce the results

Copy the corrected `analyze_v21_validation.py` into:

```text
D:\Thesis\SOFTWARE V2.1 AI Final\Assembly_monitor\analyze_v21_validation.py
```

Run from PowerShell:

```powershell
cd "D:\Thesis\SOFTWARE V2.1 AI Final\Assembly_monitor"
.\venv\Scripts\python.exe analyze_v21_validation.py
```

If your virtual environment is not present at that path, use the Python interpreter that successfully runs the Assembly Monitor. Required libraries include `matplotlib`; the script otherwise uses the Python standard library.

Before committing to all trials, run one short pilot, then open its JSONL file and confirm that records such as `session_start`, `state`, `stage_transition`, `warning`, `status_change`, and `session_end` are present as appropriate. The analyser derives the stage route and count from `stage_transition.to_stage`; it does not require `stage_route` or `stages_completed` to exist in `session_end`. If the console reports zero transitions while the interface visibly advanced stages, stop and inspect the live logger connection before collecting the formal trials.

The script's warning map initially expects `WRONG_PART`, `WRONG_ORDER`, and `WRONG_POSITION` in a JSONL field named `warning_type`. Inspect a real warning record. If the live code logs a different exact token (for example, a distinct skip-ahead label), update `EXPECTED_WARNING` to that exact logged token before counting the trial. Do not infer an error detection from the interface text alone if the analyser has no corresponding log record.

The first run writes `validation_output\trials_TEMPLATE.csv`. Open it in Excel and complete every row that corresponds to a validation run. Do not label a run from memory if you can check the video/log. Save a copy as `validation_output\trials.csv`, then rerun the script.

| Column | What to record |
|---|---|
| `log_file` | Keep the analyser-generated JSONL filename unchanged |
| `trial_id` | T2-01, T2-02, … |
| `date`, `operator`, `lighting` | Actual date, operator code/initials, and actual lighting conditions |
| `condition` | `nominal`, `induced_wrong_part`, or `induced_wrong_order` for the core test; use `induced_wrong_position` only for a properly functioning position validator |
| `true_stages_completed` | Count of stages actually completed correctly, 0–7; do not automatically enter 7 for an error trial |
| `true_stage_route` | Actual route, e.g. `Stage_01>Stage_02>Stage_03`; especially important when the operator deliberately deviates from the intended order |
| `error_introduced_at_stage` | Stage where the deliberate error was introduced, or blank for nominal runs |
| `error_elapsed_s` | Seconds from session start when the error was introduced; needed to score post-error warnings and latency |
| `notes` | Camera movement, hands obscuring the part, tracker paused, unexpected reset, or other deviation |

Then run the same command again. **Do not use a percentage if the relevant denominator is zero or the relevant trials are unscored.** The corrected report explicitly shows unscored error trials when `error_elapsed_s` is missing.

The script writes directly to `validation_output\`:

```text
v21_session_summary.csv
v21_stage_timing.csv
v21_error_events.csv
v21_metrics.json
v21_validation_report.txt
tab_v21_summary.tex
tab_v21_trials.tex
tab_v21_stage_timing.tex
tab_v21_error_handling.tex
fig_stage_timeline.pdf / .png
fig_stage_durations.pdf / .png
fig_build_times.pdf / .png   (when at least two nominal build times exist)
fig_error_detection.pdf / .png
```

For the LaTeX paths below, copy the complete `validation_output` folder next to your thesis main `.tex` file. Generated stage durations are **descriptive system-log intervals**: Stage_01 starts at the first stable non-empty state available in the log, and later intervals are gaps between reported stage transitions. Because the trial sheet does not contain independently measured start/end timestamps for every stage, they do not establish timing error or timing accuracy. The complete-build-time graph is generated only if the logger actually records valid `build_time_s` values; if these are blank, report that limitation rather than substituting the Stage_01-to-Stage_07 span and calling it build time.

**Interpretation guardrails:** report every rate as `x/n`; label all detector test metrics as synthetic-held-out-set metrics if that is the split used; do not call warning counts false-alarm rates without checking the warning types; and do not claim wrong-position detection from a run in which the position validator was disabled or paused.

---

## PART 4 - LaTeX chapter skeleton

Paste this into your thesis and fill the bracketed parts from
`v21_validation_report.txt`. It matches your V1 formatting.

```latex
\section{Validation Approach for the V2.1 System}
\label{sec:v21_approach}

The V1 evaluation in Section~\ref{sec:v1_validation} established that the
implemented pipeline could reduce a continuous camera stream to discrete
activity intervals, but it also showed the limit of that result: without an
independently recorded reference, the detected events could not be compared
against what had actually happened, and no accuracy figure could be derived.

The validation of the V2.1 system was therefore designed around that gap. For
every recorded trial, the true course of the assembly was documented
independently of the system: the number of stages actually completed by the
operator, and whether an assembly error was deliberately introduced. The
system's own output, written automatically to a structured process log, was
then compared against this record. This makes the results reported in this
chapter measurements of agreement between the system and the real process,
rather than descriptions of the system's own output.

Three separate questions were examined, because they are answered by
different evidence and should not be merged into a single figure:

\begin{enumerate}
  \item whether the system recognises the assembly stages correctly
        (Section~\ref{sec:v21_stage_recognition}),
  \item what process information it produces
        (Section~\ref{sec:v21_timing}),
  \item whether it detects assembly errors
        (Section~\ref{sec:v21_errors}).
\end{enumerate}

A total of [N] usable trials were recorded on [DATE] using the Logitech RGB
webcam and the V2.1 software: [N_NOM] nominal trials in which the assembly was
carried out correctly from Stage~01 to Stage~07, and [N_ERR] deliberate
error-injection trials. Logs with incomplete endings or missing ground truth
were flagged or excluded from the relevant metric as described below.

% ---------------------------------------------------------------
\section{Component Detection Performance}
\label{sec:v21_detection}

[Take mAP50, mAP50-95 and per-class precision/recall from your YOLO
training run. State clearly that this was measured on a held-out split of
the SYNTHETIC dataset, and that it characterises the detector, not the
complete system.]

% ---------------------------------------------------------------
\section{Camera Viewpoint Feasibility Study}
\label{sec:v21_viewpoint}

A separate question had to be settled before the positional check could be
implemented at all: whether a single camera can observe the difference
between a bushing seated in the correct bore and the same bushing seated in
the adjacent one. The two seats are 18.00\,mm apart in the CAD model, which
is 1.63 times the part's own largest dimension.

The study rendered the same bushing mesh in each of the two seat positions
under identical conditions and measured the proportion of changed pixels in
a crop around the seats, at four camera elevations.

\begin{table}[H]
\centering
\caption{Observability of the bushing seat position as a function of camera
elevation (synthetic CAD study POS-FEAS-01).}
\label{tab:pos_feas}
\renewcommand{\arraystretch}{1.4}
\setlength{\tabcolsep}{8pt}
\begin{tabular}{|c|c|c|}
\hline
\textbf{Camera elevation} & \textbf{Changed pixels} & \textbf{Assessment} \\
\hline
20\textdegree & 4.24\,\% & usable \\
\hline
35\textdegree & 3.38\,\% & usable \\
\hline
50\textdegree & 2.69\,\% & marginal \\
\hline
65\textdegree & 1.79\,\% & marginal \\
\hline
\end{tabular}
\end{table}

The reported changed-pixel proportion decreases as the camera elevation
increases in this synthetic setup. This supports the limited conclusion that
the visibility signal in the rendered seat crop depends on viewpoint. It does
not, by itself, prove that the deployed webcam is mounted at the best angle or
that the live workstation can locate the holes reliably. State the actual
camera elevation measured for the deployed setup separately. Do not write that
the real camera was selected at 35\textdegree unless the physical setup was
actually measured and matches that value.

It must be stated explicitly that POS-FEAS-01 is a synthetic study carried
out on CAD renders. It establishes that the two seat positions are
\emph{observable} from a single camera and from which angles; it is not a
measurement of detection accuracy and carries no claim about performance on
real camera images.

% ---------------------------------------------------------------
\section{Stage Recognition}
\label{sec:v21_stage_recognition}

\input{validation_output/tab_v21_summary}

Across the nominal trials with complete ground-truth entries, the number of
stages reported by the system agreed with the number actually completed in
[X] of [N] scored trials ([XX.X]\,\%). The reported stage sequence matched
the required sequence in [Y] of [M] scored nominal trials ([YY.Y]\,\%). Report
over-reported and under-reported stage counts separately; do not merge them
into a single error total.

\input{validation_output/tab_v21_trials}

% ---------------------------------------------------------------
\section{Process Timing}
\label{sec:v21_timing}

Unlike the V1 baseline, which produced undifferentiated ACTIVE intervals, the
V2.1 process logger associates stage transitions with the assembly sequence.
Across [N] nominal trials with valid build-time fields, the system-reported
build time was [MEAN]\,s on average (median [MEDIAN]\,s, sample standard
deviation [SD]\,s, range [MIN]--[MAX]\,s). These are descriptive times from
the system log; no timing-error value is claimed because independent stage
start/end timestamps were not recorded as ground truth.

\input{validation_output/tab_v21_stage_timing}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{validation_output/fig_stage_durations.pdf}
\caption{Mean duration of each assembly stage across the nominal trials.
Error bars show one standard deviation.}
\label{fig:v21_stage_durations}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{validation_output/fig_stage_timeline.pdf}
\caption{System-recorded stage-confirmation intervals for each nominal trial, rebased to the first stable non-empty state in its log.}
\label{fig:v21_timeline}
\end{figure}

This is a qualitative advance over V1: the V2.1 logger associates detected
stage transitions with named assembly stages rather than reporting only
undifferentiated activity. The stage labels and durations are generated by the
system, while the validation ground truth (the stages actually completed and
any deliberate error) is recorded separately by the operator. Because actual
stage boundary times were not independently annotated, these plots describe
the system's reported timing behaviour, not timing accuracy.

% ---------------------------------------------------------------
\section{Error Handling}
\label{sec:v21_errors}

\input{validation_output/tab_v21_error_handling}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{validation_output/fig_error_detection.pdf}
\caption{Deliberately introduced assembly errors against errors detected.}
\label{fig:v21_errors}
\end{figure}

Of the [N_ERR_SCORED] error-injection trials that included a recorded error-onset
time, the system raised the corresponding warning in [D] cases ([DD.D]\,\%).
The mean warning latency after the recorded error onset was [R]\,s. Trials
without an error-onset timestamp are listed but excluded from latency-based
scoring. Warning records in nominal trials are reported separately; they should
be called false alarms only after the warning types and their meaning have been
checked against the actual logger implementation.

**Position-validation status for this submission:** the current real-camera
screenshots showed body tracking uncertainty and seat markers that did not align
with the visible holes. The feature was therefore disabled for safe continuation
of the existing monitor. Consequently, no real-bench wrong-hole detection rate
can be claimed from the current configuration. The wrong-bore case remains a
system requirement and a limitation/future validation item, not a validated
result. POS-FEAS-01 is a synthetic observability study only; it does not show
that the present live tracker works.

% ---------------------------------------------------------------
\section{Limitations}
\label{sec:v21_limitations}

The results above describe this system, on this assembly, on this
workstation, under the lighting conditions recorded, operated by the people
listed. Four limitations should be stated explicitly.

First, the number of trials is small. A rate of 100\,\% over five trials means
that no failure was observed in five attempts, not that failure cannot occur.
The trial count is therefore reported alongside every rate.

Second, the measurements characterise the complete pipeline end to end --
detection, state machine and operator-facing output together. They do not
isolate the contribution of the individual subsystems, and a failure observed
at system level cannot be attributed to a specific component from these data
alone.

Third, the positional check assumes a fixed assembly orientation on the
fixture and a fixed camera pose. The seat positions are calibrated in image
coordinates and must be re-calibrated if either changes. Pose normalisation,
which would remove this constraint, was not implemented.

Fourth, the camera-viewpoint feasibility study is synthetic. In the present
real-camera screenshots, body tracking was reported as uncertain and the seat
markers were visibly misaligned. Position validation was therefore disabled,
and no live wrong-hole detection result is reported. A real-camera position
validation set must be collected after calibration and tracking are corrected.

% ---------------------------------------------------------------
\section{Comparison with the V1 Baseline}
\label{sec:v21_comparison}

\begin{table}[H]
\centering
\caption{Information produced by the V1 and V2.1 systems.}
\label{tab:v1_v21_comparison}
\renewcommand{\arraystretch}{1.4}
\setlength{\tabcolsep}{8pt}
\begin{tabular}{|p{5.0cm}|p{3.4cm}|p{3.4cm}|}
\hline
\textbf{Capability} & \textbf{V1} & \textbf{V2.1} \\
\hline
Activity detected & yes & yes \\
\hline
Component identified & no & yes \\
\hline
Assembly stage identified & no & yes \\
\hline
Per-stage timing & no & yes \\
\hline
Wrong part handling implemented & no & yes; validate in live trials \\
\hline
Wrong-order handling implemented & no & yes; validate in live trials \\
\hline
Wrong position detection validated on real bench & no & no (not yet validated) \\
\hline
Stage-count agreement measurable against ground truth & no & yes (for scored trials) \\
\hline
\end{tabular}
\end{table}

The comparison distinguishes implemented functionality from validated
functionality. V2.1 provides component detections, a stage model and structured
process logs, so scored stage-count and sequence outcomes can be compared with
independently recorded ground truth. The position-validation feature is not
listed as working because the real-camera tracker and seat markers failed the
basic visual check. Do not describe overall system accuracy as one number:
detector performance on synthetic images, stage agreement on live trials, and
wrong-position handling require separate evidence.
```

---

## PART 5 - If you are short of time

Priority order for a defensible submission today:

1. **Five nominal trials with completed `trials.csv`.** This supports a small, honest report of stage-count/sequence agreement and system-reported build/stage timing, provided the real logs contain the required records.
2. **Two wrong-part and two wrong-order trials**, if the corresponding warnings are actually implemented and logged by this build. Record the injection time for each error.
3. **Component detector metrics** from the exact model used by the live app, labelled as held-out synthetic-test metrics if that is the test split.
4. **Position validation as a limitation**, unless the body tracker is visibly valid and the seat markers align with both holes on the actual camera feed. Do not spend the remaining deadline repeatedly toggling settings or manually moving markers to manufacture a result.

The immediate measured core plan is **nine runs** (five nominal, two wrong-part, two wrong-order). It does not validate wrong-hole detection. If you only complete the five nominal runs, report them as a limited live functional evaluation, not a complete validation of every feature.

---

## PART 6 - For GPT

If you hand this to another assistant, the context it needs is:

- The system is a seven-stage assembly monitor with seven YOLO component
  classes, an `AssemblyStateMachine`, a position-validation prototype and a
  structured JSONL process logger. Position validation is currently not
  validated on the live camera: body tracking paused and seat markers were
  misaligned, so the feature was disabled.
- The validation compares system-reported stage counts/routes and warnings
  against independent trial ground truth; record error-onset elapsed times to
  score warning latency.
- The corrected `analyze_v21_validation.py` writes CSVs, JSON, four LaTeX
  tables and thesis-ready figures to `validation_output/`. Its live-logger
  schema assumptions must be checked against an actual JSONL file.
- Keep the claims separate: synthetic YOLO-test metrics describe the detector
  on its synthetic test split; POS-FEAS-01 describes synthetic observability;
  nominal live trials describe end-to-end behaviour on the bench. Current runs
  cannot validate wrong-hole detection while position validation is disabled
  or tracking is paused.
