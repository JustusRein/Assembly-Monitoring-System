"""
main.py
================================
Vision-Based Assembly Monitoring System — V2.1 + Assembly State
Main entry point.

This file wires together ALL subsystems from the architecture (File 5):

  SS2  Acquisition     → CameraFactory + CameraInterface
       Frame Validator → FrameValidator
  SS3  Calibration     → loads roi_config.json
  SS4  Processing      → MotionDetector
  SS5  Decision        → EventDetector + AssemblyStateMachine
  SS6  Analytics       → MetricsCalculator
  SS8  Storage         → SessionLogger
  SS9  Visualization   → LiveDisplay

TO RUN:
  python main.py

TO SWITCH CAMERA:
  Edit config/settings.py  → change CAMERA_TYPE = "realsense"
  Nothing else changes.

CONTROLS:
  q  = quit
  r  = start / stop recording video
  s  = save snapshot
"""

import cv2
import json
import numpy as np
import os
import time
from collections import deque

# ── Import all subsystems ─────────────────────────────────────────────────────
from config.settings         import (DISPLAY_SCALE, ROI_CONFIG_PATH, BACKGROUND_WARMUP_FRAMES,
                                     LOGS_DIR, RECORDINGS_DIR, FPS_TARGET,
                                     YOLO_MODEL_PATH, YOLO_CONFIDENCE,
                                     YOLO_ROLLING_WINDOW, YOLO_IMGSZ,
                                     YOLO_DEVICE,
                                     DETECTION_PRINT_INTERVAL_SEC,
                                     ASSEMBLY_SEQUENCE_JSON_PATH,
                                     STAGE_PREVIEW_DIR, STAGE_PREVIEW_ENABLED,
                                     STAGE_PREVIEW_SECONDS_PER_TURN,
                                     STAGE_PREVIEW_PANEL_SIZE,
                                     STAGE_PREVIEW_FALLBACK_STAGE, STAGE_PREVIEW_SHOW_NEXT,
                                     STAGE_PREVIEW_DIAGRAM_DIR, STAGE_PREVIEW_MODE,
                                     PROCESS_LOG_MEMORY_WINDOW_SEC,
                                     POSITION_VALIDATION_ENABLED,
                                     POSITION_CONFIG_PATH,
                                     POSITION_SHOW_SEAT_MARKERS)
from acquisition.camera_factory  import create_camera
from acquisition.frame_validator import FrameValidator
from processing.motion_detector import MotionDetector
from processing.object_detector import ObjectDetector
from processing.foreign_object_detector import ForeignObjectDetector
from decision.event_detector     import EventDetector
from decision.assembly_state_machine import AssemblyStateMachine, stage_change_to_event, warning_to_event
from decision.position_validator    import create_position_validator
from analytics.metrics           import MetricsCalculator
from storage.session_logger      import SessionLogger
from storage.process_log          import ProcessLogger, combination_id
from visualization.display       import LiveDisplay, SIDEBAR_WIDTH, PREVIEW_WIDTH
from config import settings as _S
from visualization.stage_preview import StagePreviewPlayer


MAIN_WINDOW = "Assembly Monitoring System"


def load_roi(path: str):
    """Load ROI points from JSON file created by roi_selection tool."""
    if not os.path.exists(path):
        print()
        print("  " + "!" * 66)
        print("  NO WORK AREA IS DEFINED. Expect false detections.")
        print("  " + "!" * 66)
        print(f"  Looked for: {path}")
        print()
        print("  Without one, YOLO is shown the WHOLE frame - the bench, the")
        print("  room behind it, your keyboard, any paperwork. It was trained")
        print("  on none of that, and it does not report 'unknown': it picks")
        print("  the closest class it has and gives it a confidence number")
        print("  that looks exactly like a real detection. Every WRONG_PART")
        print("  and WRONG_ORDER warning on an empty bench starts here.")
        print()
        print("  Fix it once, in about a minute:")
        print("      python tools\\set_work_area.py")
        print()
        return None
    with open(path) as f:
        data = json.load(f)
    points = np.array(data["points"], dtype=np.int32)
    print(f"[main] ROI loaded from: {path}")
    return points


def detection_in_roi(obj, roi_points) -> bool:
    """Is this detection inside the calibrated work area?

    Judged on the box CENTRE, not the whole box: a part at the edge of the
    fixture legitimately has a box that spills over the boundary, and
    rejecting it would be worse than the problem being solved. The centre
    is inside the work area only if the part itself is.
    """
    box = getattr(obj, "bounding_box", None) or getattr(obj, "bbox", None)
    if box is None or roi_points is None:
        return True
    try:
        x1, y1, x2, y2 = (float(v) for v in list(box)[:4])
    except (TypeError, ValueError):
        return True
    centre = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
    return cv2.pointPolygonTest(roi_points, centre, False) >= 0


def main():
    print("=" * 55)
    print("  Vision-Based Assembly Monitoring System")
    print("=" * 55)

    # ── 1. Load ROI (Calibration / SS3) ──────────────────────────────────
    roi_points = load_roi(ROI_CONFIG_PATH)

    # ── 2. Open camera (Acquisition / SS2) ───────────────────────────────
    camera = create_camera()
    if not camera.open():
        print("[main] FATAL: Could not open camera. Exiting.")
        return

    fps = camera.get_fps() or FPS_TARGET

    # ── 3. Initialise all subsystems ──────────────────────────────────────
    validator  = FrameValidator()
    motion_detector = MotionDetector(roi_points=roi_points)

    object_detector = ObjectDetector(
        model_path=YOLO_MODEL_PATH,
        confidence=YOLO_CONFIDENCE,
        rolling_window=YOLO_ROLLING_WINDOW,
        imgsz=YOLO_IMGSZ,
        device=YOLO_DEVICE,
    )
    logger     = SessionLogger(
        trial_id         = "T-01",      # change for each DOE trial
        lighting         = "Normal",    # change per DOE trial
        sensor_height_cm = "TBD",       # fill after lab measurement
        sensor_angle_deg = "TBD",       # fill after lab measurement
    )
    analytics  = MetricsCalculator()
    display    = LiveDisplay(roi_points=roi_points)

    # Stage preview column (V2.2): rotating view of the step being built.
    # Never fatal - if the frames are missing the panel is skipped.
    preview_player = None
    if STAGE_PREVIEW_ENABLED:
        try:
            preview_player = StagePreviewPlayer(
                STAGE_PREVIEW_DIR,
                seconds_per_revolution=STAGE_PREVIEW_SECONDS_PER_TURN,
                panel_size=STAGE_PREVIEW_PANEL_SIZE,
                diagram_dir=STAGE_PREVIEW_DIAGRAM_DIR,
                mode=STAGE_PREVIEW_MODE,
            )
            if not preview_player.available:
                preview_player = None
        except Exception as exc:                      # noqa: BLE001
            print(f"[main] Stage preview disabled: {exc}")
            preview_player = None

    def on_event(event):
        """Called every time Decision detects a completed event."""
        logger.log_event(event)
        analytics.process_event(event)
        print(f"[EVENT] {event.event_type:25s}  duration={event.duration_sec:.2f}s")

    events     = EventDetector(on_event_detected=on_event)

    def on_stage_change(change):
        logger.log_event(stage_change_to_event(change))
        if process_log is not None:
            process_log.record_stage_change(change, transition_type="STANDARD")
        print(f"[STAGE] {change.from_stage} -> {change.to_stage}")

    def on_assembly_warning(warning):
        logger.log_event(warning_to_event(warning))
        if process_log is not None:
            process_log.record_warning(warning)
        print(f"[WARNING] {warning.warning_type}: {warning.message}")

    stage_machine = AssemblyStateMachine(
        on_stage_change=on_stage_change,
        on_warning=on_assembly_warning,
        sequence_json_path=ASSEMBLY_SEQUENCE_JSON_PATH,
    )
    # ---- process log (V2.2): every observed part set and its duration ----
    # Written for analysis afterwards, by a person or by process_learner.py.
    # Never fatal: if it cannot start, monitoring continues without it.
    process_log = None
    process_goal_id = None
    try:
        known_stages = {
            combination_id(req): name
            for name, req in stage_machine.all_stage_requirements.items()
        }
        process_goal_id = combination_id(stage_machine.goal_requirement)
        process_log = ProcessLogger(
            log_dir=os.path.join(LOGS_DIR, "process"),
            known_stages=known_stages,
            metadata={
                "model_path": str(YOLO_MODEL_PATH),
                "camera_fps": fps,
                "camera_resolution": list(camera.get_resolution()),
                "assembly_sequence_json": str(ASSEMBLY_SEQUENCE_JSON_PATH),
            },
        )
    except Exception as exc:                               # noqa: BLE001
        print(f"[main] Process log disabled: {exc}")
        process_log = None

    # ---- position validation (V2.3) ----------------------------------
    # Answers one question YOLO cannot: did the part go into the seat this
    # stage requires. The two bushing seats are 18 mm apart and produce
    # identical class counts, so without this Stage_01 confirms with the
    # bushing in either hole.
    #
    # It is wired AFTER the state machine and the process log because its
    # warnings go to both, and it is attached as a veto rather than built
    # into the constructor so that an uncalibrated or failed position layer
    # leaves the monitor behaving exactly as it did before.
    def on_position_warning(warning):
        logger.log_event(warning_to_event(warning))
        if process_log is not None:
            process_log.record_warning(warning)
        print(f"[WARNING] {warning.warning_type}: {warning.message}")

    position_validator = create_position_validator(
        POSITION_CONFIG_PATH,
        enabled=POSITION_VALIDATION_ENABLED,
        on_warning=on_position_warning,
    )
    if position_validator.enabled:
        # The gate may only veto a confirmation the state machine has
        # already decided on. It can never confirm, advance or revoke a
        # stage - AssemblyStateMachine stays the only stage authority.
        stage_machine.set_position_gate(position_validator.gate)

    # ---- unknown object detection -------------------------------------
    # YOLO knows seven classes. A mouse, goggles, a tool or a cup on the
    # bench is none of them, so the detector reports nothing and the
    # monitor is blind to it. This works the other way round: it reports
    # work-area regions that neither the learned background nor any
    # recognised component explains.
    #
    # It cannot identify the object - only state that something is there
    # that nothing accounts for. That is the honest limit of the method,
    # and enough for the purpose: the operator can see what it is, they
    # only need to be told to look.
    def on_foreign_warning(warning):
        logger.log_event(warning_to_event(warning))
        if process_log is not None:
            process_log.record_warning(warning)
        print(f"[WARNING] {warning.warning_type}: {warning.message}")

    foreign_detector = ForeignObjectDetector(
        roi_points=roi_points,
        enabled=getattr(_S, "FOREIGN_OBJECT_ENABLED", True),
        warmup_frames=getattr(_S, "FOREIGN_OBJECT_WARMUP_FRAMES", 40),
        persist_sec=getattr(_S, "FOREIGN_OBJECT_PERSIST_SEC", 2.5),
        min_area_frac=getattr(_S, "FOREIGN_OBJECT_MIN_AREA_FRAC", 0.006),
        diff_threshold=getattr(_S, "FOREIGN_OBJECT_DIFF_THRESHOLD", 30),
        on_warning=on_foreign_warning,
    )

    logger.open()

    # ── 4. Video writer (optional recording) ─────────────────────────────
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    recording  = False
    writer     = None
    redo_mode  = False   # manual stage control - see key handling below
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    w, h       = camera.get_resolution()

    print(f"\n  Controls: [q] quit   [r] record/stop   [s] snapshot   [v] verify final product   [p] pause preview   [,] [.] turn preview")
    print(f"            [1]-[7] jump to stage   [e] redo mode   [0] reset   [b] re-learn bench")
    print(f"  Warming up background model ({BACKGROUND_WARMUP_FRAMES} frames)...\n")

    # ---- make the window fit the screen -------------------------------
    # WINDOW_NORMAL lets the window be resized and maximised, and OpenCV
    # scales the image to whatever size the window is. Without it the
    # window is locked to the canvas size, and a canvas wider than the
    # monitor simply loses its right-hand edge with no indication that
    # anything is missing.
    cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
    try:
        import tkinter as _tk
        _root = _tk.Tk()
        _root.withdraw()
        screen_w, screen_h = _root.winfo_screenwidth(), _root.winfo_screenheight()
        _root.destroy()
    except Exception:                                   # noqa: BLE001
        screen_w, screen_h = 1920, 1080                 # a safe assumption

    # The canvas is the camera frame plus the sidebar plus, when preview
    # renders exist, the stage-preview column. Reserve room for both so
    # the window is right from the first frame rather than resizing itself
    # the moment the preview appears.
    video_scale = getattr(_S, "DISPLAY_VIDEO_SCALE", 1.0)
    preview_below = getattr(_S, "DISPLAY_PREVIEW_BELOW", False)
    canvas_w = int(w * video_scale) + SIDEBAR_WIDTH + (
        0 if preview_below else (PREVIEW_WIDTH if preview_player else 0))
    canvas_h = max(int(h * video_scale) + (350 if (preview_below and preview_player) else 0),
                   902)
    fit = min((screen_w - 80) / canvas_w, (screen_h - 140) / canvas_h, 1.0)
    cv2.resizeWindow(MAIN_WINDOW,
                     max(640, int(canvas_w * fit)),
                     max(480, int(canvas_h * fit)))
    print(f"  Layout {canvas_w}x{canvas_h}px  (video at {video_scale * 100:.0f}%, "
          f"panels at full size)  on a {screen_w}x{screen_h} screen")
    if fit < 1.0:
        print(f"  Window further scaled to {fit * 100:.0f}% to fit. If the text "
              f"is hard to read, lower DISPLAY_VIDEO_SCALE in config/settings.py")
    else:
        print(f"  Fits at 100% - text will be sharp.")


    # Throttle for terminal detection summaries (avoids flooding the
    # terminal with one line per detection per frame).
    last_detection_print = 0.0

    # REAL LOOP RATE, FOR AN HONESTLY TIMED RECORDING.
    #
    # The camera reports 30 fps, but the processing loop does not run at
    # 30 fps: YOLO inference, the display composition and the window
    # update together cost far more than 33 ms. Writing frames that
    # arrive at ~9 fps into a file whose header claims 30 fps produces a
    # video that plays about three times too fast.
    #
    # That is not a cosmetic problem. The recording is the independent
    # ground truth for every scored trial, so a video with a wrong time
    # base silently corrupts every timing measurement taken from it -
    # and it does so in a way that looks like a system error rather than
    # a recording error.
    #
    # Two defences. First, the writer is created with the rate measured
    # from the loop itself rather than the camera's nominal rate. Second,
    # the elapsed session time is burned into each recorded frame, so the
    # reference times can be read directly off the picture and no
    # conversion or alignment is needed at all.
    loop_times = deque(maxlen=90)
    session_t0 = time.time()

    # Frame-validator accounting. See the validate() call in the loop.
    frames_total = 0
    frames_rejected = 0
    last_reject_print = 0.0

    def measured_fps(default=10.0):
        """Frames per second actually achieved, from recent iterations."""
        if len(loop_times) < 10:
            return default
        span = loop_times[-1] - loop_times[0]
        if span <= 0:
            return default
        return max(1.0, min(60.0, (len(loop_times) - 1) / span))

    # ── 5. Main loop ──────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                print("[main] Camera read failed. Stopping.")
                break

            # Frame Validation (between SS2 and SS4)
            #
            # A rejected frame never reaches the detector, so a badly
            # calibrated validator makes the whole system blind while
            # every component still reports itself as working. In the
            # 2026-10-12 13:01 session this discarded 83 % of the input
            # and the only visible symptom was that stages took a long
            # time to confirm. The rejection rate is therefore counted
            # and reported at the end of every session, and the
            # per-frame message is throttled so it cannot bury the log.
            frames_total += 1
            valid, reason = validator.validate(frame)
            if not valid:
                frames_rejected += 1
                now_rej = time.time()
                if now_rej - last_reject_print > 3.0:
                    rate = 100.0 * frames_rejected / max(1, frames_total)
                    print(f"[FrameValidator] Skipped frame: {reason}  "
                          f"({frames_rejected} of {frames_total}, "
                          f"{rate:.0f} % rejected so far)")
                    last_reject_print = now_rej
                continue

            # Processing (SS4)
            motion_result = motion_detector.process(frame)
            object_result = object_detector.process(frame)

            # DETECTIONS OUTSIDE THE WORK AREA ARE NOT PARTS.
            #
            # YOLO is given the whole frame, which on this bench includes a
            # laptop keyboard, the room behind, and loose paperwork. The
            # model was trained on none of that, so it does not say "I do
            # not know" - it picks the nearest class it has and reports a
            # number. That is where 'nut 54%' on a mobile phone and
            # 'knob 83%' on a blank sheet of paper came from, and every
            # WRONG_PART and WRONG_ORDER warning on an empty bench followed
            # from it.
            #
            # Nothing downstream can repair that, because by the time the
            # state machine sees it, a hallucinated nut is indistinguishable
            # from a real one. It has to be cut off here, by refusing to
            # believe a part that is not where parts can physically be.
            if roi_points is not None:
                object_result.objects = [
                    o for o in object_result.objects
                    if detection_in_roi(o, roi_points)
                ]

            # Assembly state machine (new) - one class name per detected box,
            # so counts such as two bushings are preserved.
            observed_classes = [obj.class_name for obj in object_result.objects]

            # Position check runs BEFORE the state machine, so that when the
            # machine reaches a confirmation and asks the gate, the verdict
            # it reads is this frame's, about this frame's target stage -
            # not a stale answer about the previous one.
            # pending_stage_name, not target_stage_name: the gate is asked
            # "may THIS stage be confirmed", so the validator has to have
            # been judging that same stage. target_stage_name still names
            # the already-confirmed stage until the next one passes, which
            # is right for the preview panel and wrong here.
            position_verdict = position_validator.update(
                stage_name=stage_machine.pending_stage_name,
                detections=object_result.objects,
                frame=frame,
            )

            # Anything in the work area that neither the background nor a
            # recognised component accounts for. Independent of the stage
            # logic: a stray object is wrong whatever stage is being built.
            foreign_result = foreign_detector.update(frame,
                                                     object_result.objects)

            stage_machine.update(observed_classes)

            # Process log: what was on the bench, and for how long. Written
            # whether or not the parts match a taught stage, so an operator
            # working in a different order is recorded rather than ignored.
            if process_log is not None:
                counts = {}
                for name in observed_classes:
                    counts[name] = counts.get(name, 0) + 1
                process_log.update(
                    counts,
                    target_stage=stage_machine.target_stage_name,
                    confirmed_stage=stage_machine.current_stage,
                    status=stage_machine.status,
                    status_detail=stage_machine.status_detail,
                )

            # Throttled per-class confidence summary in the terminal.
            # Example:  [YOLO] bushing 76% (avg 70%, x2) | body_a 91% (avg 88%, x1)
            now = time.time()
            if (
                object_result.class_stats
                and now - last_detection_print >= DETECTION_PRINT_INTERVAL_SEC
            ):
                summary = " | ".join(
                    f"{s.class_name} {s.best_confidence * 100:.0f}% "
                    f"(avg {s.rolling_confidence * 100:.0f}%, x{s.count})"
                    for s in object_result.class_stats.values()
                )
                print(f"[YOLO] {summary}")
                last_detection_print = now

            # Decision (SS5)
            # detector was undefined; use motion_detector's frame count (fallback to 0)
            warmup_remaining = max(
                0,
                BACKGROUND_WARMUP_FRAMES - motion_detector._frame_count
            )

            completed_event = events.update(
                motion_result=motion_result,
                object_result=object_result,
            )

            # Visualization (SS9)
            metrics_summary = analytics.get_summary()

            # ---- stage preview: the step the operator is on NOW ----
            preview_image = None
            preview_title = None
            preview_subtitle = None
            next_preview_image = None
            next_preview_title = None
            next_preview_subtitle = None
            if preview_player is not None:
                # The stage being BUILT now, not the one just confirmed,
                # so the panel advances by itself the moment a stage passes.
                shown_stage = (getattr(stage_machine, "target_stage_name", None)
                               or stage_machine.current_stage
                               or STAGE_PREVIEW_FALLBACK_STAGE)
                preview_image = preview_player.frame(shown_stage)
                if preview_image is not None:
                    done = (stage_machine.current_stage
                            and stage_machine.current_stage == shown_stage)
                    preview_title = shown_stage.replace("_", " ")
                    if done:
                        preview_title += "  (done)"
                    req = stage_machine.current_requirement or {}
                    if req:
                        preview_subtitle = "build this: " + ", ".join(
                            f"{k} x{v}" for k, v in req.items())

                # Look-ahead: show the 360° view of the NEXT stage directly
                # below the current one, so the operator can prepare parts
                # without changing the decision/state-machine logic.
                if STAGE_PREVIEW_SHOW_NEXT:
                    next_stage = getattr(stage_machine, "next_stage_name", None)
                    if next_stage:
                        next_preview_image = preview_player.frame(next_stage)
                        if next_preview_image is not None:
                            next_preview_title = next_stage.replace("_", " ")
                            next_req = stage_machine.all_stage_requirements.get(
                                next_stage, {}
                            )
                            if next_req:
                                next_preview_subtitle = "prepare: " + ", ".join(
                                    f"{k} x{v}" for k, v in next_req.items())

            display_frame = display.render(
                frame=frame,
                motion_result=motion_result,
                object_result=object_result,
                event_state=events.current_state,
                event_duration=events.current_duration,
                metrics_summary=metrics_summary,
                is_warmup=motion_result.is_warmup,
                warmup_remaining=warmup_remaining,
                current_stage=stage_machine.current_stage,
                waiting_for=stage_machine.waiting_for if not stage_machine.is_confirmed else None,
                expected_next=stage_machine.expected_next,
                next_action=stage_machine.next_action_text,
                status=stage_machine.status,
                status_detail=stage_machine.status_detail,
                # Look-ahead: lets the sidebar show progress and what to
                # prepare next, without changing any decision logic.
                current_requirement=stage_machine.current_requirement,
                stage_number=stage_machine.stage_number,
                total_stages=stage_machine.total_stages,
                next_stage_name=stage_machine.next_stage_name,
                upcoming_action=stage_machine.upcoming_action_text,
                preview_image=preview_image,
                preview_title=preview_title,
                preview_subtitle=preview_subtitle,
                next_preview_image=next_preview_image,
                next_preview_title=next_preview_title,
                next_preview_subtitle=next_preview_subtitle,
                preview_paused=(preview_player.paused
                                if preview_player is not None else False),
                # Position layer. The 360-degree previews above are
                # untouched by this - the panel keeps showing the correct
                # action for the current stage while the banner says what
                # is wrong, which is the pair the operator needs.
                position_verdict=position_verdict.verdict,
                position_message=position_verdict.message,
                position_detail=position_verdict.detail,
                position_stable=position_verdict.stable,
                seat_markers=(position_validator.seat_markers(frame.shape)
                              if POSITION_SHOW_SEAT_MARKERS else None),
                foreign_alert=foreign_result.alert,
                foreign_message=foreign_result.message,
                foreign_markers=foreign_detector.markers(),
            )

            # Shrink BEFORE showing, not after.
            #
            # These two statements used to be the other way round: the frame
            # was shown at full size and only then resized, so the resized
            # copy was thrown away on the next loop and DISPLAY_SCALE did
            # nothing at all. With the preview column the canvas is about
            # 2060 px wide, wider than a 1920 screen, so Windows clipped the
            # right-hand edge and the 360-degree panel disappeared off the
            # side of the monitor - while looking, from the operator's seat,
            # exactly as though the feature had been removed.
            shown = display_frame
            if DISPLAY_SCALE != 1.0:
                shown = cv2.resize(
                    display_frame,
                    None,
                    fx=DISPLAY_SCALE,
                    fy=DISPLAY_SCALE,
                    interpolation=cv2.INTER_AREA,
                )

            cv2.imshow(MAIN_WINDOW, shown)
            cv2.imshow("Motion Mask (debug)", motion_result.binary_mask)

            loop_times.append(time.time())

            # Automatic start, once the loop rate is known. See
            # AUTO_RECORD_TRIAL in settings.py: the ground-truth video is
            # too important to depend on remembering a keypress.
            if (getattr(_S, "AUTO_RECORD_TRIAL", False) and not recording
                    and writer is None and len(loop_times) >= 15):
                fname = os.path.join(RECORDINGS_DIR,
                                     f"recording_{int(time.time())}.mp4")
                rec_fps = measured_fps()
                writer = cv2.VideoWriter(fname, fourcc, rec_fps, (w, h))
                recording = True
                print()
                print("  " + "=" * 62)
                print(f"  RECORDING AUTOMATICALLY: {fname}")
                print(f"  measured loop rate {rec_fps:.1f} fps "
                      f"(camera reports {fps:.0f} fps)")
                print("  Every frame carries the session clock. Read your")
                print("  reference times straight off the picture.")
                print("  " + "=" * 62)
                print()

            if recording and writer:
                # Burn the session clock into the recording. This is the
                # number the trial annotator reads, and it is the same
                # clock the process log uses, so hand-measured reference
                # times and logged system times share one time base with
                # no offset and no alignment step.
                stamped = frame.copy()
                elapsed = time.time() - session_t0
                label = f"t = {elapsed:6.2f} s"
                cv2.rectangle(stamped, (8, 8), (250, 46), (0, 0, 0), -1)
                cv2.putText(stamped, label, (16, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                            (0, 255, 255), 2, cv2.LINE_AA)
                writer.write(stamped)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if not recording:
                    fname   = os.path.join(RECORDINGS_DIR, f"recording_{int(time.time())}.mp4")
                    rec_fps = measured_fps()
                    writer  = cv2.VideoWriter(fname, fourcc, rec_fps, (w, h))
                    recording = True
                    print(f"[main] Recording started: {fname}")
                    print(f"[main]   writing at the MEASURED loop rate "
                          f"{rec_fps:.1f} fps (camera reports {fps:.0f} fps).")
                    print(f"[main]   Each frame carries the session clock, so "
                          f"read reference times straight off the video.")
                else:
                    recording = False
                    if writer: writer.release(); writer = None
                    print("[main] Recording stopped.")
            elif key == ord('s'):
                fname = os.path.join(RECORDINGS_DIR, f"snapshot_{int(time.time())}.png")
                cv2.imwrite(fname, frame)
                print(f"[main] Snapshot saved: {fname}")
            elif key == ord('v'):
                if (
                    process_log is not None
                    and stage_machine.is_confirmed
                    and stage_machine.stage_number == stage_machine.total_stages
                ):
                    process_log.mark_final_product_verified(
                        True, verifier="operator-key-v", note="Operator verified final assembly."
                    )
                else:
                    print("[main] Final product cannot be verified yet - confirm the last stage first.")
            elif key == ord('p') and preview_player is not None:
                state = preview_player.toggle_pause()
                print(f"[main] Stage preview {'paused' if state else 'rotating'}.")
            elif key == ord(',') and preview_player is not None:
                preview_player.step(-1)
            elif key == ord('.') and preview_player is not None:
                preview_player.step(1)

            # ----------------------------------------------------------
            # MANUAL STAGE CONTROL
            # ----------------------------------------------------------
            # '1'-'7'  jump to that stage, treated as already completed
            #          (monitoring resumes looking for the next one)
            # SHIFT+number is awkward to detect reliably in OpenCV, so
            # 'e' toggles "redo mode": the next number pressed sets that
            # stage as the one to be re-observed rather than assumed done.
            # '0'      full reset back to the start
            #
            # Deliberately placed AFTER the existing keys so none of the
            # current controls change behaviour.
            elif key in [ord(str(n)) for n in range(1, 8)]:
                stage_no = key - ord('0')
                if stage_machine.set_stage(stage_no, confirmed=not redo_mode):
                    # Seat evidence from before the jump must not be able to
                    # satisfy a check after it, for the same reason the state
                    # machine clears its own frame window here.
                    position_validator.reset()
                    if redo_mode:
                        print(f"[main] Stage {stage_no} set to REDO - "
                              f"waiting to observe it again.")
                    else:
                        print(f"[main] Jumped to Stage {stage_no} "
                              f"(treated as completed).")
                    redo_mode = False
                else:
                    print(f"[main] Stage {stage_no} is out of range - ignored.")

            elif key == ord('e'):
                redo_mode = not redo_mode
                if redo_mode:
                    print("[main] Redo mode ON - next number key will set "
                          "that stage to be re-observed.")
                else:
                    print("[main] Redo mode OFF.")

            elif key == ord('b'):
                # Re-learn the empty bench. Needed after deliberately
                # changing the scene, otherwise that change is reported as
                # an unknown object for the rest of the session.
                foreign_detector.reset_background()

            elif key == ord('0'):
                stage_machine.reset()
                position_validator.reset()
                redo_mode = False
                print("[main] Assembly state reset to the beginning.")

    finally:
        if writer: writer.release()
        logger.close()
        if process_log is not None:
            process_log.close(goal_combination_id=process_goal_id)
        camera.release()
        cv2.destroyAllWindows()
        print("\n[main] System stopped cleanly.")
        print(f"[main] Final metrics: {analytics.get_summary()}")

        # Frame budget. A high rejection rate means the detector was
        # shown only a fraction of the session and every timing result
        # from it is suspect, so it is stated plainly rather than left
        # to be inferred from a scrolled-past log.
        if frames_total:
            rej = 100.0 * frames_rejected / frames_total
            print(f"[main] Frames: {frames_total} captured, "
                  f"{frames_total - frames_rejected} processed, "
                  f"{frames_rejected} rejected by the validator "
                  f"({rej:.1f} %)")
            if rej > 20.0:
                print()
                print("  " + "!" * 62)
                print(f"  {rej:.0f} % OF FRAMES NEVER REACHED THE DETECTOR.")
                print("  " + "!" * 62)
                print("  The system was blind for most of this session, so")
                print("  its timings describe the validator, not the")
                print("  assembly. Do not score a trial from this run.")
                print("  Lower MIN_SHARPNESS_SCORE in config/settings.py")
                print("  (0.0 disables the check) and run it again.")
                print()
        if foreign_detector.enabled:
            s = foreign_detector.stats
            print(f"[main] Work area: {s['alerts_raised']} unknown-object "
                  f"alerts | {s['alert_seconds']:.1f}s with an alert active")
        if position_validator.enabled:
            s = position_validator.stats
            print(f"[main] Position checks: {s['frames']} frames | "
                  f"{s['wrong_episodes']} wrong-position episodes | "
                  f"{s['blocked_confirmations']} confirmations blocked")


if __name__ == "__main__":
    main()
