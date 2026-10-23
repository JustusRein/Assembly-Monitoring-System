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
from decision.event_detector     import EventDetector
from decision.assembly_state_machine import AssemblyStateMachine, stage_change_to_event, warning_to_event
from decision.position_validator    import create_position_validator
from analytics.metrics           import MetricsCalculator
from storage.session_logger      import SessionLogger
from storage.process_log          import ProcessLogger, combination_id
from visualization.display       import LiveDisplay
from visualization.stage_preview import StagePreviewPlayer


def load_roi(path: str):
    """Load ROI points from JSON file created by roi_selection tool."""
    if not os.path.exists(path):
        print(f"[main] WARNING: ROI config not found at '{path}'.")
        print("  → Run roi_selection.py first to define the workspace.")
        print("  → Using full frame as ROI for now.")
        return None
    with open(path) as f:
        data = json.load(f)
    points = np.array(data["points"], dtype=np.int32)
    print(f"[main] ROI loaded from: {path}")
    return points


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
            memory_window_sec=PROCESS_LOG_MEMORY_WINDOW_SEC,
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

    logger.open()

    # ── 4. Video writer (optional recording) ─────────────────────────────
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    recording  = False
    writer     = None
    redo_mode  = False   # manual stage control - see key handling below
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    w, h       = camera.get_resolution()

    print(f"\n  Controls: [q] quit   [r] record/stop   [s] snapshot   [v] verify final product   [p] pause preview   [,] [.] turn preview")
    print(f"            [1]-[7] jump to stage   [e] redo mode   [0] reset")
    print(f"  Warming up background model ({BACKGROUND_WARMUP_FRAMES} frames)...\n")

    # Throttle for terminal detection summaries (avoids flooding the
    # terminal with one line per detection per frame).
    last_detection_print = 0.0

    # ── 5. Main loop ──────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                print("[main] Camera read failed. Stopping.")
                break

            # Frame Validation (between SS2 and SS4)
            valid, reason = validator.validate(frame)
            if not valid:
                print(f"[FrameValidator] Skipped frame: {reason}")
                continue

            # Processing (SS4)
            motion_result = motion_detector.process(frame)
            object_result = object_detector.process(frame)

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
            )

            cv2.imshow("Assembly Monitoring System", display_frame)
            cv2.imshow("Motion Mask (debug)", motion_result.binary_mask)

            if DISPLAY_SCALE != 1.0:
                display_frame = cv2.resize(
                    display_frame,
                    None,
                    fx=DISPLAY_SCALE,
                    fy=DISPLAY_SCALE,
                    interpolation=cv2.INTER_LINEAR,
                )

            if recording and writer:
                writer.write(frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if not recording:
                    fname   = os.path.join(RECORDINGS_DIR, f"recording_{int(time.time())}.mp4")
                    writer  = cv2.VideoWriter(fname, fourcc, fps, (w, h))
                    recording = True
                    print(f"[main] Recording started: {fname}")
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

            elif key == ord('0'):
                stage_machine.reset()
                position_validator.reset()
                redo_mode = False
                print("[main] Assembly state reset to the beginning.")

    finally:
        if writer: writer.release()
        logger.close()
        if process_log is not None:
            process_log.close(
                goal_combination_id=process_goal_id,
                goal_stage_name=stage_machine.goal_stage_name
                if hasattr(stage_machine, "goal_stage_name")
                else stage_machine.all_stage_requirements
                and list(stage_machine.all_stage_requirements)[-1],
            )
        camera.release()
        cv2.destroyAllWindows()
        print("\n[main] System stopped cleanly.")
        print(f"[main] Final metrics: {analytics.get_summary()}")
        if position_validator.enabled:
            s = position_validator.stats
            print(f"[main] Position checks: {s['frames']} frames | "
                  f"{s['wrong_episodes']} wrong-position episodes | "
                  f"{s['blocked_confirmations']} confirmations blocked")


if __name__ == "__main__":
    main()
