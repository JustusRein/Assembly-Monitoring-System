BODY-RELATIVE SEAT MARKER UPGRADE
=================================

Problem fixed
-------------
The old seat markers were absolute pixel coordinates. They stayed on the
workbench whenever body_a moved. The upgraded PositionValidator transforms
Seat_1 and Seat_2 using visual features on the detected body_a in each frame.

Files changed or added
----------------------
decision/body_pose_tracker.py                 NEW
decision/position_validator.py                UPDATED
visualization/display.py                       UPDATED
tools/calibrate_position_anchors.py            UPDATED
tools/test_body_pose_tracking.py               NEW
config/position_config.json                    UPDATED
config/position_config_TEMPLATE.json           UPDATED (if present)
config/position_config_reference.png           Existing calibration reference retained

The main.py, AssemblyStateMachine, seven-class YOLO model, process logger,
EventDetector, motion detector, and 360-degree current/next stage preview are
not replaced by this patch. main.py already passes the live frame and YOLO
body_a detection into PositionValidator; the defect was in how the validator
computed and drew seat positions.

Install safely
--------------
1. Close the Assembly Monitoring System window (press q) and stop its process.
2. Back up the current project files first.
3. Extract the patch ZIP into:
     D:\Thesis\SOFTWARE V2.1 AI Final\
   Allow overwrite only for the files in this patch. The archive paths begin
   with Assembly_monitor\, so the files land in the existing project.
4. Run the offline tests.

   In Command Prompt (cmd.exe):
   cd /d "D:\Thesis\SOFTWARE V2.1 AI Final\Assembly_monitor"
   "D:\Thesis\SOFTWARE V2.1 AI Final\venv\Scripts\python.exe" tools\test_body_pose_tracking.py
   "D:\Thesis\SOFTWARE V2.1 AI Final\venv\Scripts\python.exe" decision\position_validator.py

   In PowerShell, use:
   Set-Location "D:\Thesis\SOFTWARE V2.1 AI Final\Assembly_monitor"
   & "D:\Thesis\SOFTWARE V2.1 AI Final\venv\Scripts\python.exe" tools\test_body_pose_tracking.py
   & "D:\Thesis\SOFTWARE V2.1 AI Final\venv\Scripts\python.exe" decision\position_validator.py

5. The body-pose test should end with:
     ALL BODY-RELATIVE TRACKING SMOKE TESTS PASSED
   The position-validator self-test should show CORRECT/allow=True for the
   correct seat, WRONG/allow=False for the wrong seat, UNKNOWN/allow=False
   for a loose bushing, and NOT_APPLICABLE for a stage with no seat rule.

How to run the live monitor
---------------------------
1. Keep the Logitech camera and 1280x720 resolution unchanged.
2. Start the monitor while body_a is visible in the same view/orientation as
   the saved calibration reference. Do not start with body_a already turned
   to a substantially different face/view.
3. Run:
   "D:\Thesis\SOFTWARE V2.1 AI Final\venv\Scripts\python.exe" main.py
4. Watch the seat markers. They should follow gradual movement of body_a.
   If tracking becomes uncertain, the app hides the old markers and displays
   BODY NOT TRACKED - POSITION CHECK PAUSED. That is intentional: it must not
   leave misleading markers on the bench or silently certify a stage.
5. If the markers do not start on the holes in the calibrated body view, stop
   and send the terminal message beginning [BodyPoseTracker]. Do not proceed
   with position tests until alignment is correct.

Known limitation (do not overclaim)
-----------------------------------
This is body-relative 2-D feature tracking using a homography, not full 6-DoF
CAD pose estimation. It follows image translation, scale, in-plane rotation,
and moderate perspective changes when sufficient texture remains visible.
Large arbitrary 3-D rotations, severe occlusion, blur, or loss of body_a YOLO
recognition may make the track invalid. In those cases it hides markers and
blocks Stage_01/02 position confirmation until tracking is re-established.
The included offline tests use synthetic image transforms; they do not prove
real-camera accuracy. That must be assessed with the physical camera and
labelled real images.

Preserved behaviour
-------------------
- Seven YOLO classes are unchanged.
- The state machine remains the only authority on stage progression.
- PositionValidator can veto a Stage_01/02 confirmation, but does not invent
  or advance a stage.
- Existing process logging and 360-degree current/next preview are retained.


Important calibration-gate correction
-------------------------------------
The seat acceptance radius is now 40% of the measured seat-centre spacing.
The previous fixed 18 px minimum could make the two seat acceptance regions
overlap when the seat centres were only about 30 px apart. The supplied
calibration's gate is therefore 12.0 px, not 18.0 px. A part that cannot be
located confidently within a seat remains UNKNOWN; do not increase this gate
just to force a stage to pass.

Resolution note
---------------
The supplied calibration was made from a 1280x720 reference image. Start the
live monitor with the same camera framing and resolution as the calibration
view. If the live camera uses a different resolution or aspect ratio, re-run
the calibration tool after setting the camera to its final operating mode.
Do not interpret passing synthetic tests as proof of real-camera accuracy.
