"""
config/settings.py
==================
Central configuration for the Vision-Based Assembly Monitoring System.

WHY THIS FILE EXISTS:
  Instead of writing numbers like "threshold = 20" scattered across different
  files, ALL configurable values live here. If you want to change something,
  you change it in ONE place and it automatically updates everywhere.

  This matches the Configuration subsystem (SS10) from the system architecture
  in File 5 — "Provides settings to all subsystems".

HOW TO USE:
  from config.settings import CAMERA_INDEX, MOTION_THRESHOLD
"""

from pathlib import Path

# Anchor to this file's own location, not the current working directory.
# WHY THIS MATTERS: YOLO_MODEL_PATH below used to be the bare relative
# string "weights/best.pt". That silently resolves against whatever
# folder main.py happens to be LAUNCHED from - if launched from one
# level up (a parent folder that also happens to contain an old
# weights/best.pt from an earlier, unrelated project), Python loads
# THAT file instead, with zero error and zero warning. Class names
# then come out wrong (e.g. an old project's classes instead of the
# real ones) because YOLO reads class names from the weights file
# itself, not from anything in this config. Anchoring to __file__
# makes the path correct regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────
# CAMERA SETTINGS  (Acquisition / SS2)
# ─────────────────────────────────────────────────

# Which camera to open. 0 = first camera (usually built-in or first USB).
# Change to 1 or 2 if Logitech is not the default device.
CAMERA_INDEX = 0

# Resolution at 640x480 — sufficient for V1 motion detection.
# Higher resolution = more detail but slower processing.
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# Target frames per second. Real FPS may vary depending on USB speed.
FPS_TARGET = 30

# Camera type — change this ONE line to switch hardware.
# "logitech"   → uses OpenCV VideoCapture (Logitech webcam)
# "realsense"  → uses pyrealsense2 (Intel RealSense D415)
CAMERA_TYPE = "logitech"


# ─────────────────────────────────────────────────
# CALIBRATION SETTINGS  (Calibration / SS3)
# ─────────────────────────────────────────────────

# Path where the ROI definition is saved by the ROI tool.
ROI_CONFIG_PATH = "calibration/roi_config.json"


# ─────────────────────────────────────────────────
# PROCESSING SETTINGS  (Processing / SS4)
# ─────────────────────────────────────────────────

# Gaussian blur kernel size — must be odd number (3, 5, 7...).
# Blur reduces sensor noise before motion detection.
# Larger = more smoothing but slower.
BLUR_KERNEL_SIZE = 5

# Threshold value — pixels with difference above this count as "motion".
# Range: 0–255. Lower = more sensitive. Higher = less sensitive.
# Start at 20 and tune during DOE testing.
MOTION_THRESHOLD = 20

# Morphological filtering kernel size (removes noise dots from motion mask).
# See File 7 DOE — this is one of the tuning parameters.
MORPH_KERNEL_SIZE = 5

# Number of frames to skip at startup while background model warms up.
# During warmup, motion events are NOT logged (avoids false starts).
BACKGROUND_WARMUP_FRAMES = 50

# Minimum contour area in pixels — contours smaller than this are ignored.
# Prevents tiny dust/reflection artifacts from being detected as motion.
MIN_CONTOUR_AREA = 500

# Minimum percentage of ROI pixels that must be moving to count as "motion".
# Example: 1.5 means 1.5% of the ROI must show movement.
MOTION_SCORE_THRESHOLD = 1.5


# ─────────────────────────────────────────────────
# DECISION SETTINGS  (Decision / SS5)
# ─────────────────────────────────────────────────

# Number of CONSECUTIVE frames that must show the same state before
# confirming a state change. Prevents flickering between Active/Idle.
# Higher = more stable but slightly slower to respond.
TEMPORAL_FILTER_FRAMES = 5

# Minimum duration (seconds) of IDLE state before it is logged as an event.
# Prevents tiny pauses between movements from being logged as idle events.
# Matches File 7 Sheet 4 success criteria: idle duration > 2 seconds.
IDLE_MIN_DURATION_SEC = 2.0

# Minimum duration (seconds) of ACTIVE state before it is logged as an event.
# Prevents brief reflections or shadows from being logged as assembly events.
ACTIVE_MIN_DURATION_SEC = 0.5


# ─────────────────────────────────────────────────
# STORAGE SETTINGS  (Storage / SS8)
# ─────────────────────────────────────────────────

# Folder where CSV event logs are saved.
LOGS_DIR = "logs"

# Folder where recorded videos are saved.
RECORDINGS_DIR = "recordings"

# START RECORDING AUTOMATICALLY, WITHOUT PRESSING [r].
#
# The recorded video is the independent ground truth for every scored
# trial, so a trial whose recording was started late is worthless no
# matter how well the system performed. Relying on the operator to
# remember a keypress at the right moment is a single point of failure
# in the experimental procedure, and it failed twice in testing: once
# started late, once not at all.
#
# With this enabled the recording begins as soon as the loop rate has
# been measured, a second or so after start-up, and the operator has
# nothing to remember. [r] still works to stop it early.
#
# Leave it False for ordinary use, so the disk does not fill with
# recordings of every development run. Set it True for a trial series.
AUTO_RECORD_TRIAL = True


# ─────────────────────────────────────────────────
# FRAME VALIDATION SETTINGS  (Frame Validator — between SS2 and SS4)
# ─────────────────────────────────────────────────

# Maximum allowed sudden brightness change between frames.
# If mean brightness jumps more than this, the frame is flagged as invalid.
# Prevents lighting flicker from generating false motion events.
MAX_BRIGHTNESS_JUMP = 40

# Minimum Laplacian variance for a frame to be considered "sharp enough".
# Frames below this are too blurry to process reliably.
#
# DISABLED ON MEASURED EVIDENCE (session 2026-10-12 13:01).
#
# That session rejected 616 frames and passed only 49 to the detector:
# roughly 83 % of the input was discarded before any processing. The
# recorded sharpness was min 11.8, median 15.5, max 18.0, so a threshold
# of 18.0 excluded almost the entire distribution.
#
# The frames that DID pass produced the highest detection confidences of
# the whole project: body_a 96 %, bushing 87 %, block_pin 87 %,
# knob 84 %, threaded_pin 86 %. An image too blurry to process does not
# yield 96 % detections. The threshold was rejecting usable frames.
#
# The reason is a property of the metric, not of the camera. Laplacian
# variance is computed over the WHOLE frame, and this workstation is
# largely blank white paper. Flat regions contain no edges, so they
# contribute almost nothing to the variance and pull the score down
# however sharp the component itself is. The number was measuring the
# amount of texture in the scene, not whether the part was in focus.
#
# A sharpness gate is still the right idea; this implementation is the
# wrong measurement for this bench. A corrected version would evaluate
# sharpness inside the detected component's bounding box rather than
# across the full frame. Until that exists, the gate is off: silently
# discarding most of the input is far more damaging than occasionally
# letting a soft frame through, and YOLO's own confidence threshold
# already rejects what it cannot read.
#
# Set to a positive value to re-enable.
MIN_SHARPNESS_SCORE = 0.0
# Lowered from 50.0. At 50 the validator was rejecting almost every frame
# from this camera - a whole session logged sharpness between 8 and 49 and
# threw nearly all of it away, which is why the system looked frozen. 18
# still catches genuine motion blur while letting normal frames through.
# The real fix is the camera: focus it and move it closer. Raise this again
# once the picture is sharp.

# Display window scale factor. 1.0 = original camera size,
# 1.5 = 50% bigger, 2.0 = double size.
DISPLAY_SCALE = 1.5

# ─────────────────────────────────────────────────
# YOLO OBJECT DETECTION SETTINGS  (Processing / SS4)
# ─────────────────────────────────────────────────

# Path to the trained YOLO weights. Anchored to PROJECT_ROOT (see top of
# this file) so it resolves correctly no matter which folder main.py is
# launched from - see the note above PROJECT_ROOT for exactly why this
# matters.
YOLO_MODEL_PATH = str(PROJECT_ROOT / "weights" / "best.pt")

# Minimum confidence for a detection to be accepted at all.
# Detections below this value are discarded by the model.
#
# RAISED FROM 0.45 ON EVIDENCE FROM THE 2026-10-11 SESSION.
#
# That run was pointed at a laptop keyboard and crumpled paper, with no
# work area defined. Measured from its log:
#
#   FALSE detections (nothing of that class present at all)
#       block_pin   45 - 55 %      nut         46 - 64 %
#       threaded_pin 33 - 39 %     knob        61 - 87 %
#
#   TRUE detections (earlier sessions, part actually present)
#       body_a      88 - 91 %      bushing     72 - 76 %
#
# A threshold of 0.45 sat underneath the entire false band, so the system
# spent the session reporting WRONG_PART about parts that were not there.
# 0.65 clears all of it except knob, which is handled per-class below.
YOLO_CONFIDENCE = 0.65

# Per-class overrides, read by processing/object_detector.py.
#
# knob false-fired as high as 87 % on a blank sheet of paper, so it needs
# a threshold above its own false band. threaded_pin was previously set to
# 0.25 to help it get detected at all; at that value it fired on nothing
# whatsoever, which is worse than missing it, so it comes up to the floor.
#
# IF A REAL PART STOPS BEING DETECTED: lower only that class, by 0.05 at a
# time, and watch the terminal to see what else starts firing. Do not drop
# the global value back - that re-opens all seven classes at once.
YOLO_CLASS_CONFIDENCE = {
    # knob false-fired as high as 87 % on blank paper. It is NOT raised to
    # 0.90, because that would risk stalling Stage_06 on a real knob and we
    # have no measurement of what a real knob scores on this camera. The
    # work-area boundary is what removes the paper; this is a second line.
    "knob": 0.75,

    # threaded_pin is the awkward one, and the reason this dict exists.
    # object_detector.py records that it scores only 25-64 % on the live
    # camera, so the old global 0.45 filtered it out and Stage_06 stalled -
    # which is why someone set it to 0.25. At 0.25 it fires on bare bench.
    # Its false band in the 2026-10-11 log was 33-39 %, its real band runs
    # to 64 %. 0.40 sits in the gap: above every false reading observed,
    # below every real one. It is the tightest honest value, and it is
    # still the class most likely to need adjusting.
    "threaded_pin": 0.40,
}

# Number of frames used for the rolling average confidence
# shown in the Detection panel (visualization).
YOLO_ROLLING_WINDOW = 30

# Terminal print throttle for detections (seconds).
# Instead of printing every detection on every frame (which floods
# the terminal), a per-class confidence summary is printed at most
# once per this interval. Set to 0 to print every frame again.
DETECTION_PRINT_INTERVAL_SEC = 1.0

# Image size used for YOLO inference. MUST match the imgsz used
# during training (scripts/train_yolo.py) for best accuracy.
YOLO_IMGSZ = 640

# Inference device: 0 = first GPU, "cpu" = CPU, None = auto.
YOLO_DEVICE = None

# Safety contract for the live assembly model.
EXPECTED_YOLO_CLASSES = (
    "body_a",
    "body_b",
    "block_pin",
    "bushing",
    "nut",
    "knob",
    "threaded_pin",
)

# ─────────────────────────────────────────────────
# ASSEMBLY STATE MACHINE SETTINGS  (Decision — new)
# ─────────────────────────────────────────────────

# Path to assembly_sequence.json written by the Blender dataset generator.
# If not found, the state machine falls back to its own hardcoded copy
# of the same sequence and prints a warning - it will not crash.
ASSEMBLY_SEQUENCE_JSON_PATH = r"D:\Thesis\Blender\synthetic_dataset\subassembly\assembly_sequence.json"


# ─────────────────────────────────────────────────
# STAGE PREVIEW PANEL  (Visualization / SS9)  - V2.2
# ─────────────────────────────────────────────────

# Folder of 360-degree turntable renders produced by the Blender generator
# (pass RUN_STAGE_PREVIEWS in generate_dataset_YOLO_v19_1). One subfolder per
# stage, each holding frame_000.png ... frame_035.png with a transparent
# background. If the folder is missing the panel is simply not drawn.
STAGE_PREVIEW_DIR = r"D:\Thesis\Blender\synthetic_dataset\stage_previews"

# Master switch for the preview column.
STAGE_PREVIEW_ENABLED = True

# Seconds for one full 360-degree turn. Slower is easier to follow.
STAGE_PREVIEW_SECONDS_PER_TURN = 8.0

# Size of the preview image inside the column, in pixels.
STAGE_PREVIEW_PANEL_SIZE = 320

# The panel shows the stage the operator is building NOW. Until the first
# stage is confirmed it shows Stage_01, so there is always an instruction
# on screen.
STAGE_PREVIEW_FALLBACK_STAGE = "Stage_01"

# Show the stage immediately after the current target stage underneath the
# current-stage preview. This is display-only and does not affect decision logic.
STAGE_PREVIEW_SHOW_NEXT = True

# Image size used when the current and next previews are stacked vertically.
# Smaller than the single-preview size so both views fit comfortably in one window.
STAGE_PREVIEW_STACK_IMAGE_SIZE = 300

# Flat 2D diagrams for the preview panel (optional).
# Put one image per stage in this folder, named exactly after the stage:
#   Stage_01.png, Stage_02.png ...
# STAGE_PREVIEW_MODE:
#   "auto"  use a diagram when one exists for that stage, else the render
#   "2d"    diagrams only
#   "3d"    rotating renders only (the original behaviour)
# A diagram is often the clearer instruction; the rotating render is better
# where the operator must see how a part seats. "auto" allows both, chosen
# per stage, instead of committing the whole product to one or the other.
STAGE_PREVIEW_DIAGRAM_DIR = r"D:\Thesis\Blender\synthetic_dataset\stage_diagrams"
STAGE_PREVIEW_MODE = "auto"

# How long a part stays "on the bench" in the process log after it was last
# seen. The operator's hands cover most of the assembly while working, so
# raw per-frame detections show one part at a time and never match a stage.
# 4 s matches the way the state machine already reasons.
PROCESS_LOG_MEMORY_WINDOW_SEC = 4.0


# ─────────────────────────────────────────────────
# POSITION VALIDATION  (Decision / SS5)  - V2.3
# ─────────────────────────────────────────────────
#
# Checks whether a part went into the seat the stage requires, which YOLO
# class counts cannot answer: body_a + bushing x1 reads identically whether
# the bushing is in the correct hole or the wrong one.
#
# Feasibility: experiment POS-FEAS-01 (synthetic CAD study) established that
# the two seats are 18.00 mm apart and distinguishable from one oblique
# camera, 3.38 % seat-region change at 35 degrees elevation. That is a
# statement about OBSERVABILITY from CAD renders. It is not a measurement of
# accuracy on real camera images, and no accuracy figure should be quoted
# for this feature until the real-camera validation set exists.

POSITION_VALIDATION_ENABLED = False

# Written by tools/calibrate_position_anchors.py - one click per seat on a
# live frame. If the file is missing the validator reports itself inactive
# and the system behaves exactly as it did before this feature existed.
POSITION_CONFIG_PATH = r"D:\Thesis\SOFTWARE V2.1 AI Final\Assembly_monitor\config\position_config.json"

# Draw the calibrated seat rings on the video. Keep this on: it is how a
# drifted camera or a stale calibration becomes visible at a glance,
# instead of showing up later as inexplicable warnings.
POSITION_SHOW_SEAT_MARKERS = True


# ─────────────────────────────────────────────────
# DISPLAY SIZING
# ─────────────────────────────────────────────────
#
# The finished window is  video + 420 px sidebar + 360 px preview column.
# With a 1280-wide camera frame that is 2060 px, wider than most screens.
#
# This shrinks ONLY the video. The sidebar and the 360-degree previews are
# composited afterwards at full resolution, so their text stays sharp.
# Scaling the whole window down instead makes every label unreadable.
#
# With DISPLAY_PREVIEW_BELOW the previews no longer take a third column,
# so the video gets that width back:
#   0.81 -> 1036 video + 480 sidebar = 1516 px canvas   (fits 1707 at 99%)
#   0.70 ->  896 video + 480 sidebar = 1376 px canvas   (smaller video)
#   0.90 -> 1152 video + 480 sidebar = 1632 px canvas   (needs a wider screen)
# Raise it until the window stops fitting at 100%; the console prints the fit.
#
# Detection accuracy is unaffected: YOLO always runs on the full-resolution
# frame. This only changes what is drawn on screen.
DISPLAY_VIDEO_SCALE = 0.81


# ─────────────────────────────────────────────────
# UNKNOWN OBJECT DETECTION  (Processing / SS4)
# ─────────────────────────────────────────────────
#
# YOLO knows seven classes. A mouse, goggles, a screwdriver or a cup on the
# bench is none of them, so the detector reports nothing and the monitor is
# blind to it. This check works the other way round: it reports work-area
# regions that neither the learned background nor any recognised component
# explains.
#
# It cannot identify the object - only state that something is present that
# nothing accounts for. The operator can see what it is; they only need to
# be told to look.
#
# HOW IT TELLS AN OBJECT FROM A HAND: hands move, a placed object does not.
# A region must hold still, unexplained, for PERSIST_SEC before it counts.

FOREIGN_OBJECT_ENABLED = True

# Frames of empty bench used to learn the background at start-up.
# KEEP THE WORK AREA CLEAR WHILE THIS RUNS. Anything lying there becomes
# part of the background and is never reported. Press [b] during a session
# to re-learn it after deliberately changing the scene.
FOREIGN_OBJECT_WARMUP_FRAMES = 40

# How long a region must sit still, unexplained, before it is reported.
# Raise it if a resting hand triggers warnings; lower it if alerts feel slow.
FOREIGN_OBJECT_PERSIST_SEC = 2.5

# Smallest region worth reporting, as a fraction of the frame.
# 0.006 is roughly 5500 px in a 1280x720 frame - about the size of a mouse.
# Raise it if shadows or cables are flagged; lower it for smaller objects.
FOREIGN_OBJECT_MIN_AREA_FRAC = 0.006

# Greyscale difference above which a pixel counts as foreground. Lower
# catches darker objects on a dark bench but admits more noise.
FOREIGN_OBJECT_DIFF_THRESHOLD = 30

# Show the unexplained-foreground mask in its own window. Useful while
# tuning the two settings above; turn off for a demonstration.
FOREIGN_OBJECT_SHOW_MASK = False


# Put the two stage previews in a band UNDER the video instead of in a
# third column beside it.
#
# The camera frame is 16:9 - wide and short - while the sidebar is narrow
# and tall. In the three-column layout the canvas had to be as tall as the
# sidebar while the video was only about half that, so a large black band
# sat under the picture doing nothing, and the video had to be shrunk to
# leave room for a column it never shared height with.
#
# Below the video, that dead space becomes the preview band and the video
# gets the width back: 1030x579 on screen instead of 768x432, which is
# 1.8x the picture area, with no loss of text sharpness.
#
# Set False to go back to the three-column layout.
DISPLAY_PREVIEW_BELOW = True
