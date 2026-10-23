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

# ─────────────────────────────────────────────────
# CAMERA SETTINGS  (Acquisition / SS2)
# ─────────────────────────────────────────────────

# Which camera to open. 0 = first camera (usually built-in or first USB).
# Change to 1 or 2 if Logitech is not the default device.
CAMERA_INDEX = 0

# Resolution at 640x480 — sufficient for V1 motion detection.
# Higher resolution = more detail but slower processing.
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

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


# ─────────────────────────────────────────────────
# FRAME VALIDATION SETTINGS  (Frame Validator — between SS2 and SS4)
# ─────────────────────────────────────────────────

# Maximum allowed sudden brightness change between frames.
# If mean brightness jumps more than this, the frame is flagged as invalid.
# Prevents lighting flicker from generating false motion events.
MAX_BRIGHTNESS_JUMP = 40

# Minimum Laplacian variance for a frame to be considered "sharp enough".
# Frames below this are too blurry to process reliably.
MIN_SHARPNESS_SCORE = 50.0

# Display window scale factor. 1.0 = original camera size,
# 1.5 = 50% bigger, 2.0 = double size.
DISPLAY_SCALE = 1.5

# ─────────────────────────────────────────────────
# YOLO OBJECT DETECTION SETTINGS  (Processing / SS4)
# ─────────────────────────────────────────────────

# Path to the trained YOLO weights.
YOLO_MODEL_PATH = "weights/best.pt"

# Minimum confidence for a detection to be accepted at all.
# Detections below this value are discarded by the model.
YOLO_CONFIDENCE = 0.25

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

# ─────────────────────────────────────────────────
# ASSEMBLY STATE MACHINE SETTINGS  (Decision — new)
# ─────────────────────────────────────────────────

# Path to assembly_sequence.json written by the Blender dataset generator.
# If not found, the state machine falls back to its own hardcoded copy
# of the same sequence and prints a warning - it will not crash.
ASSEMBLY_SEQUENCE_JSON_PATH = r"D:\Thesis\Blender\synthetic_dataset\subassembly\assembly_sequence.json"
