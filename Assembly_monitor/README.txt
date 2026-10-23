NEXT-STAGE 360° PREVIEW PATCH

Replace these three files in the existing Assembly_monitor project:

1. Assembly_monitor\main.py
2. Assembly_monitor\visualization\display.py
3. Assembly_monitor\config\settings.py

No other project files need to be changed.

Behavior:
- Top preview = the stage currently being worked toward.
- Bottom preview = the immediately following stage, when it exists.
- Both previews rotate from the same StagePreviewPlayer and share [p], [,], [.]
  controls.
- At Stage_07 there is no next stage, so the existing single-preview layout remains.
- Missing preview frames remain non-fatal.

The patch does not modify YOLO, object detection, EventDetector, or assembly
state-transition logic.
