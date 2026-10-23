r"""
test_preview_render.py  -  does the 360 preview column actually get drawn?

Run:
    & "D:\Thesis\SOFTWARE V2.1 AI Final\venv\Scripts\python.exe" test_preview_render.py

No camera needed. It builds one display frame exactly as main.py does, using
your real settings and your real preview renders, and saves it as a PNG.

If preview_check.png HAS the third column, the drawing code is fine and the
column is simply off the edge of your screen - the window is wider than your
monitor. If it has NO third column, the fault is in the code path and the
numbers printed below say where.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, cv2
from config import settings as S
from visualization.display import LiveDisplay, PREVIEW_WIDTH, SIDEBAR_WIDTH
from visualization.stage_preview import StagePreviewPlayer

print("STAGE_PREVIEW_ENABLED :", getattr(S, "STAGE_PREVIEW_ENABLED", None))
print("STAGE_PREVIEW_DIR     :", getattr(S, "STAGE_PREVIEW_DIR", None))
print("STAGE_PREVIEW_SHOW_NEXT:", getattr(S, "STAGE_PREVIEW_SHOW_NEXT", None))

player = StagePreviewPlayer(
    S.STAGE_PREVIEW_DIR,
    seconds_per_revolution=S.STAGE_PREVIEW_SECONDS_PER_TURN,
    panel_size=S.STAGE_PREVIEW_PANEL_SIZE,
    diagram_dir=S.STAGE_PREVIEW_DIAGRAM_DIR,
    mode=S.STAGE_PREVIEW_MODE,
)
print("player.available      :", player.available)
print("stage names it knows  :", sorted(player._sets.keys()))

cur = player.frame("Stage_01")
nxt = player.frame("Stage_02")
print("frame('Stage_01')     :", None if cur is None else cur.shape)
print("frame('Stage_02')     :", None if nxt is None else nxt.shape)

if cur is None:
    print("\n>>> frame('Stage_01') returned None. The folder names do not match")
    print(">>> the stage names. Compare the list above with Stage_01..Stage_07.")
    raise SystemExit(1)

class MR:
    bounding_boxes = []; motion_score = 5.0; is_warmup = False
    binary_mask = np.zeros((720, 1280), np.uint8)

frame = np.full((720, 1280, 3), 60, np.uint8)
d = LiveDisplay(roi_points=None)
out = d.render(
    frame=frame, motion_result=MR(), object_result=None,
    event_state="IDLE", event_duration=0.0,
    metrics_summary={"total_active_events":0,"avg_step_time_sec":0.0,
                     "avg_cycle_time_sec":0.0,"last_step_time_sec":0.0},
    is_warmup=False, warmup_remaining=0,
    current_stage=None, waiting_for={"body_a":1,"bushing":1},
    expected_next=None, next_action="body_a x1, bushing x1",
    status="WAITING", status_detail=None,
    current_requirement={"body_a":1,"bushing":1},
    stage_number=None, total_stages=7,
    next_stage_name="Stage_02", upcoming_action="bushing x2",
    preview_image=cur, preview_title="Stage 01",
    preview_subtitle="build this: body_a x1, bushing x1",
    next_preview_image=nxt, next_preview_title="Stage 02",
    next_preview_subtitle="prepare: bushing x2",
    preview_paused=False,
)
cv2.imwrite("preview_check.png", out)

expected = 1280 + SIDEBAR_WIDTH + PREVIEW_WIDTH
print(f"\ncanvas produced       : {out.shape[1]} x {out.shape[0]} px")
print(f"expected WITH preview : {expected} px wide")
print(f"expected WITHOUT      : {1280 + SIDEBAR_WIDTH} px wide")
print("\nsaved preview_check.png")
if out.shape[1] >= expected:
    print("\n>>> THE PREVIEW COLUMN IS BEING DRAWN.")
    print(f">>> Your window is {expected} px wide. If your screen is narrower,")
    print(">>> Windows clips the right-hand side and you never see it.")
    print(">>> Open preview_check.png to confirm the column is there.")
else:
    print("\n>>> The column was NOT drawn even though the frames loaded.")
    print(">>> Send me this whole output.")
