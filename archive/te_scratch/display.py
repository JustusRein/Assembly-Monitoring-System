"""
visualization/display.py
================================
SUBSYSTEM: Visualization (SS9)

LAYOUT CHANGE (V2.1):
  Previously every panel was drawn ON TOP of the 640x480 camera frame:
  status top-left, YOLO confidence top-right, metrics bottom-left, and
  the assembly stage panel top-centre. With four panels competing for
  the same pixels, the stage panel overlapped the other two and long
  warning text ran underneath the confidence panel.

  The frame is now composited into a larger canvas:

      +---------------------------+------------------+-------------+
      |                           |                  |   STAGE     |
      |     CAMERA FRAME          |    INFO PANEL    |   PREVIEW   |
      |     640 x 480             |    420 px wide   |   360 px    |
      |     boxes + ROI only      |    all text      |   rotating  |
      |                           |                  |             |
      +---------------------------+------------------+-------------+

  The preview column (V2.2) shows the stage the operator is building now,
  rendered from the same CAD as the training data and rotating through
  360 degrees so the fit can be seen from every side. It is omitted
  automatically when no preview frames are available.

  Only bounding boxes, the ROI outline and per-detection labels remain
  on the frame itself, because those must sit on the pixels they refer
  to. Every other piece of text lives in the sidebar, where it cannot
  collide with anything and has room to wrap.

  The sidebar also shows look-ahead information (next stage + what to
  prepare) so the operator can get the following part ready while
  still completing the current step.
"""

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════
# Colours  (BGR)
# ══════════════════════════════════════════════════════════

COLOR_ACTIVE = (0, 165, 255)
COLOR_IDLE = (180, 180, 180)
COLOR_WARMUP = (50, 200, 255)

COLOR_ROI = (0, 200, 80)
COLOR_MOTION = (255, 100, 0)

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

COLOR_CONF_LOW = (0, 0, 255)       # red
COLOR_CONF_MEDIUM = (0, 215, 255)  # yellow
COLOR_CONF_HIGH = (0, 220, 0)      # green

# Sidebar theme
SIDEBAR_BG = (38, 38, 38)
SIDEBAR_DIVIDER = (70, 70, 70)
SIDEBAR_LABEL = (150, 150, 150)     # section headings
SIDEBAR_VALUE = (245, 245, 245)     # main values

SIDEBAR_WIDTH = 420
PAD = 16                            # left/right padding inside sidebar

# Stage preview column (V2.2)
PREVIEW_WIDTH = 360
PREVIEW_BG = (28, 28, 28)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def confidence_color(confidence):
    if confidence >= 0.70:
        return COLOR_CONF_HIGH
    if confidence >= 0.45:
        return COLOR_CONF_MEDIUM
    return COLOR_CONF_LOW


def status_color(status):
    """Colour for the assembly status value."""
    return {
        "OK": COLOR_CONF_HIGH,
        "WAITING": COLOR_CONF_MEDIUM,
        "WRONG_PART": COLOR_CONF_LOW,
        "WRONG_ORDER": COLOR_CONF_LOW,
    }.get(status, COLOR_WHITE)


def status_label(status):
    """Operator-readable wording for the status value."""
    return {
        "OK": "CORRECT",
        "WAITING": "WAITING",
        "WRONG_PART": "WRONG PART",
        "WRONG_ORDER": "WRONG ORDER",
    }.get(status, str(status))


class LiveDisplay:

    def __init__(self, roi_points=None):
        self._roi_points = roi_points

    # ==========================================================
    # Main entry point
    # ==========================================================

    def render(
        self,
        frame,
        motion_result,
        object_result,
        event_state,
        event_duration,
        metrics_summary,
        is_warmup,
        warmup_remaining,
        current_stage=None,
        waiting_for=None,
        expected_next=None,
        next_action=None,
        status="WAITING",
        status_detail=None,
        # ---- look-ahead, all optional so older callers still work ----
        current_requirement=None,
        stage_number=None,
        total_stages=None,
        next_stage_name=None,
        upcoming_action=None,
        # ---- stage preview column (V2.2), optional ----
        preview_image=None,
        preview_title=None,
        preview_subtitle=None,
        preview_paused=False,
    ):

        if next_action is None and expected_next:
            next_action = ", ".join(
                f"{name} x{count}" for name, count in expected_next.items()
            )

        annotated = frame.copy()

        # ---- things that MUST stay on the frame (they mark pixels) ----

        if self._roi_points is not None:
            cv2.polylines(annotated, [self._roi_points], True, COLOR_ROI, 2)

        for (x, y, w, h) in motion_result.bounding_boxes:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), COLOR_MOTION, 2)

        if object_result is not None:
            for obj in object_result.objects:
                self._draw_detection(annotated, obj)

        # ---- composite frame + sidebar ----

        return self._compose(
            annotated,
            motion_result=motion_result,
            object_result=object_result,
            event_state=event_state,
            event_duration=event_duration,
            metrics_summary=metrics_summary,
            is_warmup=is_warmup,
            warmup_remaining=warmup_remaining,
            current_stage=current_stage,
            waiting_for=waiting_for,
            current_requirement=current_requirement,
            next_action=next_action,
            status=status,
            status_detail=status_detail,
            stage_number=stage_number,
            total_stages=total_stages,
            next_stage_name=next_stage_name,
            upcoming_action=upcoming_action,
            preview_image=preview_image,
            preview_title=preview_title,
            preview_subtitle=preview_subtitle,
            preview_paused=preview_paused,
        )

    # ==========================================================
    # Detection box on the frame (stays on the frame by design)
    # ==========================================================

    def _draw_detection(self, frame, obj):

        x1, y1, x2, y2 = obj.bounding_box
        color = confidence_color(obj.confidence)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{obj.class_name} {obj.confidence * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)

        # Keep the label inside the frame even for boxes near the top edge.
        ly = y1 - 6 if y1 - 6 - th > 0 else y2 + th + 8

        cv2.rectangle(
            frame,
            (x1, ly - th - 5),
            (x1 + tw + 8, ly + 4),
            COLOR_BLACK,
            -1,
        )
        cv2.putText(frame, label, (x1 + 4, ly), FONT, 0.5, color, 1)

    # ==========================================================
    # Text helpers
    # ==========================================================

    def _wrap(self, text, max_width_px, scale, thickness):
        """Split text into lines that each fit within max_width_px."""
        words = str(text).split(" ")
        lines, current = [], ""
        for word in words:
            trial = (current + " " + word).strip()
            (w, _), _ = cv2.getTextSize(trial, FONT, scale, thickness)
            if w <= max_width_px or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def _section(self, canvas, x, y, heading):
        """Draw a small grey section heading, return the next y."""
        cv2.putText(canvas, heading.upper(), (x, y), FONT, 0.42,
                    SIDEBAR_LABEL, 1, cv2.LINE_AA)
        return y + 22

    def _value(self, canvas, x, y, text, colour=SIDEBAR_VALUE,
               scale=0.62, thickness=2, max_w=None):
        """Draw a value, wrapping if needed. Returns the next y."""
        if max_w is None:
            max_w = SIDEBAR_WIDTH - 2 * PAD
        for line in self._wrap(text, max_w, scale, thickness):
            cv2.putText(canvas, line, (x, y), FONT, scale, colour,
                        thickness, cv2.LINE_AA)
            y += int(26 * (scale / 0.62))
        return y

    def _divider(self, canvas, x, y, width):
        cv2.line(canvas, (x, y), (x + width, y), SIDEBAR_DIVIDER, 1)
        return y + 20

    # ==========================================================
    # Sidebar composition
    # ==========================================================

    def _compose(self, annotated, **d):

        fh, fw = annotated.shape[:2]
        preview_image = d.get("preview_image")
        preview_w = PREVIEW_WIDTH if preview_image is not None else 0
        canvas_w = fw + SIDEBAR_WIDTH + preview_w

        # Draw onto a deliberately over-tall scratch canvas, then crop to
        # the height actually used. The sidebar's height is not fixed: a
        # wrapped two-line warning, or several detected classes, pushes
        # the lower sections down. A fixed height silently clipped the
        # last metric row whenever that happened, so the canvas is sized
        # from the real content instead of a guess.
        SCRATCH_H = 1600
        canvas = np.zeros((SCRATCH_H, canvas_w, 3), dtype=np.uint8)
        canvas[:, :] = SIDEBAR_BG
        canvas[0:fh, 0:fw] = annotated

        x = fw + PAD
        inner_w = SIDEBAR_WIDTH - 2 * PAD
        y = 34

        # ---------- header ----------
        cv2.putText(canvas, "ASSEMBLY MONITORING", (x, y), FONT, 0.58,
                    COLOR_WHITE, 2, cv2.LINE_AA)
        y += 20
        cv2.putText(canvas, "Vision-based process verification", (x, y),
                    FONT, 0.4, SIDEBAR_LABEL, 1, cv2.LINE_AA)
        y += 16
        y = self._divider(canvas, x, y, inner_w)

        # ---------- current stage + progress ----------
        y = self._section(canvas, x, y, "Current stage")

        if d["current_stage"] is not None:
            head = d["current_stage"]
            if d["stage_number"] and d["total_stages"]:
                head += f"   ({d['stage_number']} of {d['total_stages']})"
            y = self._value(canvas, x, y, head, COLOR_CONF_HIGH, 0.72, 2)
        else:
            y = self._value(canvas, x, y, "Not yet confirmed",
                            COLOR_CONF_MEDIUM, 0.62, 2)
            if d["waiting_for"]:
                need = ", ".join(f"{k} x{v}" for k, v in d["waiting_for"].items())
                y = self._value(canvas, x, y, f"Waiting for: {need}",
                                SIDEBAR_LABEL, 0.45, 1)

        # The live requirement, printed straight from the state machine
        # so the panel can never claim something different from what the
        # logic is actually testing.
        if d.get("current_requirement"):
            req = ", ".join(f"{k} x{v}" for k, v in d["current_requirement"].items())
            y = self._value(canvas, x, y, f"needs: {req}", SIDEBAR_LABEL, 0.44, 1)

        # progress bar - only meaningful once a stage is confirmed
        if d["stage_number"] and d["total_stages"] and d["current_stage"]:
            y += 6
            bar_h = 8
            filled = int(inner_w * d["stage_number"] / d["total_stages"])
            cv2.rectangle(canvas, (x, y), (x + inner_w, y + bar_h),
                          (60, 60, 60), -1)
            cv2.rectangle(canvas, (x, y), (x + filled, y + bar_h),
                          COLOR_CONF_HIGH, -1)
            y += bar_h + 14
        else:
            y += 8

        y = self._divider(canvas, x, y, inner_w)

        # ---------- what to do now ----------
        y = self._section(canvas, x, y, "Do this now")
        if d["next_action"]:
            y = self._value(canvas, x, y, d["next_action"], COLOR_WHITE, 0.58, 2)
        else:
            y = self._value(canvas, x, y,
                            "Assembly complete" if d["current_stage"]
                            else "Place the first parts",
                            SIDEBAR_LABEL, 0.52, 1)
        y = self._divider(canvas, x, y + 4, inner_w)

        # ---------- status ----------
        y = self._section(canvas, x, y, "Status")
        y = self._value(canvas, x, y, status_label(d["status"]),
                        status_color(d["status"]), 0.7, 2)

        # detail only matters when something is wrong
        if d["status_detail"] and d["status"] in ("WRONG_PART", "WRONG_ORDER"):
            y = self._value(canvas, x, y + 2, d["status_detail"],
                            COLOR_CONF_LOW, 0.42, 1)
        y = self._divider(canvas, x, y + 4, inner_w)

        # ---------- look ahead ----------
        y = self._section(canvas, x, y, "Coming up next")
        if d["next_stage_name"]:
            y = self._value(canvas, x, y, d["next_stage_name"],
                            SIDEBAR_VALUE, 0.55, 1)
            if d["upcoming_action"]:
                y = self._value(canvas, x, y,
                                f"then prepare: {d['upcoming_action']}",
                                SIDEBAR_LABEL, 0.44, 1)
        else:
            y = self._value(canvas, x, y, "Final stage - nothing after this",
                            SIDEBAR_LABEL, 0.44, 1)
        y = self._divider(canvas, x, y + 4, inner_w)

        # ---------- detections ----------
        y = self._section(canvas, x, y, "Detected components")

        stats = None
        if d["object_result"] is not None:
            stats = getattr(d["object_result"], "class_stats", None)

        if stats:
            for name, st in sorted(stats.items()):
                conf = st.rolling_confidence
                col = confidence_color(conf)
                cv2.putText(canvas, name, (x, y), FONT, 0.48,
                            SIDEBAR_VALUE, 1, cv2.LINE_AA)
                cv2.putText(canvas, f"x{st.count}", (x + 200, y), FONT, 0.48,
                            SIDEBAR_LABEL, 1, cv2.LINE_AA)
                cv2.putText(canvas, f"{conf * 100:.0f}%", (x + 250, y), FONT,
                            0.48, col, 1, cv2.LINE_AA)
                bx = x + 305
                bw = inner_w - (bx - x)
                cv2.rectangle(canvas, (bx, y - 9), (bx + bw, y - 1),
                              (60, 60, 60), -1)
                cv2.rectangle(canvas, (bx, y - 9),
                              (bx + int(bw * min(conf, 1.0)), y - 1), col, -1)
                y += 22
        else:
            y = self._value(canvas, x, y, "None in view", SIDEBAR_LABEL, 0.45, 1)

        y = self._divider(canvas, x, y + 4, inner_w)

        # ---------- activity ----------
        y = self._section(canvas, x, y, "Activity")

        if d["is_warmup"]:
            act_txt = f"WARMUP ({d['warmup_remaining']})"
            act_col = COLOR_WARMUP
        else:
            act_txt = d["event_state"]
            act_col = COLOR_ACTIVE if "ACTIVE" in d["event_state"] else COLOR_IDLE

        cv2.putText(canvas, act_txt, (x, y), FONT, 0.56, act_col, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{d['event_duration']:.1f}s", (x + 210, y), FONT,
                    0.52, SIDEBAR_VALUE, 1, cv2.LINE_AA)
        y += 22
        cv2.putText(canvas, f"motion  {d['motion_result'].motion_score:.1f}%",
                    (x, y), FONT, 0.45, SIDEBAR_LABEL, 1, cv2.LINE_AA)
        y += 12
        y = self._divider(canvas, x, y, inner_w)

        # ---------- metrics ----------
        y = self._section(canvas, x, y, "Session metrics")
        m = d["metrics_summary"] or {}
        rows = [
            ("Active events", m.get("total_active_events", 0)),
            ("Avg step time", f"{m.get('avg_step_time_sec', 0.0):.2f}s"),
            ("Avg cycle time", f"{m.get('avg_cycle_time_sec', 0.0):.2f}s"),
            ("Last step", f"{m.get('last_step_time_sec', 0.0):.2f}s"),
        ]
        for label, val in rows:
            cv2.putText(canvas, label, (x, y), FONT, 0.45, SIDEBAR_LABEL,
                        1, cv2.LINE_AA)
            cv2.putText(canvas, str(val), (x + 230, y), FONT, 0.45,
                        SIDEBAR_VALUE, 1, cv2.LINE_AA)
            y += 20

        # ---------- stage preview column ----------
        if preview_image is not None:
            self._draw_preview(
                canvas, fw + SIDEBAR_WIDTH, preview_image,
                d.get("preview_title"), d.get("preview_subtitle"),
                d.get("preview_paused", False),
            )

        # Crop to what was actually drawn (never shorter than the frame).
        final_h = max(fh, y + PAD)
        return canvas[0:final_h, :]

    # ==========================================================
    # Stage preview column (V2.2)
    # ==========================================================

    def _draw_preview(self, canvas, x0, image, title, subtitle, paused):
        """Rotating view of the stage being built, drawn as its own column."""
        ch = canvas.shape[0]
        canvas[0:ch, x0:x0 + PREVIEW_WIDTH] = PREVIEW_BG

        x = x0 + PAD
        inner_w = PREVIEW_WIDTH - 2 * PAD
        y = 34

        cv2.putText(canvas, "HOW IT SHOULD LOOK", (x, y), FONT, 0.52,
                    COLOR_WHITE, 2, cv2.LINE_AA)
        y += 18
        cv2.putText(canvas, "360 view of the current step", (x, y), FONT,
                    0.4, SIDEBAR_LABEL, 1, cv2.LINE_AA)
        y += 14
        y = self._divider(canvas, x, y, inner_w)
        y += 8

        ih, iw = image.shape[:2]
        if iw > inner_w:                      # never overflow the column
            scale = inner_w / float(iw)
            image = cv2.resize(image, (inner_w, int(ih * scale)),
                               interpolation=cv2.INTER_AREA)
            ih, iw = image.shape[:2]
        x_img = x + (inner_w - iw) // 2
        if y + ih < ch:
            canvas[y:y + ih, x_img:x_img + iw] = image
            cv2.rectangle(canvas, (x_img - 1, y - 1),
                          (x_img + iw, y + ih), SIDEBAR_DIVIDER, 1)
            y += ih + 18

        if title:
            y = self._value(canvas, x, y, title, COLOR_CONF_HIGH, 0.66, 2)
        if subtitle:
            for line in self._wrap(subtitle, inner_w, 0.45, 1):
                y = self._value(canvas, x, y, line, SIDEBAR_VALUE, 0.45, 1)

        hint = "[p] pause   [,] [.] turn" if not paused else "PAUSED   [,] [.] turn"
        cv2.putText(canvas, hint, (x, y + 8), FONT, 0.4,
                    COLOR_WARMUP if paused else SIDEBAR_LABEL, 1, cv2.LINE_AA)
