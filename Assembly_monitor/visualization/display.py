"""
visualization/display.py
================================
SUBSYSTEM: Visualization (SS9)

LAYOUT CHANGE (V2.2):
  Previously every panel was drawn ON TOP of the 640x480 camera frame:
  status top-left, YOLO confidence top-right, metrics bottom-left, and
  the assembly stage panel top-centre. With four panels competing for
  the same pixels, the stage panel overlapped the other two and long
  warning text ran underneath the confidence panel.

  The frame is now composited into a larger canvas:

      +---------------------------+------------------+-------------+
      |                           |                  | STAGE       |
      |     CAMERA FRAME          |    INFO PANEL    | PREVIEWS    |
      |     640 x 480             |    420 px wide   | current     |
      |     boxes + ROI only      |    all text      | + next      |
      |                           |                  | rotating    |
      +---------------------------+------------------+-------------+

  The preview column (V2.2) shows the stage the operator is building now,
  rendered from the same CAD as the training data and rotating through
  360 degrees so the fit can be seen from every side. When a following
  stage exists, a second compact 360-degree view is shown underneath so
  the operator can prepare the next step. It is omitted automatically
  when no preview frames are available.

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

from config.settings import STAGE_PREVIEW_STACK_IMAGE_SIZE

try:
    from config.settings import DISPLAY_VIDEO_SCALE as _VIDEO_SCALE
except ImportError:          # older settings.py: behave exactly as before
    _VIDEO_SCALE = 1.0
try:
    from config.settings import DISPLAY_PREVIEW_BELOW as _PREVIEW_BELOW
except ImportError:
    _PREVIEW_BELOW = False


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

SIDEBAR_WIDTH = 480   # widened from 420 to fit the larger text below
PAD = 16                            # left/right padding inside sidebar

# Stage preview column (V2.2)
PREVIEW_WIDTH = 360
# Size of each preview image when the two sit side by side under
# the video rather than stacked in a column.
PREVIEW_IMAGE_IN_ROW = 240
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
        "WRONG_POSITION": COLOR_CONF_LOW,
        "WAITING_POSITION": COLOR_CONF_MEDIUM,
        "FOREIGN_OBJECT": COLOR_CONF_LOW,
    }.get(status, COLOR_WHITE)


def status_label(status):
    """Operator-readable wording for the status value."""
    return {
        "OK": "CORRECT",
        "WAITING": "WAITING",
        "WRONG_PART": "WRONG PART",
        "WRONG_ORDER": "WRONG ORDER",
        "WRONG_POSITION": "WRONG POSITION",
        "WAITING_POSITION": "CHECKING THE HOLE",
        "FOREIGN_OBJECT": "UNKNOWN OBJECT",
    }.get(status, str(status))

# Outlines for work-area regions nothing accounts for.
COLOR_FOREIGN = (0, 90, 255)          # orange-red, alerting
COLOR_FOREIGN_WATCH = (120, 160, 190) # still gathering evidence


# Statuses that mean a mistake the operator must fix.
PROBLEM_STATUSES = ("WRONG_PART", "WRONG_ORDER", "WRONG_POSITION")

# Every status whose detail line is worth printing. WAITING_POSITION is not
# a mistake - nothing is wrong, the system has not yet seen the seat filled
# - but its detail is the most useful line on the panel, because it names
# exactly what the system is waiting for. Kept as one list so a new status
# cannot be added to the colour map and then silently lose its explanation.
DETAIL_STATUSES = PROBLEM_STATUSES + ("WAITING_POSITION",)

# Seat markers drawn on the frame (position layer).
COLOR_SEAT_REQUIRED = (0, 220, 0)      # the seat this stage needs
COLOR_SEAT_WRONG = (0, 0, 255)         # a seat holding a part it should not
COLOR_SEAT_IDLE = (150, 150, 150)      # a seat not involved in this stage


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
        next_preview_image=None,
        next_preview_title=None,
        next_preview_subtitle=None,
        preview_paused=False,
        # ---- position layer, all optional so older callers still work ----
        position_verdict=None,      # "CORRECT" | "WRONG" | "UNKNOWN" | None
        position_message=None,      # operator wording for the red banner
        position_detail=None,       # short engineering wording
        position_stable=False,
        seat_markers=None,          # [{name,label,xy,occupied,required,wrong}]
        # ---- unknown object layer, optional ----
        foreign_alert=False,
        foreign_message=None,
        foreign_markers=None,       # [{bbox,age_sec,alerting}]
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

        # Seat markers belong on the frame for the same reason the boxes do:
        # they point at specific pixels. They are also how the operator can
        # see which hole the system means, and how a bad calibration becomes
        # obvious instead of producing mysterious warnings.
        if seat_markers:
            self._draw_seat_markers(annotated, seat_markers)

        if foreign_markers:
            self._draw_foreign_markers(annotated, foreign_markers)

        # Banners, stacked. Wrong part comes first: it is about the thing
        # in the operator's hand right now, which is more urgent than a
        # stray object sitting on the bench.
        banner_y = 0
        if status in PROBLEM_STATUSES and status_detail:
            banner_y = self._draw_banner(
                annotated, status_label(status), status_detail, banner_y)
        if foreign_alert and foreign_message:
            banner_y = self._draw_banner(
                annotated, "UNKNOWN OBJECT", foreign_message, banner_y)

        # The wrong-position banner sits on the frame, not in the sidebar.
        # An operator looking at their hands is looking at the video, and a
        # message about the thing under their hands has to be where their
        # eyes already are. The sidebar keeps the detailed version.
        if position_verdict == "WRONG" and position_stable and position_message:
            self._draw_position_banner(annotated, position_message)

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
            next_preview_image=next_preview_image,
            next_preview_title=next_preview_title,
            next_preview_subtitle=next_preview_subtitle,
            preview_paused=preview_paused,
            position_verdict=position_verdict,
            position_message=position_message,
            position_detail=position_detail,
            position_stable=position_stable,
            seat_markers=seat_markers,
            foreign_alert=foreign_alert,
            foreign_message=foreign_message,
            foreign_markers=foreign_markers,
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
    # Position layer: seat markers + wrong-position banner
    # ==========================================================


    def _draw_foreign_markers(self, frame, markers):
        """Outline whatever the system cannot account for.

        The box is most of the value here. The message can only say that
        something unrecognised is present - it cannot name it - so the
        outline is what tells the operator WHICH thing to pick up.
        Regions still gathering evidence are drawn faintly, so the check
        is visibly working rather than a box appearing from nowhere.
        """
        for m in markers or []:
            try:
                x, y, w, h = (int(v) for v in m["bbox"])
            except (KeyError, TypeError, ValueError):
                continue
            alerting = bool(m.get("alerting"))
            colour = COLOR_FOREIGN if alerting else COLOR_FOREIGN_WATCH
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour,
                          2 if alerting else 1)
            if not alerting:
                continue
            tag = "UNKNOWN"
            (tw, th), _ = cv2.getTextSize(tag, FONT, 0.46, 1)
            ty = y - 6 if y - 6 - th > 0 else y + h + th + 8
            cv2.rectangle(frame, (x, ty - th - 5), (x + tw + 8, ty + 4),
                          COLOR_BLACK, -1)
            cv2.putText(frame, tag, (x + 4, ty), FONT, 0.46, colour, 1,
                        cv2.LINE_AA)

    def _draw_banner(self, frame, title, message, y_offset=0):
        """Red strip across the video: the thing to fix, now.

        An operator working is looking at their hands, which means they
        are looking at the VIDEO, not at a sidebar. A warning that only
        appears as small text beside the picture is a warning they will
        miss, so every condition that needs acting on gets a banner here.

        Returns the y the next banner should start at, so two warnings
        stack instead of one painting over the other. An operator who has
        both put a stray tool down AND picked up the wrong part needs to
        be told both things, not whichever happened to be drawn last.
        """
        fh, fw = frame.shape[:2]
        pad = 10
        lines = self._wrap(message, fw - 2 * pad - 8, 0.56, 1)
        top = y_offset
        bottom = min(top + 34 + 22 * len(lines) + 2 * pad, fh)
        cv2.rectangle(frame, (0, top), (fw, bottom), (0, 0, 150), -1)
        cv2.rectangle(frame, (0, top), (fw, bottom), COLOR_CONF_LOW, 2)
        y = top + pad + 24
        cv2.putText(frame, title, (pad + 4, y), FONT, 0.86,
                    COLOR_WHITE, 2, cv2.LINE_AA)
        y += 8
        for line in lines:
            y += 22
            if y > bottom - 4:
                break
            cv2.putText(frame, line, (pad + 4, y), FONT, 0.56,
                        COLOR_WHITE, 1, cv2.LINE_AA)
        return bottom

    def _draw_seat_markers(self, frame, markers):
        """Draw only live body-relative markers; never draw stale screen points."""
        if markers and any(m.get("pose_valid") is False for m in markers):
            # A lost/uncertain body pose is not a seat marker. Hiding markers is
            # safer than showing the old calibration points on the bench.
            message = "BODY NOT TRACKED - POSITION CHECK PAUSED"
            fh, fw = frame.shape[:2]
            bar_h = 42
            cv2.rectangle(frame, (0, 0), (fw, bar_h), (0, 105, 145), -1)
            cv2.rectangle(frame, (0, 0), (fw, bar_h), (0, 190, 255), 2)
            cv2.putText(frame, message, (12, 28), FONT, 0.68,
                        COLOR_WHITE, 2, cv2.LINE_AA)
            return

        for m in markers:
            try:
                x, y = int(m["xy"][0]), int(m["xy"][1])
            except (KeyError, TypeError, IndexError, ValueError):
                continue

            if m.get("wrong"):
                colour = COLOR_SEAT_WRONG
            elif m.get("required"):
                colour = COLOR_SEAT_REQUIRED
            else:
                colour = COLOR_SEAT_IDLE

            cv2.circle(frame, (x, y), 16, colour, 2)
            if m.get("occupied"):
                cv2.circle(frame, (x, y), 5, colour, -1)
            else:
                cv2.drawMarker(frame, (x, y), colour,
                               cv2.MARKER_CROSS, 10, 1)

            tag = str(m.get("label") or m.get("name") or "")
            if not tag:
                continue
            if m.get("required"):
                tag += "  <- here"
            (tw, th), _ = cv2.getTextSize(tag, FONT, 0.52, 1)
            tx = max(2, min(x - tw // 2, frame.shape[1] - tw - 4))
            ty = y + 34 if y + 34 + th < frame.shape[0] else y - 24
            cv2.rectangle(frame, (tx - 3, ty - th - 4), (tx + tw + 3, ty + 3),
                          COLOR_BLACK, -1)
            cv2.putText(frame, tag, (tx, ty), FONT, 0.42, colour, 1,
                        cv2.LINE_AA)

    def _draw_position_banner(self, frame, message):
        """Red banner across the top of the video: the thing to fix, now.

        Drawn over a solid strip rather than blended, so it stays legible
        against a bright bench, and wrapped so a full sentence fits instead
        of being cut off at the frame edge.
        """
        fh, fw = frame.shape[:2]
        pad = 10
        max_w = fw - 2 * pad - 8

        lines = ["WRONG POSITION"] + self._wrap(message, max_w, 0.52, 1)
        heights = [30] + [20] * (len(lines) - 1)
        banner_h = sum(heights) + 2 * pad

        cv2.rectangle(frame, (0, 0), (fw, banner_h), (0, 0, 150), -1)
        cv2.rectangle(frame, (0, 0), (fw, banner_h), COLOR_CONF_LOW, 2)

        y = pad + 22
        cv2.putText(frame, lines[0], (pad + 4, y), FONT, 0.78,
                    COLOR_WHITE, 2, cv2.LINE_AA)
        y += 8
        for line in lines[1:]:
            y += 20
            cv2.putText(frame, line, (pad + 4, y), FONT, 0.52,
                        COLOR_WHITE, 1, cv2.LINE_AA)

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
        cv2.putText(canvas, heading.upper(), (x, y), FONT, 0.50,
                    SIDEBAR_LABEL, 1, cv2.LINE_AA)
        return y + 26

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

        # Shrink ONLY the video, never the panels.
        #
        # The camera frame is 1280 wide, which with the sidebar and the
        # preview column makes a 2060 px canvas - wider than most screens.
        # Scaling the whole finished canvas down to fit was the obvious
        # fix and the wrong one: it shrinks the text too, and 0.42-scale
        # OpenCV text resampled to 79 % is genuinely unreadable.
        #
        # The video is the one part that loses nothing from being smaller -
        # it is a photograph, and the bounding boxes drawn on it scale with
        # it. So the frame is reduced here, before the canvas is sized, and
        # the sidebar and previews are then composited at their native
        # resolution and stay crisp.
        if _VIDEO_SCALE != 1.0 and annotated is not None:
            annotated = cv2.resize(
                annotated, None, fx=_VIDEO_SCALE, fy=_VIDEO_SCALE,
                interpolation=cv2.INTER_AREA)

        fh, fw = annotated.shape[:2]
        preview_image = d.get("preview_image")
        next_preview_image = d.get("next_preview_image")
        has_preview = preview_image is not None or next_preview_image is not None
        # WHERE THE PREVIEWS GO, AND WHY IT MATTERS FOR THE VIDEO SIZE.
        #
        # A 16:9 camera frame is wide and short; the sidebar is narrow and
        # tall. Putting the previews in a THIRD column beside them meant
        # the canvas had to be as tall as the sidebar (~900 px) while the
        # video was only ~430 px tall, so nearly half the window was empty
        # black space under the picture - and the video had to be shrunk
        # to make room for a column it was not even sharing height with.
        #
        # Putting the previews UNDER the video instead fills that dead
        # space with something useful and hands the freed width back to
        # the picture. Same screen, substantially larger video, and the
        # text stays at full resolution because nothing is scaled down to
        # fit any more.
        preview_below = bool(_PREVIEW_BELOW) and has_preview
        preview_w = 0 if preview_below else (PREVIEW_WIDTH if has_preview else 0)
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
                                SIDEBAR_LABEL, 0.52, 1)

        # The live requirement, printed straight from the state machine
        # so the panel can never claim something different from what the
        # logic is actually testing.
        if d.get("current_requirement"):
            req = ", ".join(f"{k} x{v}" for k, v in d["current_requirement"].items())
            y = self._value(canvas, x, y, f"needs: {req}", SIDEBAR_LABEL, 0.52, 1)

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
        if d["status_detail"] and d["status"] in DETAIL_STATUSES:
            y = self._value(canvas, x, y + 2, d["status_detail"],
                            status_color(d["status"]), 0.52, 1)
        y = self._divider(canvas, x, y + 4, inner_w)

        # ---------- part position ----------
        # Shown as its own line rather than folded into Status, because it
        # answers a different question. Status says whether the right parts
        # are present in the right order; this says whether the part that
        # is present went into the right hole. A stage can be perfect on
        # one and wrong on the other, and collapsing them would hide that.
        pos_verdict = d.get("position_verdict")
        if pos_verdict and pos_verdict != "NOT_APPLICABLE":
            y = self._section(canvas, x, y, "Part position")

            stable = bool(d.get("position_stable"))
            markers = d.get("seat_markers") or []
            pose_untracked = bool(markers and markers[0].get("pose_valid") is False)
            if pose_untracked:
                text, colour = "BODY NOT TRACKED", COLOR_CONF_MEDIUM
            elif pos_verdict == "WRONG" and stable:
                text, colour = "WRONG HOLE", COLOR_CONF_LOW
            elif pos_verdict == "CORRECT" and stable:
                text, colour = "CORRECT SEAT", COLOR_CONF_HIGH
            elif pos_verdict in ("WRONG", "CORRECT"):
                text, colour = "checking...", COLOR_CONF_MEDIUM
            else:
                text, colour = "not visible", SIDEBAR_LABEL

            y = self._value(canvas, x, y, text, colour, 0.62, 2)

            if pos_verdict == "WRONG" and stable and d.get("position_message"):
                y = self._value(canvas, x, y + 2, d["position_message"],
                                COLOR_WHITE, 0.52, 1)
            elif d.get("position_detail"):
                y = self._value(canvas, x, y + 2, d["position_detail"],
                                SIDEBAR_LABEL, 0.4, 1)

            # The instruction stays on screen while the mistake is shown, so
            # the operator is never left with a complaint and no remedy.
            if pos_verdict == "WRONG" and stable:
                y = self._value(
                    canvas, x, y + 2,
                    "The preview on the right shows the correct hole.",
                    COLOR_CONF_MEDIUM, 0.52, 1)

            y = self._divider(canvas, x, y + 4, inner_w)


        # ---------- work area ----------
        # Separate from Status because it is a different kind of fact.
        # Status is about the assembly; this is about what else is on the
        # bench. The detector cannot say WHAT the object is - only that
        # nothing accounts for it - so the wording promises no more.
        if d.get("foreign_alert"):
            y = self._section(canvas, x, y, "Work area")
            y = self._value(canvas, x, y, "UNKNOWN OBJECT", COLOR_CONF_LOW,
                            0.62, 2)
            if d.get("foreign_message"):
                y = self._value(canvas, x, y + 2, d["foreign_message"],
                                COLOR_WHITE, 0.52, 1)
            y = self._value(canvas, x, y + 2, "Outlined on the video.",
                            SIDEBAR_LABEL, 0.52, 1)
            y = self._divider(canvas, x, y + 4, inner_w)

        # ---------- look ahead ----------
        y = self._section(canvas, x, y, "Coming up next")
        if d["next_stage_name"]:
            y = self._value(canvas, x, y, d["next_stage_name"],
                            SIDEBAR_VALUE, 0.55, 1)
            if d["upcoming_action"]:
                y = self._value(canvas, x, y,
                                f"then prepare: {d['upcoming_action']}",
                                SIDEBAR_LABEL, 0.52, 1)
        else:
            y = self._value(canvas, x, y, "Final stage - nothing after this",
                            SIDEBAR_LABEL, 0.52, 1)
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
                cv2.putText(canvas, name, (x, y), FONT, 0.56,
                            SIDEBAR_VALUE, 1, cv2.LINE_AA)
                cv2.putText(canvas, f"x{st.count}", (x + 230, y), FONT, 0.56,
                            SIDEBAR_LABEL, 1, cv2.LINE_AA)
                cv2.putText(canvas, f"{conf * 100:.0f}%", (x + 285, y), FONT,
                            0.56, col, 1, cv2.LINE_AA)
                bx = x + 355
                bw = inner_w - (bx - x)
                cv2.rectangle(canvas, (bx, y - 9), (bx + bw, y - 1),
                              (60, 60, 60), -1)
                cv2.rectangle(canvas, (bx, y - 9),
                              (bx + int(bw * min(conf, 1.0)), y - 1), col, -1)
                y += 26
        else:
            y = self._value(canvas, x, y, "None in view", SIDEBAR_LABEL, 0.52, 1)

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
        cv2.putText(canvas, f"{d['event_duration']:.1f}s", (x + 240, y), FONT,
                    0.58, SIDEBAR_VALUE, 1, cv2.LINE_AA)
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
            y += 24

        # ---------- stage preview column ----------
        preview_bottom = fh
        if preview_below:
            preview_bottom = self._draw_preview_row(
                canvas, fh, fw,
                preview_image, d.get("preview_title"), d.get("preview_subtitle"),
                next_preview_image, d.get("next_preview_title"),
                d.get("next_preview_subtitle"),
                d.get("preview_paused", False),
            )
        elif preview_image is not None and next_preview_image is not None:
            preview_bottom = self._draw_preview_stack(
                canvas, fw + SIDEBAR_WIDTH,
                preview_image,
                d.get("preview_title"),
                d.get("preview_subtitle"),
                next_preview_image,
                d.get("next_preview_title"),
                d.get("next_preview_subtitle"),
                d.get("preview_paused", False),
            )
        elif preview_image is not None:
            # Preserve the original single-preview layout for Stage_07, when
            # there is no next stage.
            preview_bottom = self._draw_preview(
                canvas, fw + SIDEBAR_WIDTH, preview_image,
                d.get("preview_title"), d.get("preview_subtitle"),
                d.get("preview_paused", False),
            )
        elif next_preview_image is not None:
            preview_bottom = self._draw_preview_stack(
                canvas, fw + SIDEBAR_WIDTH,
                None, None, None,
                next_preview_image,
                d.get("next_preview_title"),
                d.get("next_preview_subtitle"),
                d.get("preview_paused", False),
            )

        # Crop to what was actually drawn (never shorter than the frame).
        final_h = max(fh, y + PAD, preview_bottom + PAD)
        return canvas[0:final_h, :]

    # ==========================================================
    # Stage preview column (V2.2)
    # ==========================================================


    def _draw_preview_row(self, canvas, y0, width,
                          cur_image, cur_title, cur_subtitle,
                          nxt_image, nxt_title, nxt_subtitle, paused):
        """Current and next stage side by side, in the band under the video.

        This band is the space the old three-column layout wasted. Laying
        the two previews across it rather than stacking them in a tall
        narrow column means each one gets a square-ish area instead of a
        letterbox, and the video keeps the width the column used to take.
        """
        panels = [(cur_image, cur_title, cur_subtitle, "CURRENT STEP",
                   COLOR_CONF_HIGH),
                  (nxt_image, nxt_title, nxt_subtitle, "COMING NEXT",
                   COLOR_CONF_MEDIUM)]
        panels = [p for p in panels if p[0] is not None]
        if not panels:
            return y0

        # TEXT BESIDE THE IMAGE, NOT UNDER IT.
        #
        # Stacking the label under a 240 px render left the cell's other
        # ~270 px of width empty and forced the type down to 0.48, which is
        # what made this panel unreadable on the bench. Putting the render
        # on the left and the label to its right uses width that was going
        # to waste, so the type can be half again as large WITHOUT making
        # the window any taller - and window height is the binding
        # constraint here, because anything taller than the screen gets
        # scaled down and the text shrinks right back again.
        cell_w = width // len(panels)
        image_size = max(120, min(PREVIEW_IMAGE_IN_ROW,
                                  int(cell_w * 0.46), cell_w - 3 * PAD))
        bottom = y0

        cv2.rectangle(canvas, (0, y0), (width, canvas.shape[0]), PREVIEW_BG, -1)
        cv2.line(canvas, (0, y0), (width, y0), SIDEBAR_DIVIDER, 1)

        for i, (image, title, subtitle, heading, colour) in enumerate(panels):
            x = i * cell_w + PAD
            y = y0 + PAD + 22

            cv2.putText(canvas, heading, (x, y), FONT, 0.68,
                        COLOR_WHITE, 2, cv2.LINE_AA)
            y += 14
            cv2.line(canvas, (x, y), (x + cell_w - 2 * PAD, y),
                     SIDEBAR_DIVIDER, 1)
            y += 12

            ih, iw = image.shape[:2]
            scale = min(image_size / float(max(iw, 1)),
                        image_size / float(max(ih, 1)))
            if abs(scale - 1.0) > 1e-6:
                image = cv2.resize(
                    image,
                    (max(1, int(iw * scale)), max(1, int(ih * scale))),
                    interpolation=(cv2.INTER_AREA if scale < 1.0
                                   else cv2.INTER_LINEAR))
                ih, iw = image.shape[:2]

            image_top = y
            if y + ih <= canvas.shape[0] and x + iw <= canvas.shape[1]:
                canvas[y:y + ih, x:x + iw] = image
                cv2.rectangle(canvas, (x - 1, y - 1), (x + iw, y + ih),
                              SIDEBAR_DIVIDER, 1)
            else:
                iw = ih = 0

            # Text column: to the right of the render, top-aligned with it.
            tx = x + iw + (PAD if iw else 0)
            tw = max(80, (i * cell_w + cell_w - PAD) - tx)
            ty = image_top + 26

            if title:
                for line in self._wrap(title, tw, 0.82, 2):
                    cv2.putText(canvas, line, (tx, ty), FONT, 0.82, colour, 2,
                                cv2.LINE_AA)
                    ty += 30
                ty += 6
            if subtitle:
                for line in self._wrap(subtitle, tw, 0.60, 1):
                    cv2.putText(canvas, line, (tx, ty), FONT, 0.60,
                                SIDEBAR_VALUE, 1, cv2.LINE_AA)
                    ty += 25

            bottom = max(bottom, image_top + ih, ty)

        hint = "[p] pause   [,] [.] turn" if not paused else "PAUSED   [,] [.] turn"
        cv2.putText(canvas, hint, (PAD, bottom + 20), FONT, 0.56,
                    COLOR_WARMUP if paused else SIDEBAR_LABEL, 1, cv2.LINE_AA)
        return bottom + 30

    def _draw_preview_stack(
        self, canvas, x0,
        current_image, current_title, current_subtitle,
        next_image, next_title, next_subtitle, paused,
    ):
        """Draw current + next stage previews vertically in one column."""
        ch = canvas.shape[0]
        canvas[0:ch, x0:x0 + PREVIEW_WIDTH] = PREVIEW_BG

        x = x0 + PAD
        inner_w = PREVIEW_WIDTH - 2 * PAD
        image_size = STAGE_PREVIEW_STACK_IMAGE_SIZE
        y = 34

        if current_image is not None:
            y = self._draw_compact_preview(
                canvas, x, y, current_image, current_title, current_subtitle,
                heading="CURRENT STEP", title_colour=COLOR_CONF_HIGH,
                image_size=image_size,
            )

        if next_image is not None:
            if current_image is not None:
                y += 10
                cv2.line(canvas, (x, y), (x + inner_w, y), SIDEBAR_DIVIDER, 1)
                y += 24

            y = self._draw_compact_preview(
                canvas, x, y, next_image, next_title, next_subtitle,
                heading="COMING NEXT", title_colour=COLOR_CONF_MEDIUM,
                image_size=image_size,
            )

        # One hint controls both previews because they share the same player
        # pause/manual-rotation state.
        hint = "[p] pause   [,] [.] turn" if not paused else "PAUSED   [,] [.] turn"
        cv2.putText(canvas, hint, (x, y + 8), FONT, 0.4,
                    COLOR_WARMUP if paused else SIDEBAR_LABEL, 1, cv2.LINE_AA)
        return y + 18

    def _draw_compact_preview(
        self, canvas, x, y, image, title, subtitle,
        heading, title_colour, image_size,
    ):
        """Draw one compact preview block and return the next y position."""
        inner_w = PREVIEW_WIDTH - 2 * PAD

        cv2.putText(canvas, heading, (x, y), FONT, 0.48,
                    COLOR_WHITE, 2, cv2.LINE_AA)
        y += 18
        y = self._divider(canvas, x, y, inner_w)
        y += 8

        ih, iw = image.shape[:2]
        scale = min(
            image_size / float(max(iw, 1)),
            image_size / float(max(ih, 1)),
        )
        if abs(scale - 1.0) > 1e-6:
            image = cv2.resize(
                image,
                (max(1, int(iw * scale)), max(1, int(ih * scale))),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
            )
            ih, iw = image.shape[:2]

        x_img = x + (inner_w - iw) // 2
        if y + ih <= canvas.shape[0]:
            canvas[y:y + ih, x_img:x_img + iw] = image
            cv2.rectangle(canvas, (x_img - 1, y - 1),
                          (x_img + iw, y + ih), SIDEBAR_DIVIDER, 1)
            y += ih + 12

        if title:
            y = self._value(canvas, x, y, title, title_colour, 0.60, 2)
        if subtitle:
            y = self._value(canvas, x, y, subtitle, SIDEBAR_VALUE, 0.52, 1)

        return y

    def _draw_preview(self, canvas, x0, image, title, subtitle, paused):
        """Rotating view of one stage, using the original single-panel layout."""
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
        if iw > inner_w:
            scale = inner_w / float(iw)
            image = cv2.resize(image, (inner_w, int(ih * scale)),
                               interpolation=cv2.INTER_AREA)
            ih, iw = image.shape[:2]
        x_img = x + (inner_w - iw) // 2
        if y + ih <= ch:
            canvas[y:y + ih, x_img:x_img + iw] = image
            cv2.rectangle(canvas, (x_img - 1, y - 1),
                          (x_img + iw, y + ih), SIDEBAR_DIVIDER, 1)
            y += ih + 18

        if title:
            y = self._value(canvas, x, y, title, COLOR_CONF_HIGH, 0.66, 2)
        if subtitle:
            for line in self._wrap(subtitle, inner_w, 0.52, 1):
                y = self._value(canvas, x, y, line, SIDEBAR_VALUE, 0.52, 1)

        hint = "[p] pause   [,] [.] turn" if not paused else "PAUSED   [,] [.] turn"
        cv2.putText(canvas, hint, (x, y + 8), FONT, 0.4,
                    COLOR_WARMUP if paused else SIDEBAR_LABEL, 1, cv2.LINE_AA)
        return y + 18
