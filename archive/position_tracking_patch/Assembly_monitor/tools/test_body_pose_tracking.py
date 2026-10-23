"""Offline smoke test for body-relative seat marker tracking.

Run from the Assembly_monitor project root with the project's Python:
  python tools/test_body_pose_tracking.py

Uses the saved calibration reference image and synthetic image transforms; it
opens no camera and writes no project settings. It tests image-plane tracking,
not arbitrary 3-D pose estimation or real-camera accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision.position_validator import PositionConfig, PositionValidator  # noqa: E402


class Detection:
    def __init__(self, class_name, bounding_box, confidence=0.95):
        self.class_name = class_name
        self.bounding_box = bounding_box
        self.confidence = confidence


def transform_box(box, matrix):
    x1, y1, x2, y2 = box
    corners = np.float32([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
    ]).reshape(-1, 1, 2)
    mapped = cv2.transform(corners, matrix).reshape(-1, 2)
    return (
        int(np.floor(mapped[:, 0].min())),
        int(np.floor(mapped[:, 1].min())),
        int(np.ceil(mapped[:, 0].max())),
        int(np.ceil(mapped[:, 1].max())),
    )


def make_detections(body_box, seat1, seat2=None):
    output = [Detection("body_a", body_box)]
    if seat1 is not None:
        x, y = seat1
        output.append(Detection("bushing", (int(x - 5), int(y - 5), int(x + 5), int(y + 5))))
    if seat2 is not None:
        x, y = seat2
        output.append(Detection("bushing", (int(x - 5), int(y - 5), int(x + 5), int(y + 5))))
    return output


def get_marker(markers, name):
    return next(m for m in markers if m.get("name") == name)


def main():
    config_path = PROJECT_ROOT / "config" / "position_config.json"
    config = PositionConfig.load(str(config_path))
    if not config.is_usable:
        raise RuntimeError(f"Position config isn't usable: {config_path}")
    if not config.pose_tracking_enabled:
        raise RuntimeError("pose_tracking is not enabled in position_config.json")

    # Safety regression: seat acceptance regions must not overlap.
    # This caught a real config bug where max(18 px, 45% of spacing)
    # yielded an 18 px gate for seats only 30.1 px apart.
    raw_cfg_for_gate = json.loads(config_path.read_text(encoding="utf-8"))
    anchor_px = [
        np.asarray(spec["anchor_px_at_calibration"], dtype=float)
        for spec in raw_cfg_for_gate["seats"].values()
    ]
    if len(anchor_px) >= 2:
        separation = float(np.linalg.norm(anchor_px[0] - anchor_px[1]))
        gate = float(raw_cfg_for_gate["max_anchor_distance_px"])
        assert gate < separation / 2.0, (
            f"seat gate {gate:.1f}px overlaps neighbouring seats "
            f"({separation:.1f}px apart)"
        )
    if not config.reference_image_path or not Path(config.reference_image_path).is_file():
        raise RuntimeError(f"Reference image not found: {config.reference_image_path}")

    image = cv2.imread(config.reference_image_path)
    if image is None:
        raise RuntimeError(f"Could not read reference image: {config.reference_image_path}")
    validator = PositionValidator(config)
    tracker = validator._pose_tracker
    if tracker is None or tracker._ref_box is None:
        raise RuntimeError("BodyPoseTracker did not initialise from the calibration reference.")

    # Exact calibration anchors, in native 1280x720 reference-frame coordinates.
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    anchors = {
        name: tuple(spec["anchor_px_at_calibration"])
        for name, spec in raw["seats"].items()
    }
    body_box = tracker._ref_box

    # 1) Reference view: marker positions should match calibration anchors.
    detections = make_detections(body_box, anchors["Seat_1"])
    for _ in range(config.stable_frames_correct + 2):
        result = validator.update("Stage_01", detections, image)
    assert result.pose_valid, f"reference view failed: {result.pose_detail}"
    assert result.verdict == "CORRECT", f"expected CORRECT, got {result.verdict}"
    assert validator.gate("Stage_01")[0] is True, "correct-seat stage gate should allow"
    markers = validator.seat_markers(image.shape)
    p1 = get_marker(markers, "Seat_1")["xy"]
    assert np.linalg.norm(np.asarray(p1) - anchors["Seat_1"]) < 4, (p1, anchors["Seat_1"])
    print("PASS 1/4: markers line up with both calibration anchors in reference view")

    # 2) Translation: body, image features and YOLO bbox move together.
    translation = np.float32([[1, 0, 100], [0, 1, 50]])
    moved = cv2.warpAffine(image, translation, (image.shape[1], image.shape[0]))
    moved_box = tuple(v + (100 if i in (0, 2) else 50) for i, v in enumerate(body_box))
    moved_seat1 = (anchors["Seat_1"][0] + 100, anchors["Seat_1"][1] + 50)
    detections = make_detections(moved_box, moved_seat1)
    result = validator.update("Stage_01", detections, moved)
    assert result.pose_valid, f"translation tracking failed: {result.pose_detail}"
    markers = validator.seat_markers(moved.shape)
    p1 = get_marker(markers, "Seat_1")["xy"]
    assert np.linalg.norm(np.asarray(p1) - moved_seat1) < 5, (p1, moved_seat1)
    print("PASS 2/4: markers follow a +100 px / +50 px body translation")

    # 3) Modest in-plane rotation: markers should follow the affine image change.
    centre = (image.shape[1] / 2.0, image.shape[0] / 2.0)
    rotation = cv2.getRotationMatrix2D(centre, 4.0, 1.0)
    rotated = cv2.warpAffine(moved, rotation, (image.shape[1], image.shape[0]))
    rotated_box = transform_box(moved_box, rotation)
    moved_seat1_arr = np.array([[moved_seat1]], dtype=np.float32)
    expected_rotated = cv2.transform(moved_seat1_arr, rotation)[0, 0]
    detections = make_detections(rotated_box, expected_rotated)
    result = validator.update("Stage_01", detections, rotated)
    assert result.pose_valid, f"in-plane rotation tracking failed: {result.pose_detail}"
    markers = validator.seat_markers(rotated.shape)
    p1 = get_marker(markers, "Seat_1")["xy"]
    assert np.linalg.norm(np.asarray(p1) - expected_rotated) < 8, (p1, expected_rotated)
    print("PASS 3/4: markers follow a modest 4-degree in-plane rotation")

    # 4) Loss of body detection: never leave stale markers and block Stage_01.
    result = validator.update("Stage_01", [], rotated)
    assert not result.pose_valid and result.awaiting_pose, result
    assert validator.gate("Stage_01")[0] is False, "Stage_01 should wait while body pose is lost"
    markers = validator.seat_markers(rotated.shape)
    assert len(markers) == 1 and markers[0].get("pose_valid") is False, markers
    print("PASS 4/4: loss of body tracking hides stale markers and blocks Stage_01")
    print("ALL BODY-RELATIVE TRACKING SMOKE TESTS PASSED")
    print("LIMIT: synthetic image-plane transforms only; no claim of arbitrary 3-D pose tracking or live-camera accuracy.")


if __name__ == "__main__":
    main()
