"""Body-relative seat tracking for the assembly position validator.

The previous implementation stored Seat_1 and Seat_2 as absolute image
coordinates. Those points do not move with body_a, so markers stayed on the
bench when the operator picked up the part.

This tracker uses the saved calibration image as a reference, segments the
body_a-coloured mesh, matches visual features inside the current YOLO body_a
box, and estimates a reference-to-current homography. Seat anchors are
transformed by that homography every frame.

This supports image translation, scale, in-plane rotation and moderate
perspective change. It is NOT full 6-DoF pose estimation: if body_a is rotated
to a substantially different 3-D face/view, the tracker intentionally returns
invalid rather than displaying misleading seat positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple
import math
import time

import cv2
import numpy as np


@dataclass
class PoseTrackResult:
    valid: bool
    anchors_px: Dict[str, Tuple[float, float]]
    reason: str
    inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error_px: float = float("inf")
    body_box: Optional[Tuple[int, int, int, int]] = None


class BodyPoseTracker:
    """Estimate current seat pixels from image features on body_a."""

    def __init__(
        self,
        reference_image_path: str,
        anchors_px: Dict[str, Tuple[float, float]],
        *,
        min_inliers: int = 8,
        min_inlier_ratio: float = 0.22,
        max_reprojection_error_px: float = 6.0,
        feature_count: int = 900,
    ) -> None:
        self.reference_image_path = str(reference_image_path or "")
        self.reference_anchors = {
            str(k): (float(v[0]), float(v[1])) for k, v in anchors_px.items()
        }
        self.min_inliers = int(min_inliers)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        self.feature_count = int(feature_count)
        self._last_anchors: Optional[Dict[str, Tuple[float, float]]] = None
        self._last_valid = False
        self._initialization_error: Optional[str] = None
        self._prev_keypoints = None
        self._prev_descriptors = None
        self._prev_box = None
        self._last_success_time = 0.0

        self._ref_image = None
        self._ref_box = None
        self._ref_keypoints = None
        self._ref_descriptors = None
        self._detector = None
        self._norm = None
        self._initialize()

    @staticmethod
    def _green_mask(bgr: np.ndarray) -> np.ndarray:
        """Mask the dark teal/green printed body, excluding skin/paper/background."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # The project's green body is roughly H=75-100 with high saturation.
        # Keep low V because the dark side faces are important feature regions.
        mask = cv2.inRange(hsv, np.array([48, 38, 3], np.uint8),
                          np.array([112, 255, 230], np.uint8))
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        return mask

    @staticmethod
    def _bbox_from_reference(mask: np.ndarray,
                             anchors: Dict[str, Tuple[float, float]]) -> Optional[Tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        ax = float(np.mean([p[0] for p in anchors.values()]))
        ay = float(np.mean([p[1] for p in anchors.values()]))
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            # Prefer the coloured object surrounding the manually selected seats.
            dx = max(x - ax, 0.0, ax - (x + w))
            dy = max(y - ay, 0.0, ay - (y + h))
            distance = math.hypot(dx, dy)
            score = distance - min(math.sqrt(area) * 0.035, 25.0)
            candidates.append((score, area, (x, y, x + w, y + h)))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], -item[1]))
        x1, y1, x2, y2 = candidates[0][2]
        pad = max(6, int(round(max(x2 - x1, y2 - y1) * 0.035)))
        h, w = mask.shape[:2]
        return (max(0, x1 - pad), max(0, y1 - pad),
                min(w, x2 + pad), min(h, y2 + pad))

    def _create_feature_detector(self):
        # SIFT handles scale changes better; ORB is a safe fallback on
        # OpenCV builds that do not ship SIFT.
        if hasattr(cv2, "SIFT_create"):
            self._norm = cv2.NORM_L2
            return cv2.SIFT_create(nfeatures=self.feature_count, contrastThreshold=0.015)
        self._norm = cv2.NORM_HAMMING
        return cv2.ORB_create(nfeatures=self.feature_count, scaleFactor=1.2,
                              nlevels=8, edgeThreshold=12, fastThreshold=7)

    def _feature_mask(self, frame: np.ndarray,
                      box: Tuple[int, int, int, int],
                      *, reference: bool) -> Tuple[np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        mask = self._green_mask(crop)
        # Slight dilation admits sharp object silhouette/features right at edges.
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        if reference:
            # position_config_reference.png from the prior calibration contains
            # drawn rings and text. Mask those annotations so they cannot become
            # feature matches. New calibrations also save a clean raw reference.
            for px, py in self.reference_anchors.values():
                cx, cy = int(round(px)) - x1, int(round(py)) - y1
                cv2.circle(mask, (cx, cy), 22, 0, -1)
                cv2.rectangle(mask, (cx + 7, cy - 23), (cx + 125, cy + 19), 0, -1)
        return crop, mask

    def _initialize(self) -> None:
        path = Path(self.reference_image_path) if self.reference_image_path else None
        if path is None or not path.exists():
            self._initialization_error = f"reference image not found: {self.reference_image_path}"
            return
        image = cv2.imread(str(path))
        if image is None:
            self._initialization_error = f"could not read reference image: {path}"
            return
        mask_full = self._green_mask(image)
        box = self._bbox_from_reference(mask_full, self.reference_anchors)
        if box is None:
            self._initialization_error = "could not isolate green body_a in reference image"
            return
        crop, feature_mask = self._feature_mask(image, box, reference=True)
        detector = self._create_feature_detector()
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = detector.detectAndCompute(gray, feature_mask)
        if descriptors is None or len(keypoints) < self.min_inliers:
            self._initialization_error = (
                f"too few reference features on body_a ({0 if keypoints is None else len(keypoints)})"
            )
            return
        self._ref_image = image
        self._ref_box = box
        self._ref_keypoints = keypoints
        self._ref_descriptors = descriptors
        self._detector = detector
        print(
            f"[BodyPoseTracker] reference loaded: {path.name}; "
            f"body ROI={box}; features={len(keypoints)}; "
            f"descriptor={'SIFT' if self._norm == cv2.NORM_L2 else 'ORB'}"
        )

    @staticmethod
    def _body_detection(detections: Sequence[object]) -> Optional[Tuple[int, int, int, int, float]]:
        candidates = []
        for det in detections or []:
            if str(getattr(det, "class_name", "")) != "body_a":
                continue
            conf = float(getattr(det, "confidence", 0.0))
            box = getattr(det, "bounding_box", None)
            if box is None or len(box) < 4:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
            if x2 <= x1 or y2 <= y1:
                continue
            candidates.append((conf, (x1, y1, x2, y2, conf)))
        return max(candidates, key=lambda x: x[0])[1] if candidates else None

    def reset_tracking(self) -> None:
        """Drop the live track. Next update reacquires from the reference pose."""
        self._last_anchors = None
        self._last_valid = False
        self._prev_keypoints = None
        self._prev_descriptors = None
        self._prev_box = None
        self._last_success_time = 0.0

    def update(self, frame: Optional[np.ndarray], detections: Sequence[object]) -> PoseTrackResult:
        if self._initialization_error:
            self._last_valid = False
            return PoseTrackResult(False, {}, self._initialization_error)
        if frame is None:
            self._last_valid = False
            return PoseTrackResult(False, {}, "camera frame unavailable")

        body = self._body_detection(detections)
        if body is None:
            self._last_valid = False
            return PoseTrackResult(False, {}, "YOLO did not locate body_a")
        bx1, by1, bx2, by2, _conf = body
        fh, fw = frame.shape[:2]
        bw, bh = bx2 - bx1, by2 - by1
        pad_x, pad_y = max(5, int(bw * 0.08)), max(5, int(bh * 0.08))
        live_box = (max(0, bx1 - pad_x), max(0, by1 - pad_y),
                    min(fw, bx2 + pad_x), min(fh, by2 + pad_y))

        live_crop, live_mask = self._feature_mask(frame, live_box, reference=False)
        live_gray = cv2.cvtColor(live_crop, cv2.COLOR_BGR2GRAY)
        live_kp, live_desc = self._detector.detectAndCompute(live_gray, live_mask)
        if live_desc is None or live_kp is None or len(live_kp) < self.min_inliers:
            self._last_valid = False
            return PoseTrackResult(False, {}, "not enough visible body_a features",
                                   body_box=(bx1, by1, bx2, by2))

        # Try the previous tracked frame first (best for smooth motion), then
        # the original calibration image as a recovery path. This means that
        # if a few frames fail during a hand occlusion, the next clear frame
        # may recover either incrementally or from the original view. It still
        # does not pretend to solve arbitrary 3-D rotations: both candidates
        # must pass the same match, RANSAC, reprojection and body-box checks.
        sources = []
        if (self._prev_keypoints is not None and self._prev_descriptors is not None
                and self._prev_box is not None and self._last_anchors is not None):
            sources.append((
                "previous frame", self._prev_keypoints, self._prev_descriptors,
                self._prev_box, self._last_anchors,
            ))
        sources.append((
            "calibration reference", self._ref_keypoints, self._ref_descriptors,
            self._ref_box, self.reference_anchors,
        ))

        matcher = cv2.BFMatcher(self._norm, crossCheck=False)
        candidates = []
        failures = []
        for source_name, src_keypoints, src_descriptors, src_box, source_anchors in sources:
            if src_descriptors is None or src_keypoints is None or src_box is None:
                continue
            try:
                knn = matcher.knnMatch(src_descriptors, live_desc, k=2)
            except cv2.error as exc:
                failures.append(f"{source_name}: feature matching failed ({exc})")
                continue

            good = []
            for pair in knn:
                if len(pair) != 2:
                    continue
                m, n = pair
                if m.distance < 0.74 * n.distance:
                    good.append(m)
            if len(good) < self.min_inliers:
                failures.append(f"{source_name}: only {len(good)} reliable matches")
                continue

            sx1, sy1, _, _ = src_box
            lx1, ly1, _, _ = live_box
            src = np.float32([
                [src_keypoints[m.queryIdx].pt[0] + sx1,
                 src_keypoints[m.queryIdx].pt[1] + sy1] for m in good
            ]).reshape(-1, 1, 2)
            dst = np.float32([
                [live_kp[m.trainIdx].pt[0] + lx1,
                 live_kp[m.trainIdx].pt[1] + ly1] for m in good
            ]).reshape(-1, 1, 2)

            H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None or inlier_mask is None or not np.isfinite(H).all():
                failures.append(f"{source_name}: no reliable image transform")
                continue

            inlier_mask = inlier_mask.ravel().astype(bool)
            inliers = int(inlier_mask.sum())
            ratio = inliers / max(len(good), 1)
            projected_matches = cv2.perspectiveTransform(src, H).reshape(-1, 2)
            dst_points = dst.reshape(-1, 2)
            errors = np.linalg.norm(
                projected_matches[inlier_mask] - dst_points[inlier_mask], axis=1
            )
            reproj = float(np.median(errors)) if errors.size else float("inf")

            if (inliers < self.min_inliers or ratio < self.min_inlier_ratio
                    or reproj > self.max_reprojection_error_px):
                failures.append(
                    f"{source_name}: uncertain (inliers={inliers}, "
                    f"ratio={ratio:.2f}, reproj={reproj:.1f}px)"
                )
                continue

            source_points = np.float32(list(source_anchors.values())).reshape(-1, 1, 2)
            out_pts = cv2.perspectiveTransform(source_points, H).reshape(-1, 2)
            anchors = {
                name: (float(pt[0]), float(pt[1]))
                for name, pt in zip(source_anchors.keys(), out_pts)
            }

            # Seat centres must stay on/near the detected body_a box. This
            # rejects accidental matches to the bench or the operator's hand.
            pad = max(12.0, 0.18 * max(bw, bh))
            outside = [
                name for name, (x, y) in anchors.items()
                if not (bx1 - pad <= x <= bx2 + pad and by1 - pad <= y <= by2 + pad)
            ]
            if outside:
                failures.append(f"{source_name}: projected {outside[0]} outside body_a")
                continue

            # ---- REJECT A DEGENERATE TRANSFORM --------------------------
            #
            # The checks above are necessary but nowhere near sufficient,
            # and this is the gap that produced wrong markers on the bench.
            # body_a is a 3-D part and a homography maps a PLANE, so once
            # the part is turned to a different face no homography can
            # describe the change. The plastic is also almost textureless -
            # roughly 50 SIFT features on the whole body - so RANSAC has
            # very little to constrain it.
            #
            # Given both, RANSAC happily returns a transform that collapses
            # the geometry: in the observed failure the two seat anchors,
            # 30 px apart in the reference, landed within a few pixels of
            # each other about 90 px from either real hole. That passed the
            # inlier count, passed the reprojection test (the few matched
            # points really did map onto each other), and passed the box
            # test above, because the collapsed point sat inside body_a.
            #
            # The seats are a rigid pair, so their separation must scale
            # with the part. Comparing the projected separation against the
            # reference separation scaled by how much bigger or smaller the
            # body now appears catches exactly this class of nonsense, and
            # it is the one invariant that a collapse cannot satisfy.
            names = list(source_anchors.keys())
            if len(names) >= 2:
                ax, ay = source_anchors[names[0]]
                bx_, by_ = source_anchors[names[1]]
                ref_sep = math.hypot(ax - bx_, ay - by_)

                sx1, sy1, sx2, sy2 = src_box
                ref_diag = math.hypot(sx2 - sx1, sy2 - sy1)
                live_diag = math.hypot(bw, bh)

                if ref_sep > 1e-6 and ref_diag > 1e-6:
                    expected = ref_sep * (live_diag / ref_diag)
                    px_, py_ = anchors[names[0]]
                    qx, qy = anchors[names[1]]
                    got = math.hypot(px_ - qx, py_ - qy)
                    # Generous bounds: this is meant to catch collapse and
                    # explosion, not to second-guess honest perspective.
                    if got < 0.45 * expected or got > 2.5 * expected:
                        failures.append(
                            f"{source_name}: transform distorts the part "
                            f"(seats {got:.0f} px apart, expected about "
                            f"{expected:.0f})")
                        continue

            # Prefer high-inlier, high-ratio, low-error matches. A mild
            # preference for frame-to-frame tracking avoids drift from a
            # broader template match when both are equally strong.
            score = inliers + 5.0 * ratio - 0.5 * reproj
            if source_name == "previous frame":
                score += 0.25
            candidates.append((score, anchors, inliers, ratio, reproj, source_name))

        if not candidates:
            self._last_valid = False
            reason = "body pose uncertain: " + "; ".join(failures[:2])
            return PoseTrackResult(False, {}, reason,
                                   body_box=(bx1, by1, bx2, by2))

        _score, anchors, inliers, ratio, reproj, source_name = max(
            candidates, key=lambda item: item[0]
        )

        # Save this frame as the tracking keyframe for the next update.
        self._prev_keypoints = live_kp
        self._prev_descriptors = live_desc
        self._prev_box = live_box
        self._last_anchors = anchors
        self._last_valid = True
        self._last_success_time = time.time()
        return PoseTrackResult(
            True, anchors, f"body_a tracked from {source_name}",
            inliers, ratio, reproj, (bx1, by1, bx2, by2)
        )

