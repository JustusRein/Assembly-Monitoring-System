"""
processing/object_detector.py

Object Detection Module (Processing Subsystem)

Uses YOLO to identify assembly objects.

V2.1 confidence upgrade:
    - Per-class confidence statistics for every frame
      (best, mean, detection count per class).
    - Rolling average confidence per class over the last N frames,
      so the display can show a STABLE confidence value instead of
      a number that flickers every frame.
    - Configuration comes from config/settings.py instead of being
      hard-coded in main.py.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from ultralytics import YOLO
from config.settings import EXPECTED_YOLO_CLASSES

# Per-class confidence thresholds (V2.2).
# One global threshold cannot fit every part: real parts score 70-95%, but
# the thin threaded_pin only scores 25-64% on the live camera. With a global
# 0.45 it was filtered out and Stage_06 (knob + threaded_pin) stalled; with a
# global 0.25 the other parts produced false detections at 25-35%.
# Parts listed here use their own threshold; all others use YOLO_CONFIDENCE.
# Can be overridden by defining YOLO_CLASS_CONFIDENCE in config/settings.py.
try:
    from config.settings import YOLO_CLASS_CONFIDENCE
except ImportError:
    YOLO_CLASS_CONFIDENCE = {"threaded_pin": 0.25}


@dataclass
class DetectedObject:
    class_name: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]


@dataclass
class ClassConfidenceStats:
    """Confidence statistics for one class in one frame."""
    class_name: str
    count: int                 # detections of this class in the frame
    best_confidence: float     # highest confidence this frame
    mean_confidence: float     # mean confidence this frame
    rolling_confidence: float  # mean of best confidence over last N frames


@dataclass
class ObjectDetectionResult:
    objects: List[DetectedObject] = field(default_factory=list)

    # class_name -> ClassConfidenceStats (filled by ObjectDetector)
    class_stats: Dict[str, ClassConfidenceStats] = field(
        default_factory=dict
    )

    def best_per_class(self) -> Dict[str, DetectedObject]:
        """Return the highest-confidence detection for each class."""
        best = {}
        for obj in self.objects:
            current = best.get(obj.class_name)
            if current is None or obj.confidence > current.confidence:
                best[obj.class_name] = obj
        return best


class ObjectDetector:

    def __init__(
        self,
        model_path,
        confidence=0.25,
        rolling_window=30,
        imgsz=640,
        device=None,
    ):
        resolved_path = Path(model_path).resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"[ObjectDetector] YOLO weights not found at: {resolved_path}\n"
                f"  Check YOLO_MODEL_PATH in config/settings.py."
            )
        size_mb = resolved_path.stat().st_size / (1024 * 1024)
        print(f"[ObjectDetector] Loading weights: {resolved_path} ({size_mb:.1f} MB)")

        self.model = YOLO(str(resolved_path))

        loaded_classes = list(self.model.names.values())
        print(f"[ObjectDetector] Model classes: {loaded_classes}")

        expected_classes = list(EXPECTED_YOLO_CLASSES)
        if (set(loaded_classes) != set(expected_classes)
                or len(loaded_classes) != len(expected_classes)):
            raise ValueError(
                "[ObjectDetector] WRONG YOLO MODEL FOR THIS ASSEMBLY PROJECT.\n"
                f"  Expected classes: {expected_classes}\n"
                f"  Loaded classes:   {loaded_classes}\n"
                "  The checkpoint is not the seven-class assembly model."
            )

        if loaded_classes != expected_classes:
            print(
                "[ObjectDetector] WARNING: the same 7 class names are present, "
                "but class-ID order differs from configuration."
            )

        self.confidence = confidence
        self.class_confidence = {
            str(k): float(v) for k, v in dict(YOLO_CLASS_CONFIDENCE).items()
        }
        unknown = set(self.class_confidence) - set(loaded_classes)
        if unknown:
            print(f"[ObjectDetector] WARNING: YOLO_CLASS_CONFIDENCE has unknown "
                  f"classes {sorted(unknown)} - ignored.")
        # YOLO must return candidates down to the LOWEST threshold in use;
        # each class is then filtered with its own threshold in process().
        self._predict_conf = min([confidence] + list(self.class_confidence.values()))
        print(f"[ObjectDetector] Confidence: {confidence:.2f} for all classes"
              + "".join(f", {k} {v:.2f}" for k, v in self.class_confidence.items()))
        self.imgsz = imgsz
        self.device = device

        # class_name -> deque of best-confidence values (last N frames
        # in which the class was detected)
        self._rolling: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=rolling_window)
        )

    def process(self, frame) -> ObjectDetectionResult:

        results = self.model.predict(
            frame,
            conf=self._predict_conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detected_objects = []

        for result in results:

            names = result.names

            for box in result.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Per-class threshold
                if conf < self.class_confidence.get(names[cls], self.confidence):
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detected_objects.append(
                    DetectedObject(
                        class_name=names[cls],
                        confidence=conf,
                        bounding_box=(x1, y1, x2, y2),
                    )
                )

        # ----------------------------------------------------
        # PER-CLASS CONFIDENCE STATISTICS
        # ----------------------------------------------------

        per_class_confidences: Dict[str, List[float]] = defaultdict(list)

        for obj in detected_objects:
            per_class_confidences[obj.class_name].append(obj.confidence)

        class_stats: Dict[str, ClassConfidenceStats] = {}

        for class_name, confidences in per_class_confidences.items():

            best = max(confidences)
            mean = sum(confidences) / len(confidences)

            # Update rolling history with this frame's best confidence
            self._rolling[class_name].append(best)

            history = self._rolling[class_name]
            rolling = sum(history) / len(history)

            class_stats[class_name] = ClassConfidenceStats(
                class_name=class_name,
                count=len(confidences),
                best_confidence=best,
                mean_confidence=mean,
                rolling_confidence=rolling,
            )

        return ObjectDetectionResult(
            objects=detected_objects,
            class_stats=class_stats,
        )
