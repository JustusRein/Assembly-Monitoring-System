"""
visualization/stage_preview.py
==============================
SUBSYSTEM: Visualization (SS9)

360-degree preview of the stage the operator is currently building.

WHY FRAMES AND NOT A 3D VIEWER:
  The Blender generator renders every stage as a ring of images, one every
  10 degrees, on a transparent background (pass RUN_STAGE_PREVIEWS in
  generate_dataset_YOLO_v19_1). Showing those frames one after another looks
  like the assembly rotating, needs no 3D engine in the runtime, and costs
  almost no CPU - which matters because the camera loop and YOLO already
  own the frame budget.

  It also means the operator sees exactly the geometry the detector was
  trained on, instead of a separately drawn sketch that could drift away
  from the CAD.

The player is deliberately forgiving: if the preview folder is missing or
incomplete, frame() returns None and the display simply omits the panel.
A missing preview must never stop the monitoring system.
"""

import json
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


class StagePreviewPlayer:

    def __init__(
        self,
        preview_dir,
        seconds_per_revolution=8.0,
        panel_size=360,
        background=(38, 38, 38),
        max_cached_sets=3,
        diagram_dir=None,
        mode="auto",
    ):
        """mode:
             "3d"      rotating renders only
             "2d"      flat diagrams only (one image per stage)
             "auto"    a diagram if one exists for that stage, else the
                       rotating render

        The 2D option exists because a flat, labelled diagram is often the
        clearer work instruction, while the rotating render is better for
        judging how a part actually seats. Keeping both and choosing per
        stage avoids having to decide once for the whole product: a stage
        with a hidden feature can use the rotation, a simple one a diagram.

        Diagrams are looked up as <diagram_dir>/<Stage_name>.(png|jpg|svg-
        exported-png) - any single image file named after the stage.
        """
        self.root = Path(preview_dir)
        self.panel_size = int(panel_size)
        self.background = background
        self.seconds_per_revolution = float(seconds_per_revolution)
        self._max_cached = int(max_cached_sets)

        self.mode = str(mode).lower()
        self.diagram_dir = Path(diagram_dir) if diagram_dir else None
        self._diagrams = {}             # name -> path of a flat diagram
        self._diagram_cache = {}

        self._sets = {}                 # name -> list of frame paths
        self._cache = OrderedDict()     # name -> list of BGR images
        self._paused = False
        self._manual_offset = 0
        self._t0 = time.time()
        self.available = False

        self._load_index()
        self._load_diagrams()

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def _load_index(self):
        if not self.root.exists():
            print(f"[StagePreview] No preview folder at {self.root} - "
                  f"panel disabled. Run the Blender generator with "
                  f"RUN_STAGE_PREVIEWS = True to create it.")
            return

        index_file = self.root / "preview_index.json"
        names = []
        if index_file.exists():
            try:
                data = json.loads(index_file.read_text(encoding="utf-8"))
                names = list(data.get("sets", {}).keys())
            except (ValueError, OSError) as exc:
                print(f"[StagePreview] Could not read {index_file}: {exc}")

        if not names:        # no index: fall back to whatever folders exist
            names = [p.name for p in self.root.iterdir() if p.is_dir()]

        for name in names:
            frames = sorted((self.root / name).glob("frame_*.png"))
            if frames:
                self._sets[name] = frames

        self.available = bool(self._sets)
        if self.available:
            total = sum(len(v) for v in self._sets.values())
            print(f"[StagePreview] {len(self._sets)} preview sets, "
                  f"{total} frames, from {self.root}")
        else:
            print(f"[StagePreview] No frames found under {self.root} - "
                  f"panel disabled.")

    def _load_diagrams(self):
        """Optional flat diagrams, one image per stage."""
        if self.diagram_dir is None or not self.diagram_dir.exists():
            if self.mode == "2d":
                print(f"[StagePreview] mode='2d' but no diagram folder at "
                      f"{self.diagram_dir} - falling back to the renders.")
            return
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            for path in self.diagram_dir.glob(ext):
                self._diagrams[path.stem] = path
        if self._diagrams:
            self.available = True       # diagrams alone are enough to show
            print(f"[StagePreview] {len(self._diagrams)} flat diagrams from "
                  f"{self.diagram_dir} (mode='{self.mode}')")

    def _use_diagram(self, name):
        if self.mode == "3d":
            return False
        if self.mode == "2d":
            return name in self._diagrams
        return name in self._diagrams      # auto: diagram wins when present

    def _diagram_frame(self, name):
        if name in self._diagram_cache:
            return self._diagram_cache[name]
        img = cv2.imread(str(self._diagrams[name]), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        bg = np.zeros((self.panel_size, self.panel_size, 3), dtype=np.uint8)
        bg[:, :] = self.background
        h, w = img.shape[:2]
        scale = min(self.panel_size / float(w), self.panel_size / float(h))
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            rgb = img[:, :, :3].astype(np.float32)
            ih, iw = img.shape[:2]
            region = bg[0:ih, 0:iw].astype(np.float32)
            img = (rgb * alpha + region * (1.0 - alpha)).astype(np.uint8)
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = img[:, :, :3]
        ih, iw = img.shape[:2]
        y0 = (self.panel_size - ih) // 2
        x0 = (self.panel_size - iw) // 2
        bg[y0:y0 + ih, x0:x0 + iw] = img
        self._diagram_cache[name] = bg
        return bg

    def has(self, name):
        return name in self._sets or name in self._diagrams

    # ------------------------------------------------------------------
    # Frame loading (lazy, with a small cache)
    # ------------------------------------------------------------------

    def _load_set(self, name):
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]

        images = []
        bg = np.zeros((self.panel_size, self.panel_size, 3), dtype=np.uint8)
        bg[:, :] = self.background

        for path in self._sets[name]:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            img = cv2.resize(img, (self.panel_size, self.panel_size),
                             interpolation=cv2.INTER_AREA)
            if img.ndim == 3 and img.shape[2] == 4:
                # Composite the transparent render over the panel colour.
                alpha = img[:, :, 3:4].astype(np.float32) / 255.0
                rgb = img[:, :, :3].astype(np.float32)
                out = rgb * alpha + bg.astype(np.float32) * (1.0 - alpha)
                images.append(out.astype(np.uint8))
            elif img.ndim == 2:
                images.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            else:
                images.append(img[:, :, :3])

        self._cache[name] = images
        while len(self._cache) > self._max_cached:
            self._cache.popitem(last=False)
        return images

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def toggle_pause(self):
        self._paused = not self._paused
        return self._paused

    @property
    def paused(self):
        return self._paused

    def step(self, delta):
        """Rotate by hand, e.g. from the , and . keys."""
        self._manual_offset += int(delta)

    def frame(self, name):
        """Current frame for this stage, or None if there is no preview."""
        if not self.available or name is None:
            return None
        if self._use_diagram(name):
            frame = self._diagram_frame(name)
            if frame is not None:
                return frame            # a diagram is static; nothing to turn
        if name not in self._sets:
            return None
        images = self._load_set(name)
        if not images:
            return None
        if self._paused:
            idx = self._manual_offset
        else:
            per_frame = max(self.seconds_per_revolution / len(images), 1e-3)
            idx = int((time.time() - self._t0) / per_frame) + self._manual_offset
        return images[idx % len(images)]
