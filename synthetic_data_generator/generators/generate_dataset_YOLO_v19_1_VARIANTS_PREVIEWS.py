"""
===============================================================================
YOLO SYNTHETIC DATASET GENERATOR - V19.1 (full-rotation + variants + 360 stage previews)
Single parts + manual Blender subassembly collections (Stage_01 ... Stage_07)
===============================================================================

WHAT CHANGED IN V18
-------------------
1. FULL 3-D ROTATION (the main fix)
   V17 only spun each assembly around the vertical axis and tilted it by at
   most 10 degrees (COLLECTION_ASSEMBLY_TILT_MAX_DEG = 10), while the camera
   was forced to stay above the table. The underside of every stage - where
   the nut sits on Stage_05 - therefore faced the table in practically every
   image, and YOLO never learned the nut on the assembly.
   V18 gives every assembly and every single part a uniformly random 3-D
   orientation, so any side, including the bottom, can face the camera.

2. KEY PARTS MUST ACTUALLY BE VISIBLE
   For each stage the part that matters (bushing, body_b, block_pin, nut,
   threaded_pin, knob) must be clearly visible in most images
   (ADDON_REQUIRE_PROB), with a realistic per-part target
   (ADDON_KEY_PART_MIN_FRACTION - a bushing pressed into a bore can never be
   fully exposed). If a random pose hides it, the pose is thrown away and
   re-rolled BEFORE rendering, so no render time is wasted.

3. BETTER LABELS
   V17 built each box from 48 face centres only, so a partly visible part got
   a sliver box from two points. V18 also samples mesh vertices (the real
   silhouette edges), measures how much of each part is actually visible, and
   drops glimpses with fewer than MIN_VISIBLE_POINTS_LABEL visible points
   instead of mislabelling them. A pin head whose shaft is buried inside the
   bodies is still labelled.

4. PARTS REST ON THE TABLE (OR ARE HELD)
   Placement now uses the real mesh vertices instead of bounding-box corners.
   With rotation, a box corner lies below the geometry, which made parts
   float. A share of images is lifted on purpose, as if held by the operator.

5. MORE VARIED ENVIRONMENT
   Wider light energy and colour temperature, soft/hard shadows, an optional
   fill light, random world brightness, camera exposure jitter, near-overhead
   camera views, true-to-camera small-scale views, and hand/tool-like
   occluders.

6. YOUR EXISTING DATA IS PRESERVED
   New images go into separate folders with new names:
       synthetic_dataset/addon_single_fullrot/        nut_rot_0001.png ...
       synthetic_dataset/subassembly_addon_fullrot/   stage_05_rot_0001.png ...
   The original single-part and subassembly passes are switched OFF by
   default, and CLEAN_SUBASSEMBLY_OUTPUT_BEFORE_PRODUCTION is now False.
   A combined data_combined_v18.yaml listing old + new folders is written at
   the end for training.

7. V18.1: helper objects (occluder, fill light) always live in the scene
   root collection. V18 linked the occluder to the collection active in the
   Outliner, which put it inside Stage_07 and stopped the run.

8. threaded_pin no longer renders twice (V17 left the source object visible
   next to its temporary render copy).

-------------------------------------------------------------------------------
HOW TO RUN
    1. Open MasterScene_SUBASSEMBLY_WORKING.blend.
    2. Scripting tab -> open this file -> Run.
    3. FIRST RUN: ADDON_SMOKE_TEST = True (a few images per stage/class).
       Open the images, check that the nut / threaded_pin are clearly visible.
       In the console, "full=True" means the part reached its visibility
       target. If a stage shows mostly full=False, lower that part's value in
       ADDON_KEY_PART_MIN_FRACTION.
    4. Then set ADDON_SMOKE_TEST = False for the full run.
    5. Read the ADD-ON AUDIT table printed at the end: it reports, per stage,
       in how many images each key part is actually labelled.
===============================================================================
"""

import bpy
import bmesh
import math
import random
import glob
import os
import json
from datetime import datetime, timezone
from pathlib import Path
import mathutils
import base64
import tempfile
import numpy as np
from mathutils import Vector


# ============================================================
# SETTINGS  -  you may change these
# ============================================================

OUTPUT_DIR = Path(r"D:\Thesis\Blender\synthetic_dataset")
BACKGROUND_DIR = Path(r"D:\Thesis\Blender\Backgrounds")

IMAGES_PER_PART = 200           # 200 images for EACH of the 7 classes
IMAGES_PER_STAGE = 200          # 200 images for EACH of the 7 assembly stages

# V16_PRODUCTION_VIEWPOINT_POLICY:
# Each stage is rendered from multiple assembly poses and camera viewpoints.
# 24 azimuth sectors x 5 elevation bands provide 120 structured viewpoints;
# remaining images use the same coverage with new random pose/background/light.
V16_PRODUCTION_VIEWPOINT_POLICY = True
RES_X = 640
RES_Y = 640

# Class list in class-ID order.
PART_CLASSES = {
    "body_a":       ["1_body_a"],
    "body_b":       ["2_body_b"],
    "block_pin":    ["3_block_pin"],
    "bushing":      ["4_bushing", "4_bushing.001"],
    "nut":          ["5_nut"],
    "knob":         ["6_knob"],
    "threaded_pin": ["7_threaded_pin"],
}

CAMERA_NAME = "Camera"
LIGHT_NAME = "Light"
WORKBENCH_NAME = "Workbench"

OCCLUSION_PROBABILITY = 0.0

CAM_DIST_FACTOR_MIN = 2.6
CAM_DIST_FACTOR_MAX = 4.2
CAM_DIST_ABS_MIN = 0.03
CAM_DIST_ABS_MAX = 0.60
CAM_ELEV_MIN_DEG = 12.0
CAM_ELEV_MAX_DEG = 88.0
FRAME_FIT_MARGIN = 0.90
LIGHT_ENERGY_MIN = 80.0
LIGHT_ENERGY_MAX = 250.0

# Train / val / test split fractions (same idea as the RWTH code: 60/20/20)
SPLIT_TRAIN = 0.60
SPLIT_VAL = 0.20
# test = remaining 0.20

RANDOM_SEED = 42

# ============================================================
# ============================================================
# SUBASSEMBLY DATASET SETTINGS
# ============================================================
# V18: the ORIGINAL passes are OFF so the existing 2,800 images are not
# regenerated or overwritten. Turn on only to rebuild the base dataset.
RUN_SINGLE_PART_DATASET = False
RUN_SUBASSEMBLY_DATASET = False

# SAFE VALIDATION RUN:
# True  -> 3 images per single-part class + 3 images per Stage
# False -> full run using IMAGES_PER_PART / IMAGES_PER_STAGE
SMOKE_TEST = False
SMOKE_IMAGES_PER_PART = 3
SMOKE_IMAGES_PER_STAGE = 3

# FINAL subassembly source: manually assembled Blender collections.
# The old Stage_XX_Assembly.stl files are NOT used by this generator.
STAGE_COLLECTIONS = [f"Stage_{i:02d}" for i in range(1, 8)]

STAGE_EXPECTED_SEQUENCE = {
    "Stage_01": {"body_a": 1, "bushing": 1},
    "Stage_02": {"body_a": 1, "bushing": 2},
    "Stage_03": {"body_a": 1, "body_b": 1, "bushing": 2},
    "Stage_04": {"body_a": 1, "body_b": 1, "bushing": 2, "block_pin": 1},
    "Stage_05": {"body_a": 1, "body_b": 1, "bushing": 2, "block_pin": 1, "nut": 1},
    "Stage_06": {"knob": 1, "threaded_pin": 1},
    "Stage_07": {
        "body_a": 1,
        "body_b": 1,
        "bushing": 2,
        "block_pin": 1,
        "nut": 1,
        "knob": 1,
        "threaded_pin": 1,
    },
}

COLLECTION_ASSEMBLY_YAW_MIN_DEG = 0.0
COLLECTION_ASSEMBLY_YAW_MAX_DEG = 360.0
COLLECTION_ASSEMBLY_TILT_MAX_DEG = 10.0
COLLECTION_XY_JITTER = 0.040



# V12 production controls.
# When False, the existing dataset is left untouched.
# When True and SMOKE_TEST=False, the subassembly output directory is cleaned
# before generating the production dataset so no stale images/labels remain.
CLEAN_SUBASSEMBLY_OUTPUT_BEFORE_PRODUCTION = False   # V18: never delete by default

# Camera coverage: 24 azimuth sectors x 5 elevation bands = 120 unique
# angle combinations per stage.
STAGE_AZIMUTH_STEP_DEG = 15.0
STAGE_AZIMUTH_JITTER_DEG = 4.0
STAGE_ELEVATION_BANDS_DEG = (18.0, 30.0, 45.0, 60.0, 75.0)
STAGE_ELEVATION_JITTER_DEG = 4.0
CAMERA_SHIFT_MAX = 0.10       # random image-plane translation of the assembly
CAMERA_RELATIVE_VIEW_JITTER_DEG = 5.0

# Production view distribution, interleaved across all indices so train/val/test
# each receive a mixture of close/medium/far views.
# 45% CLOSE, 40% MEDIUM, 15% FAR.
STAGE_VIEW_CYCLE = (
    "CLOSE", "MEDIUM", "CLOSE", "MEDIUM", "CLOSE",
    "MEDIUM", "CLOSE", "MEDIUM", "CLOSE", "FAR",
    "CLOSE", "MEDIUM", "CLOSE", "MEDIUM", "CLOSE",
    "MEDIUM", "CLOSE", "MEDIUM", "CLOSE", "FAR",
)

# Legacy STL assets retained for reference only; NOT USED by collection mode.
SUBASSEMBLY_ASSETS_DIR = Path(r"D:\Thesis\Blender\SubassemblyAssets")
SUBASSEMBLY_OUTPUT_DIR = OUTPUT_DIR / "subassembly"

# YOLO annotation policy:
# A physically present component that is fully occluded from the camera does
# not receive a normal YOLO box.
VISIBILITY_AWARE_LABELS = True
VISIBLE_FACE_SAMPLES = 48
MIN_VISIBLE_FACE_HITS = 1
MIN_BOX_PIXELS = {
    "CLOSE": 20,
    "MEDIUM": 12,
    "FAR": 8,
}

# Legacy STL-specific settings are intentionally omitted in final collection mode.

# ------------------------------------------------------------------
# FINAL GENERATION CONTROLS
# ------------------------------------------------------------------
MAX_RENDER_ATTEMPTS_PER_IMAGE = 30
VERIFY_RENDER_OUTPUT = True

TARGET_FAR_MIN = 0.12
TARGET_FAR_MAX = 0.24
TARGET_MED_MIN = 0.35
TARGET_MED_MAX = 0.55
TARGET_CLOSE_MIN = 0.62
TARGET_CLOSE_MAX = 0.82

CAM_DIST_SEARCH_MIN = 0.015
CAM_DIST_SEARCH_MAX = 2.50

CAM_ELEV_LOW_MIN = 15.0
CAM_ELEV_LOW_MAX = 45.0
CAM_ELEV_HIGH_MIN = 45.0
CAM_ELEV_HIGH_MAX = 82.0


# ============================================================
# V18 ADD-ON SETTINGS  -  full-rotation images, existing data preserved
# ============================================================

RUN_ADDON_SINGLE_PART = False    # V19: already generated, not re-rendered
RUN_ADDON_SUBASSEMBLY = False    # V19: already generated, not re-rendered

# First run with True: a few images per class/stage to check visually.
ADDON_SMOKE_TEST = False   # smoke test passed - full production run
ADDON_SMOKE_IMAGES_PER_CLASS = 4
ADDON_SMOKE_IMAGES_PER_STAGE = 5

ADDON_SEED = 2026

# If an image fails after ADDON_MAX_ATTEMPTS, it is retried again with fresh
# random poses at the end of its class/stage, up to this many extra rounds,
# so the requested counts are actually reached.
ADDON_RETRY_ROUNDS = 3

ADDON_SINGLE_OUTPUT_DIR = OUTPUT_DIR / "addon_single_fullrot"
ADDON_SUBASSEMBLY_OUTPUT_DIR = OUTPUT_DIR / "subassembly_addon_fullrot"
COMBINED_DATA_YAML_PATH = OUTPUT_DIR / "data_combined_v18.yaml"

# Extra single-part images, weighted toward the weak classes.
ADDON_IMAGES_PER_CLASS = {
    "body_a":       100,
    "body_b":       100,
    "block_pin":    150,
    "bushing":      250,
    "nut":          300,
    "knob":         150,
    "threaded_pin": 300,
}

# Extra subassembly images, weighted toward stages with hidden parts.
ADDON_IMAGES_PER_STAGE = {
    "Stage_01": 100,
    "Stage_02": 120,
    "Stage_03": 150,
    "Stage_04": 200,
    "Stage_05": 300,
    "Stage_06": 200,
    "Stage_07": 300,
}

# The part(s) that define each stage and must be clearly visible in
# ADDON_REQUIRE_PROB of that stage's images: {class: how many instances}.
ADDON_REQUIRED_VISIBLE = {
    "Stage_01": {"bushing": 1},
    "Stage_02": {"bushing": 2},
    "Stage_03": {"body_b": 1},
    "Stage_04": {"block_pin": 1},
    "Stage_05": {"nut": 1},
    "Stage_06": {"threaded_pin": 1, "knob": 1},
    "Stage_07": {"nut": 1, "threaded_pin": 1},
}
# The nut and the threaded_pin sit on different sides of the finished
# assembly, so requiring both clearly visible in ONE image often fails
# (V18.1 smoke run: 2/5 Stage_07 images failed). Stage_07 alternates
# between the two instead.
ADDON_REQUIRED_ALTERNATIVES = {
    "Stage_07": [{"nut": 1}, {"threaded_pin": 1}],
}
ADDON_REQUIRE_PROB = 0.75

# "Clearly visible" = at least this share of the part's sampled surface can
# be seen (not hidden behind other parts, not outside the frame). Parts that
# sit INSIDE the assembly by design can never be fully exposed - a bushing is
# pressed into a bore, the block_pin runs through both bodies - so each part
# gets a realistic target.
ADDON_KEY_PART_MIN_FRACTION = {
    "body_b":       0.60,
    "knob":         0.60,
    "nut":          0.50,
    "threaded_pin": 0.45,
    "bushing":      0.30,
    "block_pin":    0.25,
}
ADDON_KEY_PART_DEFAULT_FRACTION = 0.50
# If a pose reaching that target is not found within this many attempts, the
# requirement drops to "the part is clearly labelled" so generation never
# stalls. The audit reports how many images met the full target.
ADDON_RELAX_AFTER_ATTEMPTS = 45
ADDON_MAX_ATTEMPTS = 90

# Label policy for ALL images: a part is boxed when at least this many of
# its sampled surface points are visible. This drops two-point slivers (V17)
# but still labels a clearly visible pin head whose shaft is buried inside
# the bodies.
MIN_VISIBLE_POINTS_LABEL = 8
VISIBILITY_VERTEX_SAMPLES = 160

# The operator holds the assembly while working, so a share of images lifts
# it off the bench (metres).
ADDON_LIFT_PROB = 0.55
ADDON_MAX_LIFT = 0.06

# Hand / tool-like occluders (never used on images that must show a key part).
ADDON_SINGLE_OCCLUSION_PROB = 0.30
ADDON_STAGE_OCCLUSION_PROB = 0.20

# Camera. Near-overhead bands included because the real camera looks down.
ADDON_ELEVATION_BANDS_DEG = (28.0, 42.0, 56.0, 70.0, 83.0)
ADDON_ELEVATION_JITTER_DEG = 4.0
ADDON_CAMERA_SHIFT_MAX = 0.08

# View mix. SMALL = the size the real camera sees the object at
# (object size / DEPLOY_FOV_WIDTH_M), always from a high angle.
DEPLOY_FOV_WIDTH_M = 0.90
ADDON_STAGE_VIEW_CYCLE = (
    "MEDIUM", "CLOSE", "FAR", "SMALL", "MEDIUM",
    "FAR", "CLOSE", "MEDIUM", "SMALL", "FAR",
)
ADDON_SINGLE_VIEW_WEIGHTS = {"SMALL": 0.25, "FAR": 0.25,
                             "MEDIUM": 0.30, "CLOSE": 0.20}
MIN_BOX_PIXELS["SMALL"] = 6

# Environment variety.
ADDON_LIGHT_ENERGY_MIN = 50.0
ADDON_LIGHT_ENERGY_MAX = 400.0
ADDON_LIGHT_KELVIN_MIN = 2800.0
ADDON_LIGHT_KELVIN_MAX = 7000.0
ADDON_FILL_LIGHT_PROB = 0.50
ADDON_WORLD_STRENGTH_MIN = 0.25
ADDON_WORLD_STRENGTH_MAX = 1.10
ADDON_EXPOSURE_JITTER = 0.5
FILL_LIGHT_NAME = "TEMP_FILL_LIGHT"


# ============================================================
# INTERNAL
# ============================================================

random.seed(RANDOM_SEED)

IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"
NDJSON_PATH = OUTPUT_DIR / "annotations.ndjson"
DATA_YAML_PATH = OUTPUT_DIR / "data.yaml"

CLASS_NAMES = list(PART_CLASSES.keys())
CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

ALL_TARGET_OBJECTS = []
for _names in PART_CLASSES.values():
    ALL_TARGET_OBJECTS.extend(_names)

BACKGROUND_FILES = []
GROUND_NAME = "PHOTO_GROUND"
_GROUND_TEX_NODE = None
OCCLUDER_NAME = "TEMP_OCCLUDER"




def resolve_single_part_source(class_name):
    """Resolve a single-part source object robustly.

    Prefer the original source names. If the old bushing names were changed or
    removed while creating the manual Stage collections, use the manually
    prepared Stage_01_bushing_01 as the authoritative bushing source.
    """
    preferred = PART_CLASSES[class_name]

    for name in preferred:
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.type == "MESH":
            return obj

    if class_name == "bushing":
        fallback_names = [
            "Stage_01_bushing_01",
            "Stage_02_bushing_01",
            "Stage_03_bushing_01",
        ]
        for name in fallback_names:
            obj = bpy.data.objects.get(name)
            if obj is not None and obj.type == "MESH":
                log(
                    f"INFO: using '{name}' as fallback single-part "
                    f"source for class 'bushing'."
                )
                return obj

        # Last-resort search by the established naming convention.
        for obj in bpy.data.objects:
            if (
                obj.type == "MESH"
                and obj.name.lower().startswith("stage_")
                and "bushing_01" in obj.name.lower()
            ):
                log(
                    f"INFO: using '{obj.name}' as fallback single-part "
                    f"source for class 'bushing'."
                )
                return obj

    return None


def resolve_all_single_part_sources():
    """Return {class_name: source_object_name} for the single-part dataset."""
    resolved = {}
    missing = []

    for class_name in CLASS_NAMES:
        obj = resolve_single_part_source(class_name)
        if obj is None:
            missing.append(class_name)
        else:
            resolved[class_name] = obj.name

    if missing:
        raise RuntimeError(
            "Missing single-part source objects/classes: "
            + ", ".join(missing)
        )

    return resolved


def validate_required_classes():
    """Validate and resolve all seven single-part dataset sources."""
    log("=" * 70)
    log("VALIDATING REQUIRED 7 YOLO SINGLE-PART CLASSES")
    log("=" * 70)

    resolved = resolve_all_single_part_sources()

    for class_name in CLASS_NAMES:
        source_name = resolved[class_name]
        log(
            f"OK: class '{class_name}' -> {source_name}"
        )

    log("ALL 7 SINGLE-PART CLASSES FOUND/RESOLVED.")
    log("=" * 70)
    return resolved



def log(msg):
    print(f"[dataset] {msg}")


def get_object(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        log(f"WARNING: object '{name}' not found.")
    return obj


def choose_split(index):
    if index <= int(IMAGES_PER_PART * SPLIT_TRAIN):
        return "train"
    if index <= int(IMAGES_PER_PART * (SPLIT_TRAIN + SPLIT_VAL)):
        return "val"
    return "test"


# ============================================================
# NDJSON  (the RWTH training format)
# ============================================================

def initialize_ndjson():
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = {
        "type": "dataset",
        "task": "detect",
        "class_names": {str(i): name for i, name in enumerate(CLASS_NAMES)},
        "Created_at": created_at,
    }
    NDJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NDJSON_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
    log(f"Initialised {NDJSON_PATH.name} with {len(CLASS_NAMES)} classes.")


def append_ndjson_record(image_path, width, height, boxes, split):
    record = {
        "file": str(image_path),
        "type": "image",
        "width": width,
        "height": height,
        "annotations": {"boxes": boxes} if boxes else {},
        "split": split,
    }
    with open(NDJSON_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def write_data_yaml():
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    content = (
        f"path: {OUTPUT_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_block}\n"
    )
    DATA_YAML_PATH.write_text(content, encoding="utf-8")
    log(f"Wrote {DATA_YAML_PATH.name} (backup standard-YOLO config).")


# ============================================================
# SETUP
# ============================================================

def setup_render():
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        try:
            scene.render.engine = "BLENDER_EEVEE"
        except Exception:
            log("Could not set EEVEE; using active engine.")
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    setup_world_background()
    cam = get_object(CAMERA_NAME)
    if cam is not None:
        scene.camera = cam


def ensure_output_dirs():
    for split in ("train", "val", "test"):
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "classes.txt").write_text(
        "\n".join(CLASS_NAMES) + "\n", encoding="utf-8"
    )
    log(f"Wrote class list ({len(CLASS_NAMES)} classes).")


def load_background_files():
    global BACKGROUND_FILES
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(str(BACKGROUND_DIR / "**" / ext), recursive=True))
        files.extend(glob.glob(str(BACKGROUND_DIR / ext)))
    files = sorted(set(files))
    BACKGROUND_FILES = files
    if not BACKGROUND_FILES:
        log("WARNING: no background images found. Backgrounds will be plain.")
    else:
        log(f"Found {len(BACKGROUND_FILES)} background images.")


# ============================================================
# WORLD BACKGROUND
# ============================================================

def setup_world_background():
    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.35, 0.35, 0.38, 1.0)
        bg.inputs["Strength"].default_value = 0.6
    cleanup_leftover_objects()
    setup_ground_plane()


def cleanup_leftover_objects():
    leftovers = ["TEMP_BACKDROP", "TEMP_OCCLUDER", "PHOTO_GROUND",
                 FILL_LIGHT_NAME]
    removed = 0
    for name in leftovers:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            if name in ALL_TARGET_OBJECTS:
                continue
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    if removed:
        log(f"Cleaned up {removed} leftover helper object(s).")


def setup_ground_plane():
    global _GROUND_TEX_NODE
    plane = bpy.data.objects.get(WORKBENCH_NAME)
    if plane is None:
        log(f"WARNING: no object named '{WORKBENCH_NAME}' found.")
        return None
    plane.hide_render = False
    plane.hide_viewport = False
    mesh = plane.data
    if hasattr(mesh, "uv_layers") and not mesh.uv_layers:
        log("Ground plane has no UV map - creating one.")
        prev_active = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = plane
        plane.select_set(True)
        try:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.smart_project()
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception as e:
            log(f"  UV creation skipped: {e}")
        plane.select_set(False)
        bpy.context.view_layer.objects.active = prev_active

    mat_name = "BG_PLANE_MATERIAL"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (300, 0)
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.location = (0, 0)
    coord = nt.nodes.new("ShaderNodeTexCoord")
    coord.location = (-250, 0)
    nt.links.new(coord.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs["Roughness"].default_value = 0.9
    bsdf.inputs["Metallic"].default_value = 0.0
    plane.data.materials.clear()
    plane.data.materials.append(mat)
    _GROUND_TEX_NODE = tex
    log("Ground plane set up to display background photos.")
    return plane


def set_random_background():
    if not BACKGROUND_FILES or _GROUND_TEX_NODE is None:
        return
    path = random.choice(BACKGROUND_FILES)
    try:
        img = bpy.data.images.load(path, check_existing=True)
    except Exception as e:
        log(f"  could not load background {path}: {e}")
        return
    try:
        img.colorspace_settings.name = "sRGB"
    except Exception:
        pass
    _GROUND_TEX_NODE.image = img


# ============================================================
# VISIBILITY
# ============================================================

def hide_all_targets():
    for name in ALL_TARGET_OBJECTS:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_render = True
            obj.hide_viewport = True


def show_only(object_name):
    hide_all_targets()
    obj = bpy.data.objects.get(object_name)
    if obj is not None:
        obj.hide_render = False
        obj.hide_viewport = False
    return obj


def unhide_all_parts():
    for name in ALL_TARGET_OBJECTS:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_render = False
            obj.hide_viewport = False


# ============================================================
# ROBUST WORLD-SPACE GEOMETRY
# ============================================================

def eval_matrix_world(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return obj.evaluated_get(depsgraph).matrix_world


def unparent_keep_world_transform(obj):
    if obj.parent is not None:
        parent_name = obj.parent.name
        world_matrix = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world_matrix
        bpy.context.view_layer.update()
        log(f"  {obj.name}: removed parent '{parent_name}'.")


def validate_part_placement(obj):
    plate_z = get_plate_top_z()
    lowest_z, highest_z = _bbox_z_limits(obj)

    if highest_z < plate_z:
        log(
            f"  ERROR: {obj.name} is entirely below the workbench "
            f"(lowest_z={lowest_z:.6f}, highest_z={highest_z:.6f}, "
            f"plate_z={plate_z:.6f}, location={tuple(round(v, 6) for v in obj.location)})"
        )
        return False

    if lowest_z < plate_z - 0.005:
        log(
            f"  ERROR: {obj.name} penetrates the workbench "
            f"(lowest_z={lowest_z:.6f}, plate_z={plate_z:.6f}, "
            f"location={tuple(round(v, 6) for v in obj.location)})"
        )
        return False

    return True


def validate_camera_position():
    cam = get_object(CAMERA_NAME)
    if cam is None:
        return False
    plate_z = get_plate_top_z()
    if cam.location.z <= plate_z:
        log(f"  ERROR: camera below workbench "
            f"(camera_z={cam.location.z:.5f}, plate_z={plate_z:.5f}).")
        return False
    return True


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def get_world_bbox_center(obj):
    corners = _world_bbox_corners(obj)
    return sum(corners, Vector((0, 0, 0))) / 8.0


def get_object_size(obj):
    corners = _world_bbox_corners(obj)
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))


def get_plate_top_z():
    bench = bpy.data.objects.get(WORKBENCH_NAME)
    if bench is None:
        return 0.0
    corners = [bench.matrix_world @ Vector(c) for c in bench.bound_box]
    return max(c.z for c in corners)


# ============================================================
# POSE
# ============================================================

def _world_bbox_corners(obj):
    mw = eval_matrix_world(obj)
    return [mw @ Vector(c) for c in obj.bound_box]


def _bbox_z_limits(obj):
    corners = _world_bbox_corners(obj)
    return min(c.z for c in corners), max(c.z for c in corners)


def _force_world_translation(obj, x=None, y=None, z=None):
    loc = obj.location.copy()
    if x is not None:
        loc.x = x
    if y is not None:
        loc.y = y
    if z is not None:
        loc.z = z

    obj.location = loc
    bpy.context.view_layer.update()
    _ = eval_matrix_world(obj)


def _bake_world_z_correction_into_mesh(obj, error):
    if obj.data is None or not hasattr(obj.data, "vertices"):
        return False

    try:
        inv_rot = obj.matrix_world.to_3x3().inverted()
        local_delta = inv_rot @ Vector((0.0, 0.0, -error))
        for vert in obj.data.vertices:
            vert.co += local_delta
        obj.data.update()
        bpy.context.view_layer.update()
        return True
    except Exception as exc:
        log(f"  mesh-space correction failed for {obj.name}: {exc}")
        return False


def place_part_on_plate(obj):
    unparent_keep_world_transform(obj)

    yaw = random.uniform(0.0, 2.0 * math.pi)
    r = random.random()
    if r < 0.70:
        tilt_deg = random.uniform(0.0, 15.0)
    elif r < 0.95:
        tilt_deg = random.uniform(15.0, 35.0)
    else:
        tilt_deg = random.uniform(35.0, 55.0)

    obj.rotation_euler = (
        math.radians(random.uniform(-tilt_deg, tilt_deg)),
        math.radians(random.uniform(-tilt_deg, tilt_deg)),
        yaw,
    )
    bpy.context.view_layer.update()
    _ = eval_matrix_world(obj)

    plate_z = get_plate_top_z()
    lowest_z, highest_z = _bbox_z_limits(obj)

    size = max(get_object_size(obj), 0.05)
    jitter = size * 0.25
    target_x = obj.location.x + random.uniform(-jitter, jitter)
    target_y = obj.location.y + random.uniform(-jitter, jitter)

    target_z = obj.location.z + (plate_z - lowest_z)
    _force_world_translation(obj, target_x, target_y, target_z)

    for _ in range(5):
        lowest_z, highest_z = _bbox_z_limits(obj)
        error = lowest_z - plate_z

        if abs(error) <= 0.001:
            return

        log(
            f"  placement correction for {obj.name}: "
            f"lowest_z={lowest_z:.6f}, plate_z={plate_z:.6f}, "
            f"dz={-error:.6f}"
        )

        obj.location.z -= error
        bpy.context.view_layer.update()
        _ = eval_matrix_world(obj)

    lowest_z, _ = _bbox_z_limits(obj)
    error = lowest_z - plate_z
    if abs(error) > 0.001:
        mw = obj.matrix_world.copy()
        mw.translation.z -= error
        obj.matrix_world = mw
        bpy.context.view_layer.update()
        _ = eval_matrix_world(obj)

    lowest_z, _ = _bbox_z_limits(obj)
    error = lowest_z - plate_z
    if abs(error) > 0.001 and obj.name.startswith("__DATASET_TMP__"):
        if _bake_world_z_correction_into_mesh(obj, error):
            log(
                f"  {obj.name}: applied final mesh-space Z correction "
                f"{-error:.6f}."
            )


# ============================================================
# LIGHTING
# ============================================================

def randomize_light(center_xy=None):
    """Randomize light energy/color/position.

    center_xy: world-space (x, y) the light should orbit, e.g. the subject's
    own bbox center. Previously this always orbited the world origin
    (0, 0), which under-lit any subject placed away from the origin -
    this disproportionately hit the manually-assembled Stage_02-07
    collections (which do not sit at the origin) while leaving Stage_01
    (freshly imported near the origin) unaffected. Defaults to the
    origin so any other caller keeps the old behaviour.
    """
    light_obj = get_object(LIGHT_NAME)
    if light_obj is None:
        return
    ld = light_obj.data
    ld.energy = random.uniform(LIGHT_ENERGY_MIN, LIGHT_ENERGY_MAX)
    warm = random.uniform(0.92, 1.0)
    cool = random.uniform(0.92, 1.0)
    ld.color = (1.0, warm, cool)

    if center_xy is None:
        center_x, center_y = 0.0, 0.0
    else:
        center_x, center_y = center_xy.x, center_xy.y

    angle = random.uniform(0, 2 * math.pi)
    radius = random.uniform(0.15, 0.4)
    height = random.uniform(0.3, 0.7)
    plate_z = get_plate_top_z()
    light_obj.location = (
        center_x + radius * math.cos(angle),
        center_y + radius * math.sin(angle),
        plate_z + height,
    )


# ============================================================
# OCCLUSION
# ============================================================

def move_to_scene_root(obj):
    """Keep helper objects in the scene root collection only.

    V18 fix: the occluder used to be linked to bpy.context.collection, i.e.
    whatever collection happened to be active in the Outliner. When that was
    Stage_07, the occluder became part of Stage_07 and stage validation
    stopped with "object 'TEMP_OCCLUDER' must start with 'Stage_07_'".
    """
    root = bpy.context.scene.collection
    for coll in list(obj.users_collection):
        if coll != root:
            coll.objects.unlink(obj)
    if obj.name not in root.objects:
        root.objects.link(obj)


def make_occluder():
    occ = bpy.data.objects.get(OCCLUDER_NAME)
    if occ is not None:
        move_to_scene_root(occ)
        return occ
    mesh = bpy.data.meshes.new(OCCLUDER_NAME + "_mesh")
    occ = bpy.data.objects.new(OCCLUDER_NAME, mesh)
    bpy.context.scene.collection.objects.link(occ)
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(mesh)
    bm.free()
    mat = bpy.data.materials.new(OCCLUDER_NAME + "_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        g = random.uniform(0.2, 0.7)
        bsdf.inputs["Base Color"].default_value = (g, g, g, 1.0)
    occ.data.materials.append(mat)
    return occ


def place_occluder_over(part_obj):
    occ = make_occluder()
    occ.hide_render = False
    occ.hide_viewport = False
    center = get_world_bbox_center(part_obj)
    size = get_object_size(part_obj)
    s = size * random.uniform(0.20, 0.35)
    occ.scale = (s, s, s)
    off = size * random.uniform(0.35, 0.55)
    ang = random.uniform(0, 2 * math.pi)
    occ.location = (
        center.x + off * math.cos(ang),
        center.y + off * math.sin(ang),
        center.z + size * random.uniform(0.1, 0.4),
    )
    occ.rotation_euler = (
        random.uniform(0, math.pi),
        random.uniform(0, math.pi),
        random.uniform(0, math.pi),
    )
    bpy.context.view_layer.update()
    return occ


def hide_occluder():
    occ = bpy.data.objects.get(OCCLUDER_NAME)
    if occ is not None:
        occ.hide_render = True
        occ.hide_viewport = True


# ============================================================
# CAMERA
# ============================================================

def place_camera_at_point(center, dist, azimuth, elevation_deg, target=None):
    cam = get_object(CAMERA_NAME)
    if cam is None:
        return

    if target is None:
        target = center

    elev = math.radians(elevation_deg)
    x = center.x + dist * math.cos(elev) * math.cos(azimuth)
    y = center.y + dist * math.cos(elev) * math.sin(azimuth)
    z = center.z + dist * math.sin(elev)

    cam.location = (x, y, z)
    direction = target - Vector((x, y, z))
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()


def project_point(scene, cam, world_point):
    from bpy_extras.object_utils import world_to_camera_view
    co = world_to_camera_view(scene, cam, world_point)
    return co.x, co.y, (co.z > 0.0)


def compute_yolo_bbox_raw(scene, cam, obj):
    mw = eval_matrix_world(obj)
    corners = [mw @ Vector(c) for c in obj.bound_box]

    xs = []
    ys = []
    any_in_front = False

    for corner in corners:
        u, v, in_front = project_point(scene, cam, corner)
        if in_front:
            any_in_front = True
        xs.append(u)
        ys.append(v)

    if not any_in_front:
        return None

    return (min(xs), max(xs), min(ys), max(ys))


def bbox_fits_in_frame(scene, cam, obj):
    raw = compute_yolo_bbox_raw(scene, cam, obj)
    if raw is None:
        return False

    min_x, max_x, min_y, max_y = raw

    lo = (1.0 - FRAME_FIT_MARGIN) / 2.0
    hi = 1.0 - lo

    return (
        min_x >= lo
        and max_x <= hi
        and min_y >= lo
        and max_y <= hi
    )


def compute_bbox_fraction(scene, cam, obj):
    raw = compute_yolo_bbox_raw(scene, cam, obj)
    if raw is None:
        return None
    min_x, max_x, min_y, max_y = raw
    return max(max_x - min_x, max_y - min_y)


def choose_target_fraction():
    r = random.random()
    if r < 0.25:
        return random.uniform(TARGET_FAR_MIN, TARGET_FAR_MAX)
    if r < 0.75:
        return random.uniform(TARGET_MED_MIN, TARGET_MED_MAX)
    return random.uniform(TARGET_CLOSE_MIN, TARGET_CLOSE_MAX)


def choose_camera_elevation():
    r = random.random()
    if r < 0.20:
        return random.uniform(CAM_ELEV_LOW_MIN, CAM_ELEV_LOW_MAX)
    if r < 0.80:
        return random.uniform(CAM_ELEV_LOW_MAX, CAM_ELEV_HIGH_MIN)
    return random.uniform(CAM_ELEV_HIGH_MIN, CAM_ELEV_HIGH_MAX)


def place_camera_random(target_obj):
    scene = bpy.context.scene
    cam = get_object(CAMERA_NAME)
    if cam is None:
        return False

    center = get_world_bbox_center(target_obj)
    target_fraction = choose_target_fraction()
    azimuth = random.uniform(0, 2 * math.pi)
    elevation = choose_camera_elevation()

    lo = CAM_DIST_SEARCH_MIN
    hi = CAM_DIST_SEARCH_MAX

    for _ in range(5):
        place_camera_at_point(center, hi, azimuth, elevation, center)
        frac = compute_bbox_fraction(scene, cam, target_obj)
        if frac is not None and frac <= target_fraction:
            break
        hi *= 1.5

    for _ in range(18):
        mid = (lo + hi) / 2.0
        place_camera_at_point(center, mid, azimuth, elevation, center)
        frac = compute_bbox_fraction(scene, cam, target_obj)

        if frac is None or frac > target_fraction:
            lo = mid
        else:
            hi = mid

    place_camera_at_point(center, hi, azimuth, elevation, center)

    if not bbox_fits_in_frame(scene, cam, target_obj):
        place_camera_at_point(center, hi * 0.92, azimuth, elevation, center)

    return bbox_fits_in_frame(scene, cam, target_obj)


def compute_yolo_bbox(scene, cam, obj):
    raw = compute_yolo_bbox_raw(scene, cam, obj)
    if raw is None:
        return None
    min_x, max_x, min_y, max_y = raw
    min_x = max(0.0, min(1.0, min_x))
    max_x = max(0.0, min(1.0, max_x))
    min_y = max(0.0, min(1.0, min_y))
    max_y = max(0.0, min(1.0, max_y))
    if max_x - min_x <= 0.0 or max_y - min_y <= 0.0:
        return None
    x_center = (min_x + max_x) / 2.0
    width = max_x - min_x
    y_center = 1.0 - (min_y + max_y) / 2.0
    height = max_y - min_y
    return (x_center, y_center, width, height)


# ============================================================
# RENDER ONE IMAGE + LABEL (+ ndjson)
# ============================================================

def render_and_verify(scene, image_path):
    scene.render.filepath = str(image_path)
    try:
        bpy.ops.render.render(write_still=True)
    except Exception as exc:
        log(f"  RENDER ERROR: {exc}")
        return False

    if VERIFY_RENDER_OUTPUT and not image_path.exists():
        log(f"  RENDER ERROR: PNG was not created: {image_path}")
        return False

    return True


def make_clean_render_copy(source_obj):
    temp = source_obj.copy()
    if source_obj.data is not None:
        temp.data = source_obj.data.copy()

    temp.name = f"__DATASET_TMP__{source_obj.name}"
    bpy.context.scene.collection.objects.link(temp)

    temp.parent = None
    temp.matrix_world = eval_matrix_world(source_obj).copy()

    for constraint in list(temp.constraints):
        temp.constraints.remove(constraint)

    temp.hide_render = False
    temp.hide_viewport = False
    bpy.context.view_layer.update()

    log(
        f"  {source_obj.name}: using clean temporary render copy "
        f"'{temp.name}' (parent/constraints removed)."
    )
    return temp


def remove_clean_render_copy(obj):
    if obj is None:
        return
    if obj.name.startswith("__DATASET_TMP__"):
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.context.view_layer.update()


def render_one(class_name, object_name, index):
    scene = bpy.context.scene
    cam = get_object(CAMERA_NAME)

    if cam is None:
        log("ERROR: Camera not found.")
        return False

    source_obj = show_only(object_name)
    if source_obj is None:
        log(f"ERROR: Blender object '{object_name}' not found.")
        return False

    if class_name == "threaded_pin":
        obj = make_clean_render_copy(source_obj)
    else:
        obj = source_obj

    split = choose_split(index)
    stem = f"{class_name}_{index:04d}"
    image_path = IMAGES_DIR / split / f"{stem}.png"
    label_path = LABELS_DIR / split / f"{stem}.txt"

    try:
        for attempt in range(1, MAX_RENDER_ATTEMPTS_PER_IMAGE + 1):
            place_part_on_plate(obj)

            if not validate_part_placement(obj):
                log(f"  retry {attempt}/{MAX_RENDER_ATTEMPTS_PER_IMAGE}: placement")
                continue

            randomize_light(get_world_bbox_center(obj))
            set_random_background()

            if not place_camera_random(obj):
                log(f"  retry {attempt}/{MAX_RENDER_ATTEMPTS_PER_IMAGE}: camera/framing")
                continue

            if not validate_camera_position():
                log(f"  retry {attempt}/{MAX_RENDER_ATTEMPTS_PER_IMAGE}: camera position")
                continue

            bbox = compute_yolo_bbox(scene, cam, obj)
            if bbox is None:
                log(f"  retry {attempt}/{MAX_RENDER_ATTEMPTS_PER_IMAGE}: bbox")
                continue

            x_c, y_c, w, h = bbox
            class_id = CLASS_ID[class_name]

            if image_path.exists():
                image_path.unlink()

            if not render_and_verify(scene, image_path):
                log(f"  retry {attempt}/{MAX_RENDER_ATTEMPTS_PER_IMAGE}: render")
                continue

            label_path.write_text(
                f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n",
                encoding="utf-8",
            )

            boxes = [[class_id, round(x_c, 6), round(y_c, 6),
                      round(w, 6), round(h, 6)]]
            append_ndjson_record(image_path, RES_X, RES_Y, boxes, split)

            log(
                f"  OK: class={class_name} id={class_id} "
                f"#{index:03d} split={split} attempt={attempt}"
            )
            return True

        log(
            f"  FAILED: class={class_name} #{index} after "
            f"{MAX_RENDER_ATTEMPTS_PER_IMAGE} attempts"
        )
        return False

    finally:
        if class_name == "threaded_pin":
            remove_clean_render_copy(obj)



# ============================================================
# SUBASSEMBLY DATASET
# ============================================================


def subassembly_log(msg):
    print(f"[dataset] [SUBASSEMBLY] {msg}")


def get_stage_collection(stage_name):
    coll = bpy.data.collections.get(stage_name)
    if coll is None:
        raise RuntimeError(
            f"Required Blender collection '{stage_name}' not found."
        )
    return coll


def is_helper_object(obj):
    """Generator helpers that must never count as assembly parts."""
    name = obj.name
    return (
        name in (OCCLUDER_NAME, GROUND_NAME, FILL_LIGHT_NAME, "TEMP_BACKDROP")
        or name.startswith("TEMP_")
        or name.startswith("__DATASET_TMP__")
        or name.startswith(OCCLUDER_NAME + ".")
    )


def collection_mesh_objects(coll):
    if coll is None:
        return []
    return [
        obj for obj in coll.all_objects
        if obj is not None
        and isinstance(obj, bpy.types.Object)
        and obj.type == "MESH"
        and not is_helper_object(obj)
    ]


def infer_stage_class(obj, stage_name):
    prefix = stage_name.lower() + "_"
    name = obj.name.lower()

    if not name.startswith(prefix):
        raise RuntimeError(
            f"{stage_name}: object '{obj.name}' must start with '{stage_name}_'."
        )

    suffix = name[len(prefix):]

    if suffix.startswith("body_a"):
        return "body_a"
    if suffix.startswith("body_b"):
        return "body_b"
    if suffix.startswith("block_pin"):
        return "block_pin"
    if suffix.startswith("bushing_"):
        return "bushing"
    if suffix.startswith("nut"):
        return "nut"
    if suffix.startswith("knob"):
        return "knob"
    if suffix.startswith("threaded_pin"):
        return "threaded_pin"

    raise RuntimeError(
        f"{stage_name}: cannot infer YOLO class from '{obj.name}'."
    )


def validate_stage_collections():
    subassembly_log("=" * 70)
    subassembly_log("VALIDATING MANUAL BLENDER STAGE COLLECTIONS")
    subassembly_log("=" * 70)

    for stage_name in STAGE_COLLECTIONS:
        coll = get_stage_collection(stage_name)
        objects = collection_mesh_objects(coll)

        if not objects:
            raise RuntimeError(
                f"{stage_name}: collection contains no mesh objects."
            )

        observed = {}
        for obj in objects:
            dataset_class = infer_stage_class(obj, stage_name)
            obj["dataset_class"] = dataset_class
            observed[dataset_class] = observed.get(dataset_class, 0) + 1

        expected = STAGE_EXPECTED_SEQUENCE[stage_name]
        if observed != expected:
            raise RuntimeError(
                f"{stage_name}: collection contents do not match expected "
                f"assembly sequence.\nExpected: {expected}\nFound: {observed}\n"
                f"Objects: {[obj.name for obj in objects]}"
            )

        subassembly_log(f"OK {stage_name}: {[obj.name for obj in objects]}")

    subassembly_log("ALL 7 MANUAL STAGE COLLECTIONS VALID.")
    subassembly_log("=" * 70)


def hide_all_stage_objects():
    for stage_name in STAGE_COLLECTIONS:
        coll = bpy.data.collections.get(stage_name)
        if coll is None:
            continue

        for obj in list(coll.all_objects):
            if (
                obj is not None
                and isinstance(obj, bpy.types.Object)
                and obj.type == "MESH"
            ):
                obj.hide_render = True



def hide_all_non_stage_meshes(stage_objects):
    """Hide every mesh except the active stage and required background meshes.

    This is the critical isolation step for subassembly rendering. The old
    single-part source objects and any leftover meshes from previous tests
    are hidden automatically.
    """
    active_names = {
        obj.name
        for obj in stage_objects
        if obj is not None and isinstance(obj, bpy.types.Object)
    }

    allowed_background = {
        GROUND_NAME,
        WORKBENCH_NAME,
    }

    for obj in list(bpy.context.scene.objects):
        if (
            obj is None
            or not isinstance(obj, bpy.types.Object)
            or obj.type != "MESH"
        ):
            continue

        if obj.name in active_names:
            obj.hide_render = False
            obj.hide_viewport = False
        elif obj.name in allowed_background:
            obj.hide_render = False
            obj.hide_viewport = False
        else:
            obj.hide_render = True
            obj.hide_viewport = True

    # Occluder is never part of the assembly smoke/final dataset.
    occluder = bpy.data.objects.get(OCCLUDER_NAME)
    if occluder is not None:
        occluder.hide_render = True
        occluder.hide_viewport = True


def validate_only_stage_meshes_visible(stage_objects, stage_name):
    """Fail-safe check: no unexpected mesh may be visible in a stage render."""
    active_names = {
        obj.name
        for obj in stage_objects
        if obj is not None and isinstance(obj, bpy.types.Object)
    }

    allowed = active_names | {GROUND_NAME, WORKBENCH_NAME}
    unexpected = []

    for obj in bpy.context.scene.objects:
        if (
            obj is None
            or not isinstance(obj, bpy.types.Object)
            or obj.type != "MESH"
            or obj.hide_render
        ):
            continue

        if obj.name not in allowed:
            unexpected.append(obj.name)

    if unexpected:
        subassembly_log(
            f"ISOLATION FAILURE {stage_name}: unexpected visible meshes: "
            f"{unexpected}"
        )
        return False

    return True


def show_only_stage(stage_name):
    hide_all_stage_objects()
    hide_all_targets()

    coll = get_stage_collection(stage_name)
    objects = collection_mesh_objects(coll)

    for obj in objects:
        obj["dataset_class"] = infer_stage_class(obj, stage_name)
        obj.hide_render = False
        obj.hide_viewport = False

    hide_all_non_stage_meshes(objects)

    if not validate_only_stage_meshes_visible(objects, stage_name):
        raise RuntimeError(
            f"{stage_name}: scene isolation failed before rendering."
        )

    bpy.context.view_layer.update()
    return objects



def capture_visibility_states():
    stage_names = set()
    for stage_name in STAGE_COLLECTIONS:
        coll = bpy.data.collections.get(stage_name)
        if coll is not None:
            stage_names.update(
                obj.name
                for obj in list(coll.all_objects)
                if obj is not None and isinstance(obj, bpy.types.Object)
            )

    source_names = set(ALL_TARGET_OBJECTS)
    states = {}

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if obj.name in stage_names or obj.name in source_names:
            states[obj.name] = (obj.hide_render, obj.hide_viewport)

    return states


def restore_visibility_states(states):
    for name, (hide_render, hide_viewport) in states.items():
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_render = hide_render
            obj.hide_viewport = hide_viewport


def _group_bbox_corners(objects):
    corners = []
    for obj in objects:
        if (
            obj is not None
            and isinstance(obj, bpy.types.Object)
            and obj.type == "MESH"
        ):
            corners.extend(_world_bbox_corners(obj))

    if not corners:
        raise RuntimeError("Assembly contains no mesh geometry.")

    return corners


def group_bbox_limits(objects):
    corners = _group_bbox_corners(objects)
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def group_bbox_center(objects):
    corners = _group_bbox_corners(objects)
    return sum(corners, Vector((0, 0, 0))) / len(corners)


def mathutils_matrix_rotation(rx, ry, rz):
    """Return a Blender 4x4 Euler rotation matrix."""
    euler = mathutils.Euler((rx, ry, rz), "XYZ")
    return euler.to_matrix().to_4x4()


def rigidly_place_stage(objects):
    """Apply one independent random rigid transform to the whole assembly."""
    if not objects:
        return {
            "yaw_deg": 0.0,
            "tilt_x_deg": 0.0,
            "tilt_y_deg": 0.0,
            "dx": 0.0,
            "dy": 0.0,
        }

    center = group_bbox_center(objects)

    yaw_deg = random.uniform(
        COLLECTION_ASSEMBLY_YAW_MIN_DEG,
        COLLECTION_ASSEMBLY_YAW_MAX_DEG,
    )
    tilt_x_deg = random.uniform(
        -COLLECTION_ASSEMBLY_TILT_MAX_DEG,
        COLLECTION_ASSEMBLY_TILT_MAX_DEG,
    )
    tilt_y_deg = random.uniform(
        -COLLECTION_ASSEMBLY_TILT_MAX_DEG,
        COLLECTION_ASSEMBLY_TILT_MAX_DEG,
    )

    yaw = math.radians(yaw_deg)
    tilt_x = math.radians(tilt_x_deg)
    tilt_y = math.radians(tilt_y_deg)
    rot = mathutils_matrix_rotation(tilt_x, tilt_y, yaw)

    for obj in objects:
        mw = obj.matrix_world.copy()
        mw.translation -= center
        mw = rot @ mw
        mw.translation += center
        obj.matrix_world = mw

    bpy.context.view_layer.update()

    min_z = group_bbox_limits(objects)[4]
    plate_z = get_plate_top_z()

    dx = random.uniform(-COLLECTION_XY_JITTER, COLLECTION_XY_JITTER)
    dy = random.uniform(-COLLECTION_XY_JITTER, COLLECTION_XY_JITTER)

    translation = Vector((dx, dy, plate_z - min_z))

    for obj in objects:
        mw = obj.matrix_world.copy()
        mw.translation += translation
        obj.matrix_world = mw

    bpy.context.view_layer.update()

    return {
        "yaw_deg": yaw_deg,
        "tilt_x_deg": tilt_x_deg,
        "tilt_y_deg": tilt_y_deg,
        "dx": dx,
        "dy": dy,
    }


# ------------------------------------------------------------
# FINAL STAGE-GEOMETRY SAFETY AUDIT
# ------------------------------------------------------------
# This does NOT move/correct manually assembled parts. It detects a part that
# has accidentally been left floating away from the rest of the assembly and
# prevents that bad geometry from entering the YOLO dataset.
STAGE_COMPONENT_GAP_FACTOR = 0.45
STAGE_COMPONENT_GAP_MIN_M = 0.008


def _object_bbox_limits_world(obj):
    corners = _world_bbox_corners(obj)
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return (
        min(xs), max(xs),
        min(ys), max(ys),
        min(zs), max(zs),
    )


def _bbox_gap_world(obj_a, obj_b):
    a = _object_bbox_limits_world(obj_a)
    b = _object_bbox_limits_world(obj_b)

    dx = max(a[0] - b[1], b[0] - a[1], 0.0)
    dy = max(a[2] - b[3], b[2] - a[3], 0.0)
    dz = max(a[4] - b[5], b[4] - a[5], 0.0)

    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _bbox_diagonal_world(obj):
    a = _object_bbox_limits_world(obj)
    return math.sqrt(
        (a[1] - a[0]) ** 2 +
        (a[3] - a[2]) ** 2 +
        (a[5] - a[4]) ** 2
    )


def validate_stage_assembly_geometry(objects, stage_name):
    """
    Check whether the manually assembled stage is spatially coherent.

    The check deliberately does not invent or modify CAD placement. It only
    prevents obviously detached components from being rendered into training
    data. This is especially important for bushings that accidentally remain
    beside the assembly.
    """
    if len(objects) <= 1:
        return True

    body_a = next(
        (obj for obj in objects if obj.get("dataset_class") == "body_a"),
        None,
    )

    if body_a is not None:
        reference = body_a
        ref_diag = max(_bbox_diagonal_world(reference), 1e-6)
        threshold = max(
            STAGE_COMPONENT_GAP_MIN_M,
            STAGE_COMPONENT_GAP_FACTOR * ref_diag,
        )

        detached = []
        for obj in objects:
            if obj is reference:
                continue

            gap = _bbox_gap_world(reference, obj)
            if gap > threshold:
                detached.append((obj.name, obj.get("dataset_class"), gap))

        if detached:
            for name, cls, gap in detached:
                subassembly_log(
                    f"GEOMETRY WARNING {stage_name}: '{name}' ({cls}) "
                    f"is detached from body_a by bbox_gap={gap:.5f} m "
                    f"(allowed <= {threshold:.5f} m)."
                )
            return False

        return True

    # Stage_06 has no body_a. For that locking subassembly, validate every
    # component against the nearest other component.
    detached = []
    for obj in objects:
        other_gaps = [
            _bbox_gap_world(obj, other)
            for other in objects
            if other is not obj
        ]
        nearest = min(other_gaps) if other_gaps else 0.0
        local_threshold = max(
            STAGE_COMPONENT_GAP_MIN_M,
            STAGE_COMPONENT_GAP_FACTOR *
            max(_bbox_diagonal_world(obj), 1e-6),
        )
        if nearest > local_threshold:
            detached.append((obj.name, obj.get("dataset_class"), nearest))

    if detached:
        for name, cls, gap in detached:
            subassembly_log(
                f"GEOMETRY WARNING {stage_name}: '{name}' ({cls}) "
                f"has no nearby assembly component; nearest bbox gap="
                f"{gap:.5f} m."
            )
        return False

    return True


def audit_all_stage_geometry():
    """
    Pre-flight audit for the manual Stage_02-Stage_07 collections.

    Stage_01 is intentionally excluded because final generation does NOT use
    the manual Stage_01 collection. It loads the proven authoritative
    CAD-derived body_a + bushing pair later, and that loader validates the
    authoritative geometry itself.
    """
    subassembly_log("=" * 70)
    subassembly_log("PRE-FLIGHT STAGE GEOMETRY AUDIT")
    subassembly_log("=" * 70)

    failures = []

    subassembly_log(
        "GEOMETRY OK Stage_01: manual collection bypassed; "
        "authoritative CAD geometry will be validated at load time."
    )

    for stage_name in STAGE_COLLECTIONS:
        if stage_name == "Stage_01":
            continue

        coll = get_stage_collection(stage_name)
        objects = collection_mesh_objects(coll)

        valid = validate_stage_assembly_geometry(objects, stage_name)

        if valid:
            subassembly_log(f"GEOMETRY OK {stage_name}")
        else:
            subassembly_log(
                f"GEOMETRY FAIL {stage_name}: "
                "one or more components appear detached."
            )
            failures.append(stage_name)

    if failures:
        raise RuntimeError(
            "Stage geometry audit failed for: "
            + ", ".join(failures)
            + ". Fix the affected manual Stage collections before "
              "generating training data."
        )

    subassembly_log(
        "MANUAL STAGE_02-STAGE_07 COLLECTIONS PASSED GEOMETRY AUDIT."
    )
    subassembly_log("=" * 70)



def validate_collection_stage_placement(objects, stage_name):
    if not objects:
        return False

    min_x, max_x, min_y, max_y, min_z, max_z = group_bbox_limits(objects)
    plate_z = get_plate_top_z()

    if max_z < plate_z:
        subassembly_log(
            f"{stage_name}: assembly is below workbench."
        )
        return False

    if min_z < plate_z - 0.005:
        subassembly_log(
            f"{stage_name}: assembly penetrates workbench "
            f"(min_z={min_z:.6f}, plate_z={plate_z:.6f})."
        )
        return False

    if max_x <= min_x or max_y <= min_y or max_z <= min_z:
        return False

    return True


def compute_group_bbox_raw(scene, cam, objects):
    corners = _group_bbox_corners(objects)
    xs, ys = [], []

    for corner in corners:
        u, v, in_front = project_point(scene, cam, corner)
        if not in_front:
            return None
        xs.append(u)
        ys.append(v)

    if not xs:
        return None

    return min(xs), max(xs), min(ys), max(ys)


def compute_group_bbox_fraction(scene, cam, objects):
    raw = compute_group_bbox_raw(scene, cam, objects)
    if raw is None:
        return None

    min_x, max_x, min_y, max_y = raw
    return max(max_x - min_x, max_y - min_y)


def _set_collection_camera(
    target_fraction,
    objects,
    azimuth_override=None,
    elevation_override=None,
    shift_x=None,
    shift_y=None,
):
    """Set a full 3-D camera viewpoint and solve distance for occupancy."""
    scene = bpy.context.scene
    cam = get_object(CAMERA_NAME)
    if cam is None or not objects:
        return False

    center = group_bbox_center(objects)
    corners = _group_bbox_corners(objects)
    radius = max((c - center).length for c in corners)
    radius = max(radius, 1e-6)

    cam.data.type = "PERSP"
    cam.data.sensor_fit = "AUTO"

    azimuth = (
        math.radians(azimuth_override)
        if azimuth_override is not None
        else random.uniform(0.0, 2.0 * math.pi)
    )
    elevation = (
        float(elevation_override)
        if elevation_override is not None
        else choose_camera_elevation()
    )

    # Move the object away from the exact center of the image. This is a
    # camera framing variation, not a geometry/assembly modification.
    cam.data.shift_x = (
        float(shift_x)
        if shift_x is not None
        else random.uniform(-CAMERA_SHIFT_MAX, CAMERA_SHIFT_MAX)
    )
    cam.data.shift_y = (
        float(shift_y)
        if shift_y is not None
        else random.uniform(-CAMERA_SHIFT_MAX, CAMERA_SHIFT_MAX)
    )

    fov = min(float(cam.data.angle_x), float(cam.data.angle_y))
    distance = radius / max(
        target_fraction * math.tan(max(fov, 1e-6) / 2.0),
        1e-9,
    )
    distance = max(distance * 1.05, 0.01)

    best = None

    for _ in range(24):
        place_camera_at_point(center, distance, azimuth, elevation, center)
        frac = compute_group_bbox_fraction(scene, cam, objects)

        if frac is None or frac <= 1e-8:
            distance *= 1.35
            continue

        error = abs(frac - target_fraction)

        if best is None or error < best[0]:
            best = (error, distance, frac)

        if error <= 0.012:
            break

        distance *= frac / target_fraction
        distance = max(distance, 0.005)

    if best is None:
        return False

    _, best_distance, _ = best
    place_camera_at_point(center, best_distance, azimuth, elevation, center)

    raw = compute_group_bbox_raw(scene, cam, objects)
    if raw is None:
        return False

    min_x, max_x, min_y, max_y = raw
    lo = (1.0 - FRAME_FIT_MARGIN) / 2.0
    hi = 1.0 - lo

    return (
        min_x >= lo and max_x <= hi
        and min_y >= lo and max_y <= hi
    )

def _sample_face_centers_world(obj, max_samples=None):
    if obj.type != "MESH" or not obj.data.polygons:
        return []

    count = max_samples or VISIBLE_FACE_SAMPLES
    polys = list(obj.data.polygons)

    if len(polys) > count:
        stride = max(len(polys) // count, 1)
        polys = polys[::stride][:count]

    mw = eval_matrix_world(obj)
    verts = obj.data.vertices

    return [
        mw @ (
            sum((verts[i].co for i in poly.vertices), Vector()) /
            len(poly.vertices)
        )
        for poly in polys
    ]


# ------------------------------------------------------------
# V18 geometry helpers (vertex based, numpy accelerated)
# ------------------------------------------------------------
_LOCAL_VERT_CACHE = {}


def _local_vertices(obj):
    """Object-space vertex coordinates as an (N, 3) array, cached per mesh."""
    if obj is None or obj.type != "MESH":
        return np.zeros((0, 3), dtype=np.float64)
    key = obj.data.as_pointer()
    cached = _LOCAL_VERT_CACHE.get(key)
    if cached is not None and len(cached) == len(obj.data.vertices):
        return cached
    n = len(obj.data.vertices)
    arr = np.empty(n * 3, dtype=np.float64)
    if n:
        obj.data.vertices.foreach_get("co", arr)
    arr = arr.reshape(n, 3)
    _LOCAL_VERT_CACHE[key] = arr
    return arr


def _world_vertices(obj):
    local = _local_vertices(obj)
    if not len(local):
        return local
    m = np.array(eval_matrix_world(obj), dtype=np.float64)
    return local @ m[:3, :3].T + m[:3, 3]


def _vertex_z_limits(objects):
    """Real lowest / highest point of the geometry (not bounding-box corners,
    which lie below the mesh once a part is rotated)."""
    zs = []
    for obj in objects:
        w = _world_vertices(obj)
        if len(w):
            zs.append((w[:, 2].min(), w[:, 2].max()))
    if not zs:
        return None, None
    return min(z[0] for z in zs), max(z[1] for z in zs)


def _sample_surface_points_world(obj):
    """Face centres (V17) plus a spread of vertices, so that the samples
    reach the true silhouette edges of the part."""
    pts = list(_sample_face_centers_world(obj))
    w = _world_vertices(obj)
    n = len(w)
    if n:
        stride = max(n // VISIBILITY_VERTEX_SAMPLES, 1)
        for row in w[::stride][:VISIBILITY_VERTEX_SAMPLES]:
            pts.append(Vector((float(row[0]), float(row[1]), float(row[2]))))
    return pts


def visibility_analysis(scene, cam, obj):
    """Return (bbox, visible_fraction, visible_points).

    visible_fraction = share of sampled surface points that the camera can
    actually see: inside the frame and not hidden behind another object.
    bbox is the YOLO box (x_c, y_c, w, h) of the visible points, or None.
    """
    points = _sample_surface_points_world(obj)
    if not points:
        return None, 0.0, 0

    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin = cam.matrix_world.translation.copy()
    visible_uv = []

    for point in points:
        u, v, in_front = project_point(scene, cam, point)
        if not in_front or u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
            continue

        direction = point - origin
        distance = direction.length
        if distance <= 1e-9:
            continue
        direction.normalize()

        try:
            hit, _loc, _nrm, _idx, hit_obj, _mat = scene.ray_cast(
                depsgraph, origin, direction, distance=distance + 1e-4)
        except TypeError:
            hit, _loc, _nrm, _idx, hit_obj, _mat = scene.ray_cast(
                depsgraph, origin, direction)

        if hit and hit_obj is not None and hit_obj.name == obj.name:
            visible_uv.append((u, v))

    fraction = len(visible_uv) / float(len(points))
    count = len(visible_uv)
    if count < 2:
        return None, fraction, count

    xs = [p[0] for p in visible_uv]
    ys = [p[1] for p in visible_uv]
    min_x, max_x = max(0.0, min(xs)), min(1.0, max(xs))
    min_y, max_y = max(0.0, min(ys)), min(1.0, max(ys))
    if max_x <= min_x or max_y <= min_y:
        return None, fraction, count

    return (
        ((min_x + max_x) / 2.0, 1.0 - (min_y + max_y) / 2.0,
         max_x - min_x, max_y - min_y),
        fraction,
        count,
    )


def visible_object_bbox(scene, cam, obj):
    """Return (bbox, visible). V18: a part is only labelled when at least
    MIN_VISIBLE_POINTS_LABEL of its sampled points are visible. V17 labelled
    any part with two visible face centres, producing sliver boxes."""
    if not VISIBILITY_AWARE_LABELS:
        return compute_yolo_bbox(scene, cam, obj), True

    bbox, _fraction, count = visibility_analysis(scene, cam, obj)
    if bbox is None or count < MIN_VISIBLE_POINTS_LABEL:
        return None, False
    return bbox, True


def choose_subassembly_split(index):
    count = SMOKE_IMAGES_PER_STAGE if SMOKE_TEST else IMAGES_PER_STAGE

    train_end = max(1, int(count * SPLIT_TRAIN))
    val_end = max(train_end + 1, int(count * (SPLIT_TRAIN + SPLIT_VAL)))

    if index <= train_end:
        return "train"
    if index <= val_end:
        return "val"
    return "test"



def clean_subassembly_output_for_production():
    """Remove only the subassembly dataset output, never the source Blender data."""
    if SMOKE_TEST or not CLEAN_SUBASSEMBLY_OUTPUT_BEFORE_PRODUCTION:
        return

    if not SUBASSEMBLY_OUTPUT_DIR.exists():
        return

    subassembly_log(
        f"CLEANING old production subassembly output: {SUBASSEMBLY_OUTPUT_DIR}"
    )

    for path in sorted(SUBASSEMBLY_OUTPUT_DIR.iterdir(), reverse=True):
        if path.is_dir():
            for child in sorted(path.rglob("*"), reverse=True):
                if child.is_file():
                    try:
                        child.unlink()
                    except OSError:
                        pass
                elif child.is_dir():
                    try:
                        child.rmdir()
                    except OSError:
                        pass
            try:
                path.rmdir()
            except OSError:
                pass
        else:
            try:
                path.unlink()
            except OSError:
                pass


def write_assembly_sequence_metadata():
    """Write the exact expected assembly order for later state-recognition code."""
    path = SUBASSEMBLY_OUTPUT_DIR / "assembly_sequence.json"
    payload = {
        "sequence": STAGE_COLLECTIONS,
        "expected_components": STAGE_EXPECTED_SEQUENCE,
        "note": (
            "YOLO classes identify visible components. This file defines the "
            "expected assembly-state order for later warning/state-recognition logic."
        ),
    }
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def write_subassembly_data_yaml():
    path = SUBASSEMBLY_OUTPUT_DIR / "data.yaml"
    names = "\n".join(
        f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)
    )

    content = (
        f"path: {SUBASSEMBLY_OUTPUT_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names}\n"
    )

    path.write_text(content, encoding="utf-8")
    return path


def initialize_collection_subassembly_ndjson():
    path = SUBASSEMBLY_OUTPUT_DIR / "annotations.ndjson"
    header = {
        "type": "dataset",
        "task": "object_detection",
        "classes": CLASS_NAMES,
        "dataset_mode": "manual_blender_collections",
        "stages": STAGE_COLLECTIONS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    SUBASSEMBLY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(header) + "\n", encoding="utf-8")
    return path


def append_collection_subassembly_ndjson(
    path, image_path, boxes, split, stage_name
):
    record = {
        "file": str(image_path),
        "type": "image",
        "width": RES_X,
        "height": RES_Y,
        "annotations": {"boxes": boxes} if boxes else {},
        "split": split,
        "stage": stage_name,
        "stage_id": int(stage_name.split("_")[1]),
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")



# ============================================================
# AUTHORITATIVE STAGE-01 GEOMETRY
# ============================================================
# Stage 01 uses the verified CAD-derived component STLs whose common
# coordinates place the bushing correctly inside body_a.
# Stages 02-07 remain collection-driven and are intentionally unchanged.
# ============================================================

STAGE01_AUTHORITATIVE_OUTPUT_DIR = OUTPUT_DIR / "subassembly_stage01_authoritative_test"
STAGE01_FIX_OUTPUT_DIR = STAGE01_AUTHORITATIVE_OUTPUT_DIR
STAGE01_FIX_IMAGES_PER_RUN = 36


# ---------------------------------------------------------------------
# GEOMETRY REMOVED FOR PUBLICATION
#
# The original script embedded the Stage-01 body and bushing meshes here
# as base64 binary STL.  That geometry is derived from a STEP file
# supplied by an industrial partner and is not published.
#
# The loader below falls back to local STL files under
# SUBASSEMBLY_ASSETS_DIR / "stage_01" / ...
# Place your own CAD there:
#     stage_01/1_body_a.stl   (or body_a.stl)
#     stage_01/4_bushing.stl  (or bushing_A.stl)
# ---------------------------------------------------------------------
_STAGE01_BODY_B64 = None  # geometry removed - see note above
_STAGE01_BUSH_B64 = None  # geometry removed - see note above
def stage01_fix_log(msg):
    print(f"[stage01-fix] {msg}")

def _stage01_decode_asset(b64_text, filename):
    # Prefer the verified embedded asset.  The previous V10 file contained a
    # corrupted Stage-01 body base64 payload; this implementation validates
    # the payload and, if necessary, falls back to the real local STL asset.
    try:
        clean = "".join(str(b64_text).split())
        clean += "=" * ((-len(clean)) % 4)
        data = base64.b64decode(clean, validate=True)
        if data[:5] != b"Subas" or len(data) < 84:
            raise ValueError("decoded asset is not a valid binary STL-like payload")
    except Exception as exc:
        # Actual Stage-01 files already present on the user's machine.
        candidates = [
            SUBASSEMBLY_ASSETS_DIR / "stage_01" / "1_body_a.stl",
            SUBASSEMBLY_ASSETS_DIR / "stage_01" / "body_a.stl",
        ] if "body_a" in filename else [
            SUBASSEMBLY_ASSETS_DIR / "stage_01" / "4_bushing.stl",
            SUBASSEMBLY_ASSETS_DIR / "stage_01" / "bushing_A.stl",
        ]
        for candidate in candidates:
            if candidate.exists():
                subassembly_log(
                    f"Stage_01 asset decode fallback: using local STL {candidate}"
                )
                return candidate
        raise RuntimeError(
            f"Stage-01 embedded asset '{filename}' is invalid and no local "
            f"fallback STL was found. Decode error: {exc}"
        )

    temp_dir = Path(tempfile.gettempdir()) / "thesis_stage01_authoritative"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / filename
    path.write_bytes(data)
    return path

def _stage01_import_stl(path, object_name):
    before = set(bpy.data.objects.keys())
    try:
        bpy.ops.wm.stl_import(
            filepath=str(path),
            global_scale=0.001,
            use_scene_unit=False,
            use_mesh_validate=True,
        )
    except Exception as first_exc:
        try:
            bpy.ops.import_mesh.stl(filepath=str(path))
        except Exception as second_exc:
            raise RuntimeError(
                f"Could not import Stage-01 STL {path}: "
                f"wm.stl_import={first_exc}; import_mesh.stl={second_exc}"
            )

    bpy.context.view_layer.update()
    created = [
        obj for obj in bpy.data.objects
        if obj.name not in before
        and obj is not None
        and isinstance(obj, bpy.types.Object)
        and obj.type == "MESH"
    ]
    if not created:
        raise RuntimeError(f"Stage-01 STL import created no mesh: {path}")

    obj = created[0]
    obj.name = object_name
    return obj

def _stage01_copy_material(source_names, target):
    source = None
    for name in source_names:
        candidate = bpy.data.objects.get(name)
        if (
            candidate is not None
            and isinstance(candidate, bpy.types.Object)
            and candidate.type == "MESH"
            and len(candidate.data.materials) > 0
        ):
            source = candidate
            break

    if source is None:
        raise RuntimeError(
            f"Could not find material source for {target.name}. Tried {source_names}"
        )

    target.data.materials.clear()
    for mat in source.data.materials:
        target.data.materials.append(mat)

    stage01_fix_log(
        f"Material copied: {target.name} <- {source.name} "
        f"({source.data.materials[0].name})"
    )

def _stage01_bbox(obj):
    corners = _world_bbox_corners(obj)
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))

def _stage01_gap(a, b):
    aa = _stage01_bbox(a)
    bb = _stage01_bbox(b)
    dx = max(aa[0] - bb[1], bb[0] - aa[1], 0.0)
    dy = max(aa[2] - bb[3], bb[2] - aa[3], 0.0)
    dz = max(aa[4] - bb[5], bb[4] - aa[5], 0.0)
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def _stage01_prepare_scene():
    for obj in list(bpy.context.scene.objects):
        if (
            obj is not None
            and isinstance(obj, bpy.types.Object)
            and obj.type == "MESH"
        ):
            obj.hide_render = True
            obj.hide_viewport = True

    for name in (GROUND_NAME, WORKBENCH_NAME):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_render = False
            obj.hide_viewport = False

    occluder = bpy.data.objects.get(OCCLUDER_NAME)
    if occluder is not None:
        occluder.hide_render = True
        occluder.hide_viewport = True

def _stage01_load_authoritative_geometry():
    body_path = _stage01_decode_asset(
        _STAGE01_BODY_B64, "stage_01_body_a_authoritative.stl"
    )
    bush_path = _stage01_decode_asset(
        _STAGE01_BUSH_B64, "stage_01_bushing_A_authoritative.stl"
    )

    body = _stage01_import_stl(body_path, "Stage_01_body_a_AUTHORITATIVE")
    bush = _stage01_import_stl(bush_path, "Stage_01_bushing_01_AUTHORITATIVE")

    body["dataset_class"] = "body_a"
    bush["dataset_class"] = "bushing"

    _stage01_copy_material(["1_body_a"], body)
    _stage01_copy_material(
        ["4_bushing", "4_bushing.001", "Stage_01_bushing_01", "Stage_02_bushing_01"],
        bush,
    )

    coll = bpy.data.collections.new("__STAGE01_AUTHORITATIVE_TEST__")
    bpy.context.scene.collection.children.link(coll)

    for obj in (body, bush):
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        coll.objects.link(obj)

    bpy.context.view_layer.update()
    return [body, bush], coll



def _cleanup_stage01_authoritative(objects, collection):
    """Remove temporary authoritative Stage-01 objects and collection."""
    for obj in list(objects or []):
        if obj is not None and obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

    if collection is not None and collection.name in bpy.data.collections:
        bpy.data.collections.remove(collection)

    bpy.context.view_layer.update()


def _prepare_authoritative_stage01_for_generation():
    """
    Load the proven Stage-01 CAD-derived body_a + bushing pair.
    Return temporary objects, collection, and their common-coordinate matrices.
    """
    _stage01_prepare_scene()
    objects, collection = _stage01_load_authoritative_geometry()

    base_matrices = {
        obj.name: obj.matrix_world.copy()
        for obj in objects
    }

    body = next(obj for obj in objects if obj.get("dataset_class") == "body_a")
    bush = next(obj for obj in objects if obj.get("dataset_class") == "bushing")

    body_box = _stage01_bbox(body)
    bush_box = _stage01_bbox(bush)
    bush_center = Vector((
        (bush_box[0] + bush_box[1]) / 2.0,
        (bush_box[2] + bush_box[3]) / 2.0,
        (bush_box[4] + bush_box[5]) / 2.0,
    ))

    inside_xy = (
        body_box[0] - 0.002 <= bush_center.x <= body_box[1] + 0.002
        and body_box[2] - 0.002 <= bush_center.y <= body_box[3] + 0.002
    )

    if not inside_xy:
        _cleanup_stage01_authoritative(objects, collection)
        raise RuntimeError(
            "AUTHORITATIVE Stage_01 check failed: bushing is outside body_a."
        )

    subassembly_log(
        "Stage_01 authoritative CAD geometry loaded: body_a + bushing."
    )
    subassembly_log(
        "Stage_01 manual Blender collection is intentionally bypassed."
    )

    return objects, collection, base_matrices


def stage_view_for_index(index):
    """
    Production view schedule.

    Every stage receives:
      45% CLOSE, 40% MEDIUM, 15% FAR.
    The schedule is interleaved so each split contains all three scales.
    """
    if SMOKE_TEST:
        return ["CLOSE", "MEDIUM", "FAR"][min(index - 1, 2)]
    return STAGE_VIEW_CYCLE[(index - 1) % len(STAGE_VIEW_CYCLE)]


def stage_target_fraction(view_name):
    if view_name == "CLOSE":
        return random.uniform(0.60, 0.80)
    if view_name == "MEDIUM":
        return random.uniform(0.38, 0.58)
    return random.uniform(0.16, 0.28)


def stage_camera_angles_for_index(index):
    """
    Generate a true 3-D viewpoint for every image.

    A golden-angle azimuth progression prevents repeated camera sides while
    five elevation bands ensure top/mid/low views are all represented.
    Random jitter makes consecutive samples non-identical.
    """
    zero = index - 1

    # 137.508° is the golden angle. The modulo sweep distributes viewpoints
    # around the full 360° without repeating a simple left/right cycle.
    golden_angle = 137.508
    azimuth = (zero * golden_angle) % 360.0
    azimuth += random.uniform(-CAMERA_RELATIVE_VIEW_JITTER_DEG,
                              CAMERA_RELATIVE_VIEW_JITTER_DEG)

    bands = STAGE_ELEVATION_BANDS_DEG
    elevation = bands[zero % len(bands)]
    elevation += random.uniform(
        -STAGE_ELEVATION_JITTER_DEG,
        STAGE_ELEVATION_JITTER_DEG,
    )

    elevation = max(12.0, min(78.0, elevation))
    return azimuth, elevation

def render_collection_stage_one(
    stage_name,
    objects,
    base_matrices,
    index,
    images_dir,
    labels_dir,
    ndjson_path,
):
    scene = bpy.context.scene
    cam = get_object(CAMERA_NAME)

    if cam is None:
        subassembly_log("ERROR: Camera not found.")
        return False

    split = choose_subassembly_split(index)
    stem = f"{stage_name.lower()}_{index:04d}"
    image_path = images_dir / split / f"{stem}.png"
    label_path = labels_dir / split / f"{stem}.txt"

    view_name = stage_view_for_index(index)
    target_fraction = stage_target_fraction(view_name)

    # Use a deterministic angle grid for complete viewpoint coverage, with
    # light random jitter so images do not look duplicated.
    azimuth_deg, elevation_deg = stage_camera_angles_for_index(index)

    for attempt in range(1, MAX_RENDER_ATTEMPTS_PER_IMAGE + 1):
        try:
            attempt_azimuth = azimuth_deg
            attempt_elevation = elevation_deg

            # Every image gets an independent object pose. Retries receive a
            # new camera side and new image-plane framing.
            if attempt > 1:
                attempt_azimuth = (
                    attempt_azimuth
                    + random.uniform(-22.0, 22.0) * min(attempt, 3)
                ) % 360.0
                attempt_elevation = max(
                    12.0,
                    min(
                        78.0,
                        attempt_elevation
                        + random.uniform(-8.0, 8.0) * min(attempt, 2),
                    ),
                )

            pose_meta = None

            for obj in objects:
                obj.matrix_world = base_matrices[obj.name].copy()
                obj.hide_render = False

            pose_meta = rigidly_place_stage(objects)

            # Re-apply strict isolation after any stage movement.
            hide_all_non_stage_meshes(objects)

            if not validate_only_stage_meshes_visible(objects, stage_name):
                subassembly_log(
                    f"{stage_name} #{index}: retry {attempt}: "
                    "unexpected visible mesh"
                )
                continue

            if not validate_collection_stage_placement(
                objects, stage_name
            ):
                subassembly_log(
                    f"{stage_name} #{index}: retry {attempt}: placement"
                )
                continue

            randomize_light(group_bbox_center(objects))
            set_random_background()

            shift_x = random.uniform(-CAMERA_SHIFT_MAX, CAMERA_SHIFT_MAX)
            shift_y = random.uniform(-CAMERA_SHIFT_MAX, CAMERA_SHIFT_MAX)

            if not _set_collection_camera(
                target_fraction,
                objects,
                azimuth_override=attempt_azimuth,
                elevation_override=attempt_elevation,
                shift_x=shift_x,
                shift_y=shift_y,
            ):
                subassembly_log(
                    f"{stage_name} #{index}: retry {attempt}: camera "
                    f"az={attempt_azimuth:.1f} el={attempt_elevation:.1f}"
                )
                continue

            if not validate_camera_position():
                subassembly_log(
                    f"{stage_name} #{index}: retry {attempt}: camera position"
                )
                continue

            boxes = []

            for obj in objects:
                bbox, visible = visible_object_bbox(scene, cam, obj)

                if not visible or bbox is None:
                    continue

                x_c, y_c, w, h = bbox

                if w <= 0.0 or h <= 0.0:
                    continue

                min_px = MIN_BOX_PIXELS.get(view_name, 8)
                if w * RES_X < min_px or h * RES_Y < min_px:
                    continue

                class_name = obj["dataset_class"]
                class_id = CLASS_ID[class_name]

                boxes.append([
                    class_id,
                    round(x_c, 6),
                    round(y_c, 6),
                    round(w, 6),
                    round(h, 6),
                ])

            if not boxes:
                subassembly_log(
                    f"{stage_name} #{index}: retry {attempt}: no valid boxes"
                )
                continue

            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)

            if image_path.exists():
                image_path.unlink()
            if label_path.exists():
                label_path.unlink()

            if not render_and_verify(scene, image_path):
                subassembly_log(
                    f"{stage_name} #{index}: retry {attempt}: render"
                )
                continue

            label_path.write_text(
                "\n".join(
                    f"{b[0]} {b[1]:.6f} {b[2]:.6f} "
                    f"{b[3]:.6f} {b[4]:.6f}"
                    for b in boxes
                ) + "\n",
                encoding="utf-8",
            )

            append_collection_subassembly_ndjson(
                ndjson_path,
                image_path,
                boxes,
                split,
                stage_name,
            )

            subassembly_log(
                f"OK {stage_name} #{index:03d} "
                f"view={view_name} target={target_fraction:.3f} "
                f"cam_az={attempt_azimuth:.1f} cam_el={attempt_elevation:.1f} "
                f"shift=({shift_x:.3f},{shift_y:.3f}) "
                f"obj_yaw={pose_meta['yaw_deg']:.1f} "
                f"obj_tilt=({pose_meta['tilt_x_deg']:.1f},{pose_meta['tilt_y_deg']:.1f}) "
                f"boxes={len(boxes)} split={split}"
            )
            return True

        except Exception as exc:
            subassembly_log(
                f"ERROR {stage_name} #{index} attempt {attempt}: "
                f"{type(exc).__name__}: {exc}"
            )

    subassembly_log(
        f"FAILED {stage_name} #{index} after "
        f"{MAX_RENDER_ATTEMPTS_PER_IMAGE} attempts"
    )
    return False


def generate_collection_subassembly_dataset():
    validate_stage_collections()
    audit_all_stage_geometry()

    images_per_stage = (
        SMOKE_IMAGES_PER_STAGE if SMOKE_TEST else IMAGES_PER_STAGE
    )

    clean_subassembly_output_for_production()

    images_dir = SUBASSEMBLY_OUTPUT_DIR / "images"
    labels_dir = SUBASSEMBLY_OUTPUT_DIR / "labels"

    for split in ("train", "val", "test"):
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    ndjson_path = initialize_collection_subassembly_ndjson()

    (SUBASSEMBLY_OUTPUT_DIR / "classes.txt").write_text(
        "\n".join(CLASS_NAMES) + "\n",
        encoding="utf-8",
    )
    write_subassembly_data_yaml()
    write_assembly_sequence_metadata()

    visibility_backup = capture_visibility_states()

    hide_all_stage_objects()
    hide_all_targets()
    hide_all_non_stage_meshes([])

    total_ok = 0
    total_fail = 0

    try:
        subassembly_log("=" * 70)
        subassembly_log("FINAL REALISTIC SUBASSEMBLY DATASET GENERATION")
        subassembly_log(
            "Stage_01 = AUTHORITATIVE CAD-derived body_a + bushing"
        )
        subassembly_log(
            "Stage_02-07 = existing manual Blender collections, UNCHANGED"
        )
        subassembly_log(
            f"images/stage={images_per_stage} SMOKE_TEST={SMOKE_TEST}"
        )
        subassembly_log(
            "Production camera distribution: 45% CLOSE / 40% MEDIUM / 15% FAR"
        )
        subassembly_log(
            "Production camera coverage: 24 azimuth sectors x 5 elevation bands"
        )
        subassembly_log(
            "Lighting: randomized energy, colour temperature and position"
        )
        subassembly_log(
            "Backgrounds: random selection from Backgrounds directory"
        )
        subassembly_log(
            "Independent object pose: random 0-360 yaw + ±10° tilt + XY jitter"
        )
        subassembly_log(
            "Camera: full 360° azimuth + 5 elevation bands + image-plane shift"
        )
        subassembly_log("=" * 70)

        for stage_name in STAGE_COLLECTIONS:
            stage01_objects = None
            stage01_collection = None

            try:
                if stage_name == "Stage_01":
                    # IMPORTANT: do not use the existing Stage_01 collection.
                    objects, stage01_collection, base_matrices = (
                        _prepare_authoritative_stage01_for_generation()
                    )
                    stage01_objects = objects

                    hide_all_stage_objects()
                    hide_all_targets()
                    hide_all_non_stage_meshes(objects)

                    for obj in objects:
                        obj.hide_render = False
                        obj.hide_viewport = False

                    if not validate_only_stage_meshes_visible(
                        objects, stage_name
                    ):
                        raise RuntimeError(
                            "Stage_01 authoritative scene isolation failed."
                        )

                else:
                    # Existing manual collections: NO GEOMETRY OR MATERIAL CHANGES.
                    objects = show_only_stage(stage_name)
                    base_matrices = {
                        obj.name: obj.matrix_world.copy()
                        for obj in objects
                    }

                stage_ok = 0

                for index in range(1, images_per_stage + 1):
                    ok = render_collection_stage_one(
                        stage_name,
                        objects,
                        base_matrices,
                        index,
                        images_dir,
                        labels_dir,
                        ndjson_path,
                    )

                    if ok:
                        stage_ok += 1
                        total_ok += 1
                    else:
                        total_fail += 1

                if stage_ok == 0:
                    raise RuntimeError(
                        f"FATAL: {stage_name} generated ZERO images."
                    )

                subassembly_log(
                    f"FINAL {stage_name}: {stage_ok}/{images_per_stage}"
                )

            finally:
                if stage_name == "Stage_01":
                    _cleanup_stage01_authoritative(
                        stage01_objects,
                        stage01_collection,
                    )

                # Guaranteed cleanup between stages.
                hide_all_stage_objects()
                hide_all_targets()
                hide_all_non_stage_meshes([])

    finally:
        restore_visibility_states(visibility_backup)

    subassembly_log("=" * 70)
    subassembly_log("FINAL REALISTIC SUBASSEMBLY DATASET COMPLETE")
    subassembly_log(f"TOTAL VALID IMAGES: {total_ok}")
    subassembly_log(f"TOTAL FAILED: {total_fail}")
    subassembly_log(f"Images: {images_dir}")

    expected_total = len(STAGE_COLLECTIONS) * images_per_stage
    if total_ok != expected_total:
        raise RuntimeError(
            f"SUBASSEMBLY DATASET INCOMPLETE: generated "
            f"{total_ok}/{expected_total} images ({total_fail} failed). "
            "Do not train the final model until every stage has the "
            "requested number of images."
        )
    subassembly_log(f"Labels: {labels_dir}")
    subassembly_log(f"NDJSON: {ndjson_path}")
    subassembly_log("=" * 70)




def final_production_dataset_audit():
    """
    Verify the completed production dataset on disk.

    Expected:
      200 single-part images per class.
      200 subassembly images per stage.
    Also verifies train/val/test counts according to the generator split.
    """
    failures = []

    # Single-part dataset.
    expected_part = IMAGES_PER_PART
    for class_name in CLASS_NAMES:
        for split in ("train", "val", "test"):
            folder = IMAGES_DIR / split
            count = len(list(folder.glob(f"{class_name}_*.png")))
            expected = (
                int(expected_part * SPLIT_TRAIN)
                if split == "train"
                else int(expected_part * SPLIT_VAL)
                if split == "val"
                else expected_part
                     - int(expected_part * SPLIT_TRAIN)
                     - int(expected_part * SPLIT_VAL)
            )
            if count != expected:
                failures.append(
                    f"single-part {class_name}/{split}: {count} != {expected}"
                )

    # Subassembly dataset.
    expected_stage = IMAGES_PER_STAGE
    for stage_name in STAGE_COLLECTIONS:
        stem = stage_name.lower()
        for split in ("train", "val", "test"):
            folder = SUBASSEMBLY_OUTPUT_DIR / "images" / split
            count = len(list(folder.glob(f"{stem}_*.png")))
            expected = (
                int(expected_stage * SPLIT_TRAIN)
                if split == "train"
                else int(expected_stage * SPLIT_VAL)
                if split == "val"
                else expected_stage
                     - int(expected_stage * SPLIT_TRAIN)
                     - int(expected_stage * SPLIT_VAL)
            )
            if count != expected:
                failures.append(
                    f"subassembly {stage_name}/{split}: {count} != {expected}"
                )

    if failures:
        raise RuntimeError(
            "FINAL DATASET AUDIT FAILED:\n  - "
            + "\n  - ".join(failures)
        )

    log("=" * 70)
    log("FINAL DATASET AUDIT PASSED")
    log("Single-part: 7 classes × 200 = 1,400 images")
    log("Subassembly: 7 stages × 200 = 1,400 images")
    log("TOTAL: 2,800 images")
    log("Train/Val/Test split verified on disk.")
    log("=" * 70)




# ============================================================
# V18 ADD-ON ENGINE
# ============================================================

def addon_log(msg):
    print(f"[dataset] [V18] {msg}")


# ------------------------------------------------------------
# Orientation and placement
# ------------------------------------------------------------

def random_rotation_matrix():
    """Uniformly random 3-D rotation (Shoemake). Every face of the object,
    including the bottom, is equally likely to point at the camera."""
    u1, u2, u3 = random.random(), random.random(), random.random()
    a = math.sqrt(1.0 - u1)
    b = math.sqrt(u1)
    q = mathutils.Quaternion((
        b * math.cos(2.0 * math.pi * u3),
        a * math.sin(2.0 * math.pi * u2),
        a * math.cos(2.0 * math.pi * u2),
        b * math.sin(2.0 * math.pi * u3),
    ))
    return q.to_matrix().to_4x4()


def _rotate_rigidly_about_center(objects, rot):
    center = group_bbox_center(objects)
    for obj in objects:
        mw = obj.matrix_world.copy()
        mw.translation -= center
        mw = rot @ mw
        mw.translation += center
        obj.matrix_world = mw
    bpy.context.view_layer.update()


def _translate(objects, delta):
    for obj in objects:
        mw = obj.matrix_world.copy()
        mw.translation += delta
        obj.matrix_world = mw
    bpy.context.view_layer.update()


def place_objects_full_rotation(objects, allow_lift=True, xy_jitter=None):
    """Random 3-D orientation, then rest the REAL geometry on the plate
    (or lift it as if held). Returns the lift in metres."""
    _rotate_rigidly_about_center(objects, random_rotation_matrix())

    low, _ = _vertex_z_limits(objects)
    if low is None:
        return 0.0

    plate_z = get_plate_top_z()
    lift = 0.0
    if allow_lift and random.random() < ADDON_LIFT_PROB:
        lift = random.uniform(0.005, ADDON_MAX_LIFT)

    j = COLLECTION_XY_JITTER if xy_jitter is None else xy_jitter
    _translate(objects, Vector((random.uniform(-j, j),
                                random.uniform(-j, j),
                                plate_z - low + lift)))
    return lift


def objects_rest_on_or_above_plate(objects):
    low, high = _vertex_z_limits(objects)
    if low is None:
        return False
    plate_z = get_plate_top_z()
    return low >= plate_z - 0.002 and high > plate_z


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

def _kelvin_to_rgb(kelvin):
    t = kelvin / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
    else:
        r = 329.698727446 * ((t - 60) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60) ** -0.0755148492)
        b = 255.0
    clamp = lambda x: max(0.0, min(255.0, x)) / 255.0
    return (clamp(r), clamp(g), clamp(b))


def _get_fill_light():
    obj = bpy.data.objects.get(FILL_LIGHT_NAME)
    if obj is not None:
        return obj
    data = bpy.data.lights.new(FILL_LIGHT_NAME, type="AREA")
    obj = bpy.data.objects.new(FILL_LIGHT_NAME, data)
    bpy.context.scene.collection.objects.link(obj)
    return obj


_ENV_DEFAULTS = {}


def _remember_environment():
    scene = bpy.context.scene
    _ENV_DEFAULTS["exposure"] = scene.view_settings.exposure
    bg = scene.world.node_tree.nodes.get("Background") if scene.world else None
    if bg is not None:
        _ENV_DEFAULTS["bg_color"] = tuple(bg.inputs["Color"].default_value)
        _ENV_DEFAULTS["bg_strength"] = bg.inputs["Strength"].default_value


def _restore_environment():
    scene = bpy.context.scene
    if "exposure" in _ENV_DEFAULTS:
        scene.view_settings.exposure = _ENV_DEFAULTS["exposure"]
    bg = scene.world.node_tree.nodes.get("Background") if scene.world else None
    if bg is not None and "bg_color" in _ENV_DEFAULTS:
        bg.inputs["Color"].default_value = _ENV_DEFAULTS["bg_color"]
        bg.inputs["Strength"].default_value = _ENV_DEFAULTS["bg_strength"]
    fill = bpy.data.objects.get(FILL_LIGHT_NAME)
    if fill is not None:
        data = fill.data
        bpy.data.objects.remove(fill, do_unlink=True)
        if data is not None and data.users == 0:
            bpy.data.lights.remove(data)

    # Remove the occluder completely so it cannot end up saved in the .blend.
    occ = bpy.data.objects.get(OCCLUDER_NAME)
    if occ is not None:
        mesh = occ.data
        mats = list(mesh.materials) if mesh is not None else []
        bpy.data.objects.remove(occ, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        for mat in mats:
            if mat is not None and mat.users == 0:
                bpy.data.materials.remove(mat)


def randomize_environment(center):
    """Key light, optional fill light, world brightness and exposure."""
    randomize_light(center)
    key = get_object(LIGHT_NAME)
    if key is not None:
        key.data.energy = random.uniform(ADDON_LIGHT_ENERGY_MIN,
                                         ADDON_LIGHT_ENERGY_MAX)
        key.data.color = _kelvin_to_rgb(random.uniform(
            ADDON_LIGHT_KELVIN_MIN, ADDON_LIGHT_KELVIN_MAX))
        if hasattr(key.data, "shadow_soft_size"):
            key.data.shadow_soft_size = random.uniform(0.01, 0.35)

    fill = _get_fill_light()
    if random.random() < ADDON_FILL_LIGHT_PROB:
        ang = random.uniform(0.0, 2.0 * math.pi)
        rad = random.uniform(0.3, 0.8)
        pos = Vector((center.x + rad * math.cos(ang),
                      center.y + rad * math.sin(ang),
                      get_plate_top_z() + random.uniform(0.25, 0.9)))
        fill.location = pos
        fill.rotation_euler = (center - pos).to_track_quat("-Z", "Y").to_euler()
        fill.data.energy = random.uniform(15.0, 150.0)
        fill.data.size = random.uniform(0.3, 1.2)
        fill.data.color = _kelvin_to_rgb(random.uniform(4000.0, 6500.0))
        fill.hide_render = False
    else:
        fill.hide_render = True

    scene = bpy.context.scene
    bg = scene.world.node_tree.nodes.get("Background") if scene.world else None
    if bg is not None:
        g = random.uniform(0.25, 0.5)
        bg.inputs["Color"].default_value = (
            g * random.uniform(0.9, 1.1), g, g * random.uniform(0.9, 1.15), 1.0)
        bg.inputs["Strength"].default_value = random.uniform(
            ADDON_WORLD_STRENGTH_MIN, ADDON_WORLD_STRENGTH_MAX)

    scene.view_settings.exposure = random.uniform(-ADDON_EXPOSURE_JITTER,
                                                  ADDON_EXPOSURE_JITTER)


def place_random_occluder(near_obj):
    """Hand / glove / tool coloured block partly covering a part."""
    occ = place_occluder_over(near_obj)
    tones = [
        (0.80, 0.60, 0.48), (0.62, 0.43, 0.32), (0.45, 0.30, 0.22),
        (0.15, 0.25, 0.55), (0.10, 0.10, 0.10), (0.55, 0.55, 0.58),
    ]
    if occ.data.materials:
        mat = occ.data.materials[0]
        bsdf = mat.node_tree.nodes.get("Principled BSDF") if mat.use_nodes else None
        if bsdf is not None:
            r, g, b = random.choice(tones)
            bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    occ.scale = tuple(v * random.uniform(0.8, 1.6) for v in occ.scale)
    bpy.context.view_layer.update()
    return occ


# ------------------------------------------------------------
# Camera
# ------------------------------------------------------------

def _addon_elevation(index, view):
    bands = ADDON_ELEVATION_BANDS_DEG
    el = bands[(index - 1) % len(bands)] + random.uniform(
        -ADDON_ELEVATION_JITTER_DEG, ADDON_ELEVATION_JITTER_DEG)
    if view == "SMALL":
        # True-scale views mimic the real overhead camera.
        el = random.uniform(58.0, 86.0)
    return max(15.0, min(86.0, el))


def _deploy_fraction(objects):
    """Share of the frame the object occupies in the real camera."""
    corners = _group_bbox_corners(objects)
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    size = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    return size / DEPLOY_FOV_WIDTH_M


def _addon_target_fraction(view, objects):
    if view == "SMALL":
        return max(0.02, min(0.20, _deploy_fraction(objects)
                                   * random.uniform(0.9, 1.6)))
    if view == "CLOSE":
        return random.uniform(0.55, 0.78)
    if view == "MEDIUM":
        return random.uniform(0.32, 0.52)
    return random.uniform(0.14, 0.28)


def _addon_single_view():
    r = random.random()
    acc = 0.0
    for view, w in ADDON_SINGLE_VIEW_WEIGHTS.items():
        acc += w
        if r <= acc:
            return view
    return "MEDIUM"


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

def _addon_split(index, total):
    train_end = max(1, int(total * SPLIT_TRAIN))
    val_end = max(train_end + 1, int(total * (SPLIT_TRAIN + SPLIT_VAL)))
    if index <= train_end:
        return "train"
    if index <= val_end:
        return "val"
    return "test"


def _addon_clean_folder(root):
    """Delete previous ADD-ON output (files named *_rot_*) in this add-on folder
    only. Needed because a smoke run puts e.g. nut_rot_0004 in test/, while the
    full run puts nut_rot_0004 in train/ - the old one would stay behind in
    test/. Never touches the original V17 dataset folders."""
    removed = 0
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            folder = root / kind / split
            if not folder.exists():
                continue
            for f in folder.glob("*_rot_*"):
                if f.suffix.lower() in (".png", ".txt"):
                    f.unlink()
                    removed += 1
    if removed:
        addon_log(f"Removed {removed} previous add-on files from {root.name}")


def _addon_init_folder(root, mode):
    if not ADDON_SMOKE_TEST:
        _addon_clean_folder(root)
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n",
                                      encoding="utf-8")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"test: images/test\nnc: {len(CLASS_NAMES)}\nnames:\n{names}\n",
        encoding="utf-8")
    ndjson = root / "annotations.ndjson"
    ndjson.write_text(json.dumps({
        "type": "dataset", "task": "object_detection",
        "classes": CLASS_NAMES, "dataset_mode": mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }) + "\n", encoding="utf-8")
    return ndjson


def _write_label(label_path, boxes):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        "\n".join(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"
                  for b in boxes) + "\n", encoding="utf-8")


def _append_ndjson(path, image_path, boxes, split, extra):
    rec = {"file": str(image_path), "type": "image",
           "width": RES_X, "height": RES_Y,
           "annotations": {"boxes": boxes} if boxes else {},
           "split": split}
    rec.update(extra)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _collect_boxes(scene, cam, objects, view):
    """Visibility-aware boxes for every object. Returns (boxes, info) where
    info is a list of (class_name, visible_fraction, labelled)."""
    boxes, info = [], []
    min_px = MIN_BOX_PIXELS.get(view, 8)
    for obj in objects:
        cls = obj.get("dataset_class") or obj.get("_v18_class")
        if cls is None:
            continue
        bbox, frac, count = visibility_analysis(scene, cam, obj)
        labelled = False
        if bbox is not None and count >= MIN_VISIBLE_POINTS_LABEL:
            x_c, y_c, w, h = bbox
            if w * RES_X >= min_px and h * RES_Y >= min_px:
                boxes.append([CLASS_ID[cls], round(x_c, 6), round(y_c, 6),
                              round(w, 6), round(h, 6)])
                labelled = True
        info.append((cls, frac, labelled))
    return boxes, info


def _key_fraction(cls):
    return ADDON_KEY_PART_MIN_FRACTION.get(cls, ADDON_KEY_PART_DEFAULT_FRACTION)


def _requirement_met(info, required, relaxed=False):
    """relaxed=False: every required part reaches its visibility target.
    relaxed=True : every required part is at least clearly labelled."""
    for cls, need in required.items():
        thr = 0.0 if relaxed else _key_fraction(cls)
        n = sum(1 for c, f, lab in info if c == cls and lab and f >= thr)
        if n < need:
            return False
    return True


# ------------------------------------------------------------
# Single parts
# ------------------------------------------------------------

def _addon_camera_single(obj, target_fraction, elevation_deg):
    scene = bpy.context.scene
    cam = get_object(CAMERA_NAME)
    if cam is None:
        return False
    cam.data.shift_x = random.uniform(-ADDON_CAMERA_SHIFT_MAX, ADDON_CAMERA_SHIFT_MAX)
    cam.data.shift_y = random.uniform(-ADDON_CAMERA_SHIFT_MAX, ADDON_CAMERA_SHIFT_MAX)

    center = get_world_bbox_center(obj)
    azimuth = random.uniform(0.0, 2.0 * math.pi)
    lo, hi = CAM_DIST_SEARCH_MIN, CAM_DIST_SEARCH_MAX

    for _ in range(6):
        place_camera_at_point(center, hi, azimuth, elevation_deg, center)
        frac = compute_bbox_fraction(scene, cam, obj)
        if frac is not None and frac <= target_fraction:
            break
        hi *= 1.5

    for _ in range(18):
        mid = (lo + hi) / 2.0
        place_camera_at_point(center, mid, azimuth, elevation_deg, center)
        frac = compute_bbox_fraction(scene, cam, obj)
        if frac is None or frac > target_fraction:
            lo = mid
        else:
            hi = mid

    place_camera_at_point(center, hi, azimuth, elevation_deg, center)
    if not bbox_fits_in_frame(scene, cam, obj):
        # V17 moved the camera CLOSER here, making the fit worse.
        place_camera_at_point(center, hi * 1.10, azimuth, elevation_deg, center)
    return bbox_fits_in_frame(scene, cam, obj)


def _addon_render_single(class_name, obj, base_matrix, index, total,
                         root, ndjson, stats):
    scene = bpy.context.scene
    cam = get_object(CAMERA_NAME)
    split = _addon_split(index, total)
    stem = f"{class_name}_rot_{index:04d}"
    image_path = root / "images" / split / f"{stem}.png"
    label_path = root / "labels" / split / f"{stem}.txt"

    for attempt in range(1, ADDON_MAX_ATTEMPTS + 1):
        hide_occluder()
        # Always start from the same base pose, otherwise the part would
        # random-walk across the table image after image.
        obj.matrix_world = base_matrix.copy()
        bpy.context.view_layer.update()
        place_objects_full_rotation([obj], allow_lift=True,
                                    xy_jitter=max(get_object_size(obj), 0.05) * 0.25)
        if not objects_rest_on_or_above_plate([obj]):
            continue

        view = _addon_single_view()
        target = _addon_target_fraction(view, [obj])
        elevation = _addon_elevation(index + attempt, view)
        if not _addon_camera_single(obj, target, elevation):
            continue
        if not validate_camera_position():
            continue

        occluded = random.random() < ADDON_SINGLE_OCCLUSION_PROB
        if occluded:
            place_random_occluder(obj)

        boxes, info = _collect_boxes(scene, cam, [obj], view)
        if not boxes:
            continue
        frac = info[0][1]
        # Unoccluded single parts must be (almost) completely in view;
        # occluded ones must still show a clear majority of the part.
        if frac < (0.45 if occluded else 0.85):
            continue

        randomize_environment(get_world_bbox_center(obj))
        set_random_background()

        if image_path.exists():
            image_path.unlink()
        if not render_and_verify(scene, image_path):
            continue

        _write_label(label_path, boxes)
        _append_ndjson(ndjson, image_path, boxes, split,
                       {"class": class_name, "view": view,
                        "occluded": occluded,
                        "visible_fraction": round(frac, 3)})
        stats["ok"] += 1
        stats["occluded"] += int(occluded)
        hide_occluder()
        addon_log(f"OK {class_name} rot#{index:03d} view={view} "
                  f"vis={frac:.2f} occ={occluded} split={split} try={attempt}")
        return True

    hide_occluder()
    addon_log(f"FAILED {class_name} rot#{index} after {ADDON_MAX_ATTEMPTS} attempts")
    return False


def generate_addon_single_part_dataset(resolved_sources):
    random.seed(ADDON_SEED)
    root = ADDON_SINGLE_OUTPUT_DIR
    ndjson = _addon_init_folder(root, "addon_single_full_rotation")

    addon_log("=" * 70)
    addon_log(f"SINGLE-PART ADD-ON -> {root}")
    addon_log("=" * 70)

    hide_all_stage_objects()
    hide_all_targets()
    summary = []

    for class_name in CLASS_NAMES:
        total = (ADDON_SMOKE_IMAGES_PER_CLASS if ADDON_SMOKE_TEST
                 else ADDON_IMAGES_PER_CLASS[class_name])
        source = bpy.data.objects.get(resolved_sources[class_name])
        if source is None:
            raise RuntimeError(f"Missing source object for {class_name}")

        unparent_keep_world_transform(source)
        source_matrix = source.matrix_world.copy()

        if class_name == "threaded_pin":
            obj = make_clean_render_copy(source)
            source.hide_render = True        # V17 left it visible twice
            source.hide_viewport = True
        else:
            obj = source

        obj["_v18_class"] = class_name
        hide_all_non_stage_meshes([obj])
        obj.hide_render = False
        obj.hide_viewport = False

        stats = {"ok": 0, "occluded": 0}
        try:
            base_matrix = obj.matrix_world.copy()
            failed = []
            for index in range(1, total + 1):
                if not _addon_render_single(class_name, obj, base_matrix,
                                            index, total, root, ndjson, stats):
                    failed.append(index)
            for rnd in range(1, ADDON_RETRY_ROUNDS + 1):
                if not failed:
                    break
                addon_log(f"RETRY round {rnd}: {class_name} {len(failed)} image(s)")
                failed = [i for i in failed
                          if not _addon_render_single(class_name, obj, base_matrix,
                                                      i, total, root, ndjson, stats)]
        finally:
            hide_occluder()
            if obj is not source:
                remove_clean_render_copy(obj)
            else:
                if "_v18_class" in source.keys():
                    del source["_v18_class"]
            source.matrix_world = source_matrix
            bpy.context.view_layer.update()
            hide_all_targets()

        ADDON_RESULTS["single"][class_name] = (stats["ok"], total)
        line = f"{class_name}: {stats['ok']}/{total} (occluded {stats['occluded']})"
        summary.append(line)
        addon_log("FINAL " + line)

    return summary


# ------------------------------------------------------------
# Subassemblies
# ------------------------------------------------------------

def _addon_render_stage(stage_name, objects, base_matrices, index, total,
                        root, ndjson, stats):
    scene = bpy.context.scene
    cam = get_object(CAMERA_NAME)
    split = _addon_split(index, total)
    stem = f"{stage_name.lower()}_rot_{index:04d}"
    image_path = root / "images" / split / f"{stem}.png"
    label_path = root / "labels" / split / f"{stem}.txt"

    view = ADDON_STAGE_VIEW_CYCLE[(index - 1) % len(ADDON_STAGE_VIEW_CYCLE)]
    required = ADDON_REQUIRED_VISIBLE.get(stage_name, {})
    alternatives = ADDON_REQUIRED_ALTERNATIVES.get(stage_name)
    if alternatives:
        required = alternatives[(index - 1) % len(alternatives)]
    must_show = bool(required) and view != "SMALL" and (
        random.random() < ADDON_REQUIRE_PROB)
    base_az = ((index - 1) * 137.508) % 360.0

    for attempt in range(1, ADDON_MAX_ATTEMPTS + 1):
        hide_occluder()
        for obj in objects:
            obj.matrix_world = base_matrices[obj.name].copy()
        bpy.context.view_layer.update()

        lift = place_objects_full_rotation(objects, allow_lift=True)

        hide_all_non_stage_meshes(objects)
        if not validate_only_stage_meshes_visible(objects, stage_name):
            continue
        if not objects_rest_on_or_above_plate(objects):
            continue

        target = _addon_target_fraction(view, objects)
        az = (base_az + random.uniform(-25.0, 25.0) * min(attempt, 4)) % 360.0
        el = _addon_elevation(index + attempt - 1, view)
        if not _set_collection_camera(
            target, objects, azimuth_override=az, elevation_override=el,
            shift_x=random.uniform(-ADDON_CAMERA_SHIFT_MAX, ADDON_CAMERA_SHIFT_MAX),
            shift_y=random.uniform(-ADDON_CAMERA_SHIFT_MAX, ADDON_CAMERA_SHIFT_MAX),
        ):
            continue
        if not validate_camera_position():
            continue

        occluded = (not must_show) and random.random() < ADDON_STAGE_OCCLUSION_PROB
        if occluded:
            place_random_occluder(random.choice(objects))

        boxes, info = _collect_boxes(scene, cam, objects, view)
        if not boxes:
            continue

        full_ok = False
        if must_show:
            relaxed = attempt > ADDON_RELAX_AFTER_ATTEMPTS
            if not _requirement_met(info, required, relaxed=relaxed):
                continue
            full_ok = _requirement_met(info, required, relaxed=False)

        # Pose accepted: only now spend time on lighting and rendering.
        randomize_environment(group_bbox_center(objects))
        set_random_background()

        if image_path.exists():
            image_path.unlink()
        if not render_and_verify(scene, image_path):
            continue

        _write_label(label_path, boxes)
        _append_ndjson(ndjson, image_path, boxes, split, {
            "stage": stage_name, "stage_id": int(stage_name.split("_")[1]),
            "view": view, "must_show": must_show, "occluded": occluded,
            "lift_m": round(lift, 4),
            "visible": {c: round(f, 3) for c, f, lab in info if lab},
        })

        stats["ok"] += 1
        stats["must_show"] += int(must_show)
        stats["full"] += int(full_ok)
        for cls in required:
            if any(c == cls and lab for c, f, lab in info):
                stats["labelled"][cls] = stats["labelled"].get(cls, 0) + 1

        hide_occluder()
        addon_log(f"OK {stage_name} rot#{index:03d} view={view} "
                  f"must_show={must_show} full={full_ok} lift={lift*100:.1f}cm "
                  f"boxes={len(boxes)} split={split} try={attempt}")
        return True

    hide_occluder()
    addon_log(f"FAILED {stage_name} rot#{index} after {ADDON_MAX_ATTEMPTS} attempts")
    return False


def purge_helpers_from_stage_collections():
    """Move any helper object that ended up inside a Stage_XX collection
    (e.g. from an earlier run) back to the scene root."""
    moved = []
    for stage_name in STAGE_COLLECTIONS:
        coll = bpy.data.collections.get(stage_name)
        if coll is None:
            continue
        for obj in list(coll.all_objects):
            if obj is not None and is_helper_object(obj):
                move_to_scene_root(obj)
                moved.append(f"{obj.name} (was in {stage_name})")
    for m in moved:
        addon_log(f"Moved helper out of stage collection: {m}")


def generate_addon_subassembly_dataset():
    random.seed(ADDON_SEED + 1)
    root = ADDON_SUBASSEMBLY_OUTPUT_DIR

    _LOCAL_VERT_CACHE.clear()
    purge_helpers_from_stage_collections()
    validate_stage_collections()
    audit_all_stage_geometry()
    ndjson = _addon_init_folder(root, "addon_subassembly_full_rotation")

    addon_log("=" * 70)
    addon_log(f"SUBASSEMBLY ADD-ON -> {root}")
    addon_log("=" * 70)

    visibility_backup = capture_visibility_states()
    hide_all_stage_objects()
    hide_all_targets()
    hide_all_non_stage_meshes([])
    summary = []

    try:
        for stage_name in STAGE_COLLECTIONS:
            total = (ADDON_SMOKE_IMAGES_PER_STAGE if ADDON_SMOKE_TEST
                     else ADDON_IMAGES_PER_STAGE[stage_name])
            s01_objects = s01_collection = None
            base_matrices = {}
            objects = []
            stats = {"ok": 0, "must_show": 0, "full": 0, "labelled": {}}

            try:
                if stage_name == "Stage_01":
                    objects, s01_collection, base_matrices = (
                        _prepare_authoritative_stage01_for_generation())
                    s01_objects = objects
                    hide_all_stage_objects()
                    hide_all_targets()
                    hide_all_non_stage_meshes(objects)
                    for obj in objects:
                        obj.hide_render = False
                        obj.hide_viewport = False
                else:
                    objects = show_only_stage(stage_name)
                    base_matrices = {o.name: o.matrix_world.copy()
                                     for o in objects}

                failed = []
                for index in range(1, total + 1):
                    if not _addon_render_stage(stage_name, objects, base_matrices,
                                               index, total, root, ndjson, stats):
                        failed.append(index)
                for rnd in range(1, ADDON_RETRY_ROUNDS + 1):
                    if not failed:
                        break
                    addon_log(f"RETRY round {rnd}: {stage_name} {len(failed)} image(s)")
                    failed = [i for i in failed
                              if not _addon_render_stage(stage_name, objects,
                                                         base_matrices, i, total,
                                                         root, ndjson, stats)]
            finally:
                hide_occluder()
                # Put the manual collection back exactly where it was.
                # (V17 left every stage in its last random pose.)
                for obj in objects:
                    if obj.name in base_matrices:
                        obj.matrix_world = base_matrices[obj.name].copy()
                bpy.context.view_layer.update()
                if stage_name == "Stage_01":
                    _cleanup_stage01_authoritative(s01_objects, s01_collection)
                hide_all_stage_objects()
                hide_all_targets()
                hide_all_non_stage_meshes([])
                _LOCAL_VERT_CACHE.clear()

            req = ADDON_REQUIRED_VISIBLE.get(stage_name, {})
            detail = ", ".join(
                f"{c} labelled in {stats['labelled'].get(c, 0)}/{stats['ok']}"
                for c in req)
            ADDON_RESULTS["stage"][stage_name] = (stats["ok"], total)
            line = (f"{stage_name}: {stats['ok']}/{total} images | "
                    f"must-show {stats['must_show']} (fully visible "
                    f"{stats['full']}) | {detail}")
            summary.append(line)
            addon_log("FINAL " + line)
    finally:
        restore_visibility_states(visibility_backup)

    return summary


# ------------------------------------------------------------
# Audit and combined training config
# ------------------------------------------------------------

def addon_audit(root):
    """Count images per split and boxes per class from the label files."""
    if not root.exists():
        return
    addon_log(f"AUDIT {root.name}")
    per_class = {c: 0 for c in CLASS_NAMES}
    for split in ("train", "val", "test"):
        imgs = list((root / "images" / split).glob("*.png"))
        labels = list((root / "labels" / split).glob("*.txt"))
        addon_log(f"  {split:<5}: {len(imgs):>5} images, {len(labels):>5} labels")
        for lp in labels:
            for line in lp.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts:
                    per_class[CLASS_NAMES[int(parts[0])]] += 1
    for c in CLASS_NAMES:
        addon_log(f"  boxes {c:<13}: {per_class[c]}")


def write_combined_data_yaml():
    """One training config listing the original data and the add-ons.
    Only folders that exist on disk are included."""
    candidates = [
        OUTPUT_DIR,
        SUBASSEMBLY_OUTPUT_DIR,
        ADDON_SINGLE_OUTPUT_DIR,
        ADDON_SUBASSEMBLY_OUTPUT_DIR,
        VARIANT_OUTPUT_DIR,
    ]
    lines = []
    for split in ("train", "val", "test"):
        dirs = [(c / "images" / split).as_posix() for c in candidates
                if (c / "images" / split).exists()
                and any((c / "images" / split).glob("*.png"))]
        lines.append(f"{split}:")
        lines.extend(f"  - {d}" for d in dirs)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    COMBINED_DATA_YAML_PATH.write_text(
        "\n".join(lines) + f"\nnc: {len(CLASS_NAMES)}\nnames:\n{names}\n",
        encoding="utf-8")
    addon_log(f"Wrote combined training config: {COMBINED_DATA_YAML_PATH}")


ADDON_RESULTS = {"single": {}, "stage": {}}


def addon_final_verdict():
    """Clear COMPLETE / INCOMPLETE verdict + generation_report.json.
    Also counts the files actually on disk, not only the in-run counters."""
    problems = []
    report = {"single": {}, "stage": {}}

    for kind, root, stem_of in (
        ("single", ADDON_SINGLE_OUTPUT_DIR, lambda k: f"{k}_rot_"),
        ("stage", ADDON_SUBASSEMBLY_OUTPUT_DIR, lambda k: f"{k.lower()}_rot_"),
    ):
        for key, (ok, total) in ADDON_RESULTS[kind].items():
            on_disk = sum(
                len(list((root / "images" / sp).glob(stem_of(key) + "*.png")))
                for sp in ("train", "val", "test"))
            labels = sum(
                len(list((root / "labels" / sp).glob(stem_of(key) + "*.txt")))
                for sp in ("train", "val", "test"))
            report[kind][key] = {"requested": total, "generated": ok,
                                 "images_on_disk": on_disk,
                                 "labels_on_disk": labels}
            if ok != total or on_disk != total or labels != total:
                problems.append(f"{key}: requested {total}, generated {ok}, "
                                f"{on_disk} images / {labels} labels on disk")

    report["complete"] = not problems
    report["problems"] = problems
    (OUTPUT_DIR / "generation_report_v18.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    log("=" * 70)
    if problems:
        log("RESULT: INCOMPLETE  -  DO NOT TRAIN YET")
        for p in problems:
            log("   " + p)
        log("Re-run the generator (it retries automatically), or send this output.")
    else:
        n_single = sum(v[1] for v in ADDON_RESULTS["single"].values())
        n_stage = sum(v[1] for v in ADDON_RESULTS["stage"].values())
        log(f"RESULT: COMPLETE  -  single {n_single}/{n_single}, "
            f"subassembly {n_stage}/{n_stage}, every image has a label.")
        log("You can go to training now.")
    log(f"Report: {OUTPUT_DIR / 'generation_report_v18.json'}")
    log("=" * 70)
    return not problems


# ============================================================
# V19: PART-COMBINATION (VARIANT) DATASET
# ------------------------------------------------------------
# Small-batch assembly has many valid orders, not one. YOLO still only
# detects the 7 PARTS - combinations are NOT extra classes. What the model
# needs is to see the assembly in every partial configuration, not only the
# 7 canonical stages, so the runtime can recognise an unknown-but-valid
# combination instead of guessing.
#
# Each image renders the Stage_07 assembly with the parts that are not in the
# combination hidden, so every combination is shown in its true assembled
# position. The combination of each image is written to the NDJSON and to
# variant_index.json, which the runtime reads to name states and to tell a
# known stage from an unknown variant.
# ============================================================

RUN_ADDON_VARIANTS = True
VARIANT_OUTPUT_DIR = OUTPUT_DIR / "variants_fullrot"
VARIANT_IMAGES_PER_COMBO = 40
VARIANT_SMOKE_IMAGES = 2
VARIANT_REQUIRE_PROB = 0.70      # share of images where every part must show
VARIANT_SOURCE_STAGE = "Stage_07"

# False = only physically sensible combinations (the nut needs the block_pin,
#         the bodies are joined after the bushings, ...)
# True  = every subset, including ones that cannot be built
VARIANT_ALLOW_ALL_COMBINATIONS = False

# The 7 canonical stages as component counts, so the index can mark which
# combinations are known stages and which are new variants.
VARIANT_CANONICAL = {
    (1, 0, 1, 0, 0, 0): "Stage_01",
    (1, 0, 2, 0, 0, 0): "Stage_02",
    (1, 1, 2, 0, 0, 0): "Stage_03",
    (1, 1, 2, 1, 0, 0): "Stage_04",
    (1, 1, 2, 1, 1, 0): "Stage_05",
    (0, 0, 0, 0, 0, 2): "Stage_06",
    (1, 1, 2, 1, 1, 2): "Stage_07",
}


def _addon_clean_folder_prefix(root, prefix):
    """Delete previous output of THIS pass only (files starting with prefix)."""
    removed = 0
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            folder = root / kind / split
            if not folder.exists():
                continue
            for f in folder.glob(prefix + "*"):
                if f.suffix.lower() in (".png", ".txt"):
                    f.unlink()
                    removed += 1
    if removed:
        addon_log(f"Removed {removed} previous variant files from {root.name}")


def variant_combinations():
    """(body_a, body_b, bushings, block_pin, nut, knob_sub), where knob_sub is
    0 = none, 1 = knob only, 2 = knob + threaded_pin."""
    combos = []
    for body_b in (0, 1):
        for bush in (0, 1, 2):
            for block in (0, 1):
                for nut in (0, 1):
                    for knobsub in (0, 1, 2):
                        body_a = 1 if (body_b or bush or block or nut) else 0
                        if not VARIANT_ALLOW_ALL_COMBINATIONS:
                            if nut and not block:
                                continue      # the nut secures the block_pin
                            if block and not body_b:
                                continue      # the block_pin joins both bodies
                            if body_b and bush == 0:
                                continue      # bushings go in before body_b
                        c = (body_a, body_b, bush, block, nut, knobsub)
                        if sum(c) == 0:
                            continue
                        combos.append(c)
    return sorted(set(combos))


def variant_id(c):
    return f"A{c[0]}-B{c[1]}-BU{c[2]}-BP{c[3]}-N{c[4]}-K{c[5]}"


def variant_components(c):
    comp = {}
    if c[0]:
        comp["body_a"] = 1
    if c[1]:
        comp["body_b"] = 1
    if c[2]:
        comp["bushing"] = c[2]
    if c[3]:
        comp["block_pin"] = 1
    if c[4]:
        comp["nut"] = 1
    if c[5] >= 1:
        comp["knob"] = 1
    if c[5] >= 2:
        comp["threaded_pin"] = 1
    return comp


def variant_select_objects(by_class, c):
    chosen = []
    for cls, count in variant_components(c).items():
        available = by_class.get(cls, [])
        if len(available) < count:
            raise RuntimeError(
                f"{VARIANT_SOURCE_STAGE} has {len(available)} x {cls}, needs {count}")
        chosen.extend(available[:count])
    return chosen


def _variant_render_one(combo, objects, base_matrices, index, total,
                        root, ndjson, stats):
    scene = bpy.context.scene
    cam = get_object(CAMERA_NAME)
    split = _addon_split(index, total)
    vid = variant_id(combo)
    stem = f"var_{vid}_{index:04d}"
    image_path = root / "images" / split / f"{stem}.png"
    label_path = root / "labels" / split / f"{stem}.txt"

    view = ADDON_STAGE_VIEW_CYCLE[(index - 1) % len(ADDON_STAGE_VIEW_CYCLE)]
    components = variant_components(combo)
    must_show = view != "SMALL" and random.random() < VARIANT_REQUIRE_PROB

    for attempt in range(1, ADDON_MAX_ATTEMPTS + 1):
        hide_occluder()
        for obj in objects:
            obj.matrix_world = base_matrices[obj.name].copy()
        bpy.context.view_layer.update()

        lift = place_objects_full_rotation(objects, allow_lift=True)
        hide_all_non_stage_meshes(objects)
        if not objects_rest_on_or_above_plate(objects):
            continue

        target = _addon_target_fraction(view, objects)
        az = (((index - 1) * 137.508)
              + random.uniform(-25.0, 25.0) * min(attempt, 4)) % 360.0
        el = _addon_elevation(index + attempt - 1, view)
        if not _set_collection_camera(
            target, objects, azimuth_override=az, elevation_override=el,
            shift_x=random.uniform(-ADDON_CAMERA_SHIFT_MAX, ADDON_CAMERA_SHIFT_MAX),
            shift_y=random.uniform(-ADDON_CAMERA_SHIFT_MAX, ADDON_CAMERA_SHIFT_MAX),
        ):
            continue
        if not validate_camera_position():
            continue

        boxes, info = _collect_boxes(scene, cam, objects, view)
        if not boxes:
            continue

        all_shown = all(
            sum(1 for c, f, lab in info if c == cls and lab) >= n
            for cls, n in components.items())
        if must_show and not all_shown and attempt <= ADDON_RELAX_AFTER_ATTEMPTS:
            continue

        if random.random() < ADDON_STAGE_OCCLUSION_PROB:
            place_random_occluder(random.choice(objects))

        randomize_environment(group_bbox_center(objects))
        set_random_background()
        if image_path.exists():
            image_path.unlink()
        if not render_and_verify(scene, image_path):
            continue

        _write_label(label_path, boxes)
        _append_ndjson(ndjson, image_path, boxes, split, {
            "variant_id": vid,
            "combination": components,
            "canonical_stage": VARIANT_CANONICAL.get(combo),
            "view": view,
            "lift_m": round(lift, 4),
            "all_parts_visible": bool(all_shown),
        })
        stats["ok"] += 1
        stats["all_visible"] += int(all_shown)
        hide_occluder()
        addon_log(f"OK {vid} #{index:03d} view={view} parts={len(objects)} "
                  f"boxes={len(boxes)} all_visible={all_shown} try={attempt}")
        return True

    hide_occluder()
    addon_log(f"FAILED {vid} #{index}")
    return False


def generate_variant_dataset():
    random.seed(ADDON_SEED + 2)
    root = VARIANT_OUTPUT_DIR
    _LOCAL_VERT_CACHE.clear()
    purge_helpers_from_stage_collections()
    validate_stage_collections()

    combos = variant_combinations()
    per_combo = VARIANT_SMOKE_IMAGES if ADDON_SMOKE_TEST else VARIANT_IMAGES_PER_COMBO

    addon_log("=" * 70)
    addon_log(f"VARIANT (COMBINATION) DATASET -> {root}")
    addon_log(f"{len(combos)} combinations x {per_combo} images "
              f"= {len(combos) * per_combo}")
    addon_log(f"  known stages: {sum(1 for c in combos if c in VARIANT_CANONICAL)}, "
              f"new variants: {sum(1 for c in combos if c not in VARIANT_CANONICAL)}")
    addon_log("=" * 70)

    if not ADDON_SMOKE_TEST:
        _addon_clean_folder_prefix(root, "var_")
    ndjson = _addon_init_folder(root, "variant_combinations")

    visibility_backup = capture_visibility_states()
    hide_all_stage_objects()
    hide_all_targets()
    hide_all_non_stage_meshes([])

    index_out = {}
    summary = []
    try:
        all_objects = show_only_stage(VARIANT_SOURCE_STAGE)
        base_all = {o.name: o.matrix_world.copy() for o in all_objects}
        by_class = {}
        for obj in all_objects:
            by_class.setdefault(obj["dataset_class"], []).append(obj)

        for combo in combos:
            vid = variant_id(combo)
            objects = variant_select_objects(by_class, combo)
            base_matrices = {o.name: base_all[o.name] for o in objects}
            stats = {"ok": 0, "all_visible": 0}

            hide_all_non_stage_meshes(objects)
            for obj in objects:
                obj.hide_render = False
                obj.hide_viewport = False

            failed = []
            for i in range(1, per_combo + 1):
                if not _variant_render_one(combo, objects, base_matrices, i,
                                           per_combo, root, ndjson, stats):
                    failed.append(i)
            for _rnd in range(1, ADDON_RETRY_ROUNDS + 1):
                if not failed:
                    break
                failed = [i for i in failed
                          if not _variant_render_one(combo, objects, base_matrices,
                                                     i, per_combo, root, ndjson,
                                                     stats)]

            for obj in all_objects:
                obj.matrix_world = base_all[obj.name].copy()
            bpy.context.view_layer.update()

            index_out[vid] = {
                "components": variant_components(combo),
                "canonical_stage": VARIANT_CANONICAL.get(combo),
                "images": stats["ok"],
                "images_all_parts_visible": stats["all_visible"],
            }
            line = (f"{vid}: {stats['ok']}/{per_combo}"
                    + (f"  [{VARIANT_CANONICAL[combo]}]"
                       if combo in VARIANT_CANONICAL else "  [variant]"))
            summary.append(line)
            addon_log("FINAL " + line)
    finally:
        restore_visibility_states(visibility_backup)
        hide_all_stage_objects()
        hide_all_targets()
        hide_all_non_stage_meshes([])
        _LOCAL_VERT_CACHE.clear()

    # The runtime reads this file to name states and to tell a known stage
    # from an unknown-but-valid combination.
    (root / "variant_index.json").write_text(
        json.dumps({
            "classes": CLASS_NAMES,
            "source_stage": VARIANT_SOURCE_STAGE,
            "canonical_stages": {variant_id(k): v
                                 for k, v in VARIANT_CANONICAL.items()},
            "combinations": index_out,
        }, indent=2), encoding="utf-8")
    addon_log(f"Wrote {root / 'variant_index.json'}")
    return summary


# ============================================================
# V19.1: 360-DEGREE STAGE PREVIEWS (turntable)
# ------------------------------------------------------------
# Renders each stage as a ring of frames (default 36, one every 10 degrees)
# on a TRANSPARENT background, with fixed lighting and fixed framing.
#
# The monitor does not need a 3D engine: it loads the frames for the current
# stage and shows them one after another, which looks like the assembly
# rotating. Because the frames come from the same CAD and the same renderer
# as the training data, the operator sees exactly what the system was taught.
#
# These images are PREVIEWS ONLY. They carry no labels and are written to
# their own folder, so they can never end up in the training set.
# ============================================================

RUN_STAGE_PREVIEWS = True
PREVIEW_OUTPUT_DIR = OUTPUT_DIR / "stage_previews"
PREVIEW_FRAMES = 36                  # 36 frames = one every 10 degrees
PREVIEW_ELEVATION_DEG = 28.0         # viewing angle, fixed for all frames
PREVIEW_TARGET_FRACTION = 0.62       # how much of the frame the assembly fills
PREVIEW_RESOLUTION = 480             # square output, good for a side panel
PREVIEW_LIGHT_ENERGY = 180.0
PREVIEW_LIGHT_KELVIN = 5200.0
PREVIEW_INCLUDE_VARIANTS = False     # True also previews the V19 combinations


def _preview_setup_scene(scene):
    """Neutral, repeatable look: transparent background, fixed light,
    no ground plane, no workbench. Returns what must be restored."""
    saved = {
        "res_x": scene.render.resolution_x,
        "res_y": scene.render.resolution_y,
        "transparent": scene.render.film_transparent,
        "ground": None,
        "bench": None,
        "exposure": scene.view_settings.exposure,
    }
    scene.render.resolution_x = PREVIEW_RESOLUTION
    scene.render.resolution_y = PREVIEW_RESOLUTION
    scene.render.film_transparent = True
    scene.view_settings.exposure = 0.0

    for key, name in (("ground", GROUND_NAME), ("bench", WORKBENCH_NAME)):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            saved[key] = (obj, obj.hide_render, obj.hide_viewport)
            obj.hide_render = True
            obj.hide_viewport = True

    key_light = get_object(LIGHT_NAME)
    if key_light is not None:
        saved["light"] = (key_light.data.energy, tuple(key_light.data.color),
                          key_light.location.copy())
        key_light.data.energy = PREVIEW_LIGHT_ENERGY
        key_light.data.color = _kelvin_to_rgb(PREVIEW_LIGHT_KELVIN)
    return saved


def _preview_restore_scene(scene, saved):
    scene.render.resolution_x = saved["res_x"]
    scene.render.resolution_y = saved["res_y"]
    scene.render.film_transparent = saved["transparent"]
    scene.view_settings.exposure = saved["exposure"]
    for key in ("ground", "bench"):
        if saved.get(key):
            obj, hr, hv = saved[key]
            obj.hide_render = hr
            obj.hide_viewport = hv
    if "light" in saved:
        key_light = get_object(LIGHT_NAME)
        if key_light is not None:
            energy, color, loc = saved["light"]
            key_light.data.energy = energy
            key_light.data.color = color
            key_light.location = loc


def _preview_render_one_set(name, objects, out_dir, scene, cam):
    """Render PREVIEW_FRAMES frames around the object set."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("frame_*.png"):
        f.unlink()

    center = group_bbox_center(objects)
    key_light = get_object(LIGHT_NAME)

    written = 0
    for i in range(PREVIEW_FRAMES):
        az = (360.0 / PREVIEW_FRAMES) * i
        if not _set_collection_camera(
            PREVIEW_TARGET_FRACTION, objects,
            azimuth_override=az, elevation_override=PREVIEW_ELEVATION_DEG,
            shift_x=0.0, shift_y=0.0,
        ):
            addon_log(f"  preview {name}: camera failed at {az:.0f} deg")
            continue

        # Light follows the camera so the assembly is lit the same way in
        # every frame instead of going dark on the far side.
        if key_light is not None and cam is not None:
            direction = (cam.location - center).normalized()
            key_light.location = (center + direction * 1.2
                                  + Vector((0.0, 0.0, 0.8)))

        path = out_dir / f"frame_{i:03d}.png"
        if path.exists():
            path.unlink()
        if render_and_verify(scene, path):
            written += 1
    return written


def generate_stage_previews():
    """Turntable previews for the 7 stages (and optionally the variants)."""
    scene = bpy.context.scene
    cam = get_object(CAMERA_NAME)
    root = PREVIEW_OUTPUT_DIR
    root.mkdir(parents=True, exist_ok=True)

    _LOCAL_VERT_CACHE.clear()
    purge_helpers_from_stage_collections()
    validate_stage_collections()

    addon_log("=" * 70)
    addon_log(f"STAGE PREVIEWS (360 deg turntable) -> {root}")
    addon_log(f"{PREVIEW_FRAMES} frames per stage at "
              f"{360.0 / PREVIEW_FRAMES:.0f} deg steps, "
              f"{PREVIEW_RESOLUTION}x{PREVIEW_RESOLUTION}, transparent background")
    addon_log("=" * 70)

    visibility_backup = capture_visibility_states()
    saved = _preview_setup_scene(scene)
    hide_occluder()
    hide_all_stage_objects()
    hide_all_targets()

    index = {}
    summary = []
    try:
        for stage_name in STAGE_COLLECTIONS:
            s01_collection = None
            s01_objects = None
            try:
                if stage_name == "Stage_01":
                    objects, s01_collection, _base = (
                        _prepare_authoritative_stage01_for_generation())
                    s01_objects = objects
                    hide_all_stage_objects()
                    hide_all_targets()
                    hide_all_non_stage_meshes(objects)
                    for obj in objects:
                        obj.hide_render = False
                        obj.hide_viewport = False
                else:
                    objects = show_only_stage(stage_name)

                n = _preview_render_one_set(stage_name, objects,
                                            root / stage_name, scene, cam)
                index[stage_name] = {
                    "frames": n,
                    "folder": (root / stage_name).as_posix(),
                    "components": dict(STAGE_EXPECTED_SEQUENCE.get(stage_name, {})),
                }
                summary.append(f"preview {stage_name}: {n}/{PREVIEW_FRAMES} frames")
                addon_log(f"FINAL preview {stage_name}: {n}/{PREVIEW_FRAMES} frames")
            finally:
                if s01_collection is not None:
                    _cleanup_stage01_authoritative(s01_objects, s01_collection)
                hide_all_stage_objects()
                hide_all_targets()

        if PREVIEW_INCLUDE_VARIANTS and "variant_combinations" in globals():
            all_objects = show_only_stage(VARIANT_SOURCE_STAGE)
            base_all = {o.name: o.matrix_world.copy() for o in all_objects}
            by_class = {}
            for obj in all_objects:
                by_class.setdefault(obj["dataset_class"], []).append(obj)
            for combo in variant_combinations():
                if combo in VARIANT_CANONICAL:
                    continue            # already rendered as a stage
                vid = variant_id(combo)
                objects = variant_select_objects(by_class, combo)
                hide_all_non_stage_meshes(objects)
                for obj in objects:
                    obj.hide_render = False
                    obj.hide_viewport = False
                n = _preview_render_one_set(vid, objects, root / vid, scene, cam)
                index[vid] = {
                    "frames": n,
                    "folder": (root / vid).as_posix(),
                    "components": variant_components(combo),
                }
                summary.append(f"preview {vid}: {n}/{PREVIEW_FRAMES} frames")
                addon_log(f"FINAL preview {vid}: {n}/{PREVIEW_FRAMES} frames")
            for obj in all_objects:
                obj.matrix_world = base_all[obj.name].copy()
            bpy.context.view_layer.update()
    finally:
        _preview_restore_scene(scene, saved)
        restore_visibility_states(visibility_backup)
        hide_all_stage_objects()
        hide_all_targets()
        _LOCAL_VERT_CACHE.clear()

    (root / "preview_index.json").write_text(
        json.dumps({
            "frames_per_set": PREVIEW_FRAMES,
            "degrees_per_frame": 360.0 / PREVIEW_FRAMES,
            "resolution": PREVIEW_RESOLUTION,
            "transparent_background": True,
            "sets": index,
        }, indent=2), encoding="utf-8")
    addon_log(f"Wrote {root / 'preview_index.json'}")
    return summary


def main():
    log("=" * 70)
    log("YOLO SYNTHETIC DATASET GENERATOR - V19.1 (full-rotation + variants + 360 stage previews)")
    log(f"Original passes: single={RUN_SINGLE_PART_DATASET} "
        f"subassembly={RUN_SUBASSEMBLY_DATASET}")
    log(f"Add-on passes:   single={RUN_ADDON_SINGLE_PART} "
        f"subassembly={RUN_ADDON_SUBASSEMBLY} variants={RUN_ADDON_VARIANTS} "
        f"previews={RUN_STAGE_PREVIEWS} smoke={ADDON_SMOKE_TEST}")
    log("=" * 70)

    setup_render()
    ensure_output_dirs()
    load_background_files()
    if not BACKGROUND_FILES:
        raise RuntimeError(f"No background images found in: {BACKGROUND_DIR}")

    resolved_sources = validate_required_classes()
    hide_all_stage_objects()
    hide_all_targets()
    _remember_environment()

    single_summary, stage_summary, variant_summary = [], [], []
    preview_summary = []
    try:
        # ---------------- original V17 passes (off by default) -----------
        if RUN_SINGLE_PART_DATASET:
            original_count = IMAGES_PER_PART
            if SMOKE_TEST:
                globals()["IMAGES_PER_PART"] = SMOKE_IMAGES_PER_PART
            log(f"ORIGINAL SINGLE-PART PASS: {IMAGES_PER_PART} images/class")
            initialize_ndjson()
            for class_name in CLASS_NAMES:
                for index in range(1, IMAGES_PER_PART + 1):
                    render_one(class_name, resolved_sources[class_name], index)
            globals()["IMAGES_PER_PART"] = original_count

        if RUN_SUBASSEMBLY_DATASET:
            generate_collection_subassembly_dataset()

        # ---------------- V18 add-on passes -------------------------------
        if RUN_ADDON_SINGLE_PART:
            single_summary = generate_addon_single_part_dataset(resolved_sources)

        if RUN_ADDON_SUBASSEMBLY:
            stage_summary = generate_addon_subassembly_dataset()

        # ---------------- V19 variant (combination) pass -------------------
        if RUN_ADDON_VARIANTS:
            variant_summary = generate_variant_dataset()

        # ---------------- V19.1 stage previews (no labels) -----------------
        if RUN_STAGE_PREVIEWS:
            preview_summary = generate_stage_previews()
    finally:
        _restore_environment()
        hide_all_stage_objects()
        unhide_all_parts()

    if (RUN_SINGLE_PART_DATASET and RUN_SUBASSEMBLY_DATASET
            and not SMOKE_TEST):
        final_production_dataset_audit()

    log("=" * 70)
    log("ADD-ON AUDIT")
    for line in single_summary + stage_summary + variant_summary + preview_summary:
        log("  " + line)
    if RUN_ADDON_SINGLE_PART:
        addon_audit(ADDON_SINGLE_OUTPUT_DIR)
    if RUN_ADDON_SUBASSEMBLY:
        addon_audit(ADDON_SUBASSEMBLY_OUTPUT_DIR)
    if RUN_ADDON_VARIANTS:
        addon_audit(VARIANT_OUTPUT_DIR)
    write_combined_data_yaml()
    if not ADDON_SMOKE_TEST and (RUN_ADDON_SINGLE_PART or RUN_ADDON_SUBASSEMBLY):
        addon_final_verdict()
    log("=" * 70)
    if ADDON_SMOKE_TEST:
        log("ADDON_SMOKE_TEST=True: open the images and check that the key "
            "parts (nut, threaded_pin, bushings) are clearly visible. Then set "
            "ADDON_SMOKE_TEST=False for the full run.")
    log("=" * 70)


if __name__ == "__main__":
    main()
