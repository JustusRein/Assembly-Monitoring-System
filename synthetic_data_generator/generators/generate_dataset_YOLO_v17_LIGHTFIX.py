"""
===============================================================================
FINAL REALISTIC YOLO SYNTHETIC DATASET GENERATOR - V16
Script A + NDJSON  (single parts on plate, real colours, backgrounds)
===============================================================================

DIAGNOSTIC VERSION - read this first:
    Nothing in your working logic was changed or rewritten. Exactly TWO
    small print blocks were added, both inside render_subassembly_one():

    1. Right after camera placement succeeds: prints the camera's real
       position, the assembly group's center, and each imported part's
       Z-range - so we can see exactly where everything actually is.

    2. Right after the PNG is written to disk: loads that EXACT file back
       and prints its average pixel brightness. This is the deciding
       number:
         - near 0.0        -> genuinely black (light/camera/visibility bug)
         - ~0.06-0.08      -> camera is seeing the neutral world colour,
                               not your parts (camera aim/placement bug)
         - something else  -> tells us a new, different story

    Also temporarily restricted to Stage 1 only, 1 image, so you get an
    Stage_01 uses the verified authoritative CAD-derived component pair; Stage_02 ... Stage_07 use the manually assembled Blender collections.

-------------------------------------------------------------------------------
HOW TO RUN:
    1. Open MasterScene_SUBASSEMBLY_WORKING.blend.
    2. Scripting tab -> New -> paste this whole file -> press Run.
    3. Copy EVERY line of console output and send it back - especially
       any line starting with "DIAG:".
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
RUN_SINGLE_PART_DATASET = True
RUN_SUBASSEMBLY_DATASET = True

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
CLEAN_SUBASSEMBLY_OUTPUT_BEFORE_PRODUCTION = True

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
    leftovers = ["TEMP_BACKDROP", "TEMP_OCCLUDER", "PHOTO_GROUND"]
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

def make_occluder():
    occ = bpy.data.objects.get(OCCLUDER_NAME)
    if occ is not None:
        return occ
    mesh = bpy.data.meshes.new(OCCLUDER_NAME + "_mesh")
    occ = bpy.data.objects.new(OCCLUDER_NAME, mesh)
    bpy.context.collection.objects.link(occ)
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


def collection_mesh_objects(coll):
    if coll is None:
        return []
    return [
        obj for obj in coll.all_objects
        if obj is not None
        and isinstance(obj, bpy.types.Object)
        and obj.type == "MESH"
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


def visible_object_bbox(scene, cam, obj):
    """Return bbox only if sampled geometry of the object is actually visible."""
    if not VISIBILITY_AWARE_LABELS:
        return compute_yolo_bbox(scene, cam, obj), True

    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = _sample_face_centers_world(obj)

    if not points:
        return None, False

    visible = []

    for point in points:
        origin = cam.matrix_world.translation.copy()
        direction = point - origin
        distance = direction.length

        if distance <= 1e-9:
            continue

        direction.normalize()

        try:
            hit, hit_location, normal, face_index, hit_obj, hit_matrix = (
                scene.ray_cast(
                    depsgraph,
                    origin,
                    direction,
                    distance=distance + 1e-6,
                )
            )
        except TypeError:
            hit, hit_location, normal, face_index, hit_obj, hit_matrix = (
                scene.ray_cast(depsgraph, origin, direction)
            )

        if hit and hit_obj == obj:
            visible.append(point)

    if len(visible) < MIN_VISIBLE_FACE_HITS:
        return None, False

    projected = [
        project_point(scene, cam, p)
        for p in visible
    ]

    projected = [(u, v) for u, v, front in projected if front]

    if len(projected) < 2:
        return None, False

    xs = [p[0] for p in projected]
    ys = [p[1] for p in projected]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if max_x <= min_x or max_y <= min_y:
        return None, False

    return (
        (
            (min_x + max_x) / 2.0,
            1.0 - (min_y + max_y) / 2.0,
            max_x - min_x,
            max_y - min_y,
        ),
        True,
    )


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



def main():
    log("=" * 70)
    log("FINAL COLLECTION-DRIVEN YOLO SYNTHETIC DATASET GENERATOR - V17 (light-fix)")
    log("SINGLE PARTS + MANUAL BLENDER SUBASSEMBLY COLLECTIONS")
    log("PRODUCTION: 200 images/class + 200 images/stage")
    log("=" * 70)

    setup_render()
    ensure_output_dirs()
    load_background_files()

    if not BACKGROUND_FILES:
        raise RuntimeError(
            f"No background images found in: {BACKGROUND_DIR}"
        )

    resolved_sources = validate_required_classes()

    # Keep all manually assembled stages hidden during single-part generation.
    hide_all_stage_objects()
    hide_all_targets()

    if RUN_SINGLE_PART_DATASET:
        original_count = IMAGES_PER_PART
        if SMOKE_TEST:
            globals()["IMAGES_PER_PART"] = SMOKE_IMAGES_PER_PART

        log("=" * 70)
        log(f"SINGLE-PART DATASET START: {IMAGES_PER_PART} images/class")
        log("=" * 70)

        initialize_ndjson()

        total_ok = 0
        total_fail = 0

        for class_name in CLASS_NAMES:
            object_name = resolved_sources[class_name]

            for index in range(1, IMAGES_PER_PART + 1):
                ok = render_one(class_name, object_name, index)
                if ok:
                    total_ok += 1
                else:
                    total_fail += 1

        globals()["IMAGES_PER_PART"] = original_count

        log(
            f"SINGLE-PART DATASET COMPLETE: "
            f"valid={total_ok}, failed={total_fail}"
        )

    if RUN_SUBASSEMBLY_DATASET:
        generate_collection_subassembly_dataset()

    hide_all_stage_objects()
    unhide_all_parts()

    if not SMOKE_TEST:
        final_production_dataset_audit()

    log("=" * 70)
    log("FINAL DATASET GENERATION COMPLETE")
    log("Individual parts: 1,400 images | Ordered stages: 1,400 images | TOTAL: 2,800 images")
    if SMOKE_TEST:
        log(
            "SMOKE_TEST=True. Verify the generated smoke dataset. "
            "Then set SMOKE_TEST=False for the full production run."
        )
    log("=" * 70)


if __name__ == "__main__":
    main()
