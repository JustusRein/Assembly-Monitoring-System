"""
STL-based Synthetic YOLO Dataset Generator
------------------------------------------
- Selects the newest subfolder in INPUT_ROOT
- Finds all STL files recursively
- Renders each STL from randomized viewpoints
- Uses the STL filename as the YOLO class name
- Creates train/val/test splits and YOLO labels
"""

import bpy
import math
import random
import re
from pathlib import Path
from datetime import datetime
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_ROOT = Path(
    r"C:\Users\rein\Git projects Justus\Assembly-Monitoring-System-V2.1"
    r"\synthetic_data_generator\input"
)

OUTPUT_ROOT = Path(
    r"C:\Users\rein\Git projects Justus\Assembly-Monitoring-System-V2.1"
    r"\synthetic_data_generator\output_stl_dataset"
)

IMAGES_PER_STL = 5

RES_X = 640
RES_Y = 640

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20

RANDOM_SEED = 42

CAMERA_NAME = "Camera"
LIGHT_NAME = "Light"

# Randomization
RANDOMIZE_ROTATION = True
RANDOMIZE_LIGHT = True

# Output is written to a new timestamped folder.
# Existing datasets are not deleted.
# ============================================================


random.seed(RANDOM_SEED)


def log(message):
    print(f"[STL-DATASET] {message}")


def safe_name(name):
    """Create a filesystem-safe name while preserving the STL stem."""
    name = re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE)
    return name.strip("._") or "unnamed_part"


def find_newest_input_folder():
    """Find the most recently modified subfolder in INPUT_ROOT."""
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {INPUT_ROOT}"
        )

    folders = [p for p in INPUT_ROOT.iterdir() if p.is_dir()]

    if not folders:
        raise RuntimeError(
            f"No subfolders found in input directory: {INPUT_ROOT}"
        )

    newest = max(folders, key=lambda p: p.stat().st_mtime)
    log(f"Newest input folder: {newest}")
    return newest


def find_stl_files(folder):
    """Find STL files recursively in the selected input folder."""
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() == ".stl"
    )

    if not files:
        raise RuntimeError(f"No STL files found in: {folder}")

    return files


def import_stl(filepath):
    """Import an STL using the available Blender STL operator."""
    before = set(bpy.data.objects)

    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=str(filepath))
    elif hasattr(bpy.ops.import_mesh, "stl"):
        bpy.ops.import_mesh.stl(filepath=str(filepath))
    else:
        raise RuntimeError(
            "No STL import operator found. "
            "Enable Blender's STL import functionality."
        )

    imported = [
        obj for obj in bpy.data.objects
        if obj not in before and obj.type == "MESH"
    ]

    if not imported:
        raise RuntimeError(f"STL import created no mesh: {filepath}")

    # Usually one mesh is imported. Join multiple meshes so the
    # entire STL file is treated as one YOLO object/class instance.
    bpy.ops.object.select_all(action="DESELECT")

    for obj in imported:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = imported[0]

    if len(imported) > 1:
        bpy.ops.object.join()

    obj = bpy.context.view_layer.objects.active
    obj.name = safe_name(filepath.stem)
    obj.data.name = obj.name + "_Mesh"

    return obj


def get_world_corners(obj):
    bpy.context.view_layer.update()
    return [
        obj.matrix_world @ Vector(corner)
        for corner in obj.bound_box
    ]


def get_bbox_center_and_size(obj):
    corners = get_world_corners(obj)

    minimum = Vector((
        min(p.x for p in corners),
        min(p.y for p in corners),
        min(p.z for p in corners),
    ))
    maximum = Vector((
        max(p.x for p in corners),
        max(p.y for p in corners),
        max(p.z for p in corners),
    ))

    center = (minimum + maximum) / 2
    size = maximum - minimum

    return center, size, minimum, maximum


def center_object(obj):
    """Center the imported geometry around the world origin."""
    center, _, _, _ = get_bbox_center_and_size(obj)

    for vertex in obj.data.vertices:
        vertex.co -= obj.matrix_world.inverted().to_3x3() @ center

    bpy.context.view_layer.update()


def ensure_camera():
    camera = bpy.data.objects.get(CAMERA_NAME)

    if camera is None or camera.type != "CAMERA":
        camera_data = bpy.data.cameras.new(CAMERA_NAME)
        camera = bpy.data.objects.new(CAMERA_NAME, camera_data)
        bpy.context.scene.collection.objects.link(camera)

    bpy.context.scene.camera = camera
    camera.data.lens = 50
    camera.data.clip_start = 0.001
    camera.data.clip_end = 1000

    return camera


def ensure_light():
    """Create or reuse an area light compatible with Blender 4.3."""

    light_obj = bpy.data.objects.get("DatasetLight")

    if light_obj is None or light_obj.type != "LIGHT":
        light_data = bpy.data.lights.new(
            name="DatasetLight",
            type="AREA"
        )
        light_obj = bpy.data.objects.new(
            "DatasetLight",
            light_data
        )
        bpy.context.scene.collection.objects.link(light_obj)

    light_obj.data.type = "AREA"
    light_obj.data.shape = "DISK"
    light_obj.data.energy = 1500
    light_obj.data.size = 5.0

    light_obj.location = (4.0, -4.0, 6.0)

    direction = Vector((0.0, 0.0, 0.0)) - light_obj.location
    light_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    return light_obj


def setup_render():
    scene = bpy.context.scene

    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        try:
            scene.render.engine = "BLENDER_EEVEE"
        except Exception:
            pass

    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False

    scene.render.image_settings.color_depth = "8"

    # Use a neutral, randomized world background.
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("STL_Dataset_World")
        scene.world = world

    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")

    if bg:
        bg.inputs["Color"].default_value = (0.25, 0.25, 0.25, 1)
        bg.inputs["Strength"].default_value = 0.8

    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_percentage = 100


def set_random_world_background():
    world = bpy.context.scene.world
    bg = world.node_tree.nodes.get("Background")

    if bg:
        value = random.uniform(0.12, 0.55)
        bg.inputs["Color"].default_value = (
            value, value, value, 1
        )


def aim_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def prepare_object(obj):
    """Place object on the ground and apply a random orientation."""
    obj.location = (0, 0, 0)
    obj.rotation_euler = (0, 0, 0)
    bpy.context.view_layer.update()

    center, _, minimum, _ = get_bbox_center_and_size(obj)
    obj.location.z -= minimum.z
    bpy.context.view_layer.update()

    if RANDOMIZE_ROTATION:
        obj.rotation_euler = (
            math.radians(random.uniform(-20, 20)),
            math.radians(random.uniform(-20, 20)),
            math.radians(random.uniform(0, 360)),
        )

    bpy.context.view_layer.update()

    # Correct the height after rotation so the object sits on z=0.
    _, _, minimum, _ = get_bbox_center_and_size(obj)
    obj.location.z -= minimum.z
    bpy.context.view_layer.update()


def position_camera(obj, camera, light):
    """Choose a random viewpoint and frame the whole object."""
    center, size, _, _ = get_bbox_center_and_size(obj)

    max_dim = max(size.x, size.y, size.z, 0.01)
    distance = max_dim * random.uniform(2.2, 3.5)

    azimuth = random.uniform(0, 2 * math.pi)
    elevation = math.radians(random.uniform(15, 80))

    camera.location = (
        center.x + distance * math.cos(elevation) * math.cos(azimuth),
        center.y + distance * math.cos(elevation) * math.sin(azimuth),
        center.z + distance * math.sin(elevation),
    )
    aim_at(camera, center)

    if RANDOMIZE_LIGHT:
        light.location = (
            center.x + random.uniform(-2, 2) * max_dim,
            center.y + random.uniform(-2, 2) * max_dim,
            center.z + random.uniform(1, 3) * max_dim,
        )
        light.data.energy = random.uniform(600, 1800)
        aim_at(light, center)


def get_yolo_bbox(obj, camera):
    """Project the object's 3D bounding box into normalized YOLO coordinates."""
    scene = bpy.context.scene
    corners = get_world_corners(obj)

    projected = [
        world_to_camera_view(scene, camera, point)
        for point in corners
    ]

    # All corners must be in front of the camera.
    if any(p.z <= 0 for p in projected):
        return None

    min_x = min(p.x for p in projected)
    max_x = max(p.x for p in projected)
    min_y = min(p.y for p in projected)
    max_y = max(p.y for p in projected)

    # Reject boxes outside the image instead of silently creating
    # inaccurate labels.
    if (
        min_x < 0 or min_y < 0
        or max_x > 1 or max_y > 1
        or max_x <= min_x or max_y <= min_y
    ):
        return None

    x_center = (min_x + max_x) / 2
    y_center = 1 - (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y

    return x_center, y_center, width, height


def create_output_dirs(output_dir):
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(
            parents=True, exist_ok=True
        )
        (output_dir / "labels" / split).mkdir(
            parents=True, exist_ok=True
        )


def choose_split(index, total):
    train_end = int(total * TRAIN_RATIO)
    val_end = int(total * (TRAIN_RATIO + VAL_RATIO))

    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def render_stl(filepath, class_id, class_name, output_dir,
               camera, light):
    log(f"Importing: {filepath.name}")

    # Hide scene meshes from any previously opened Blender scene.
    original_states = {}
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            original_states[obj.name] = obj.hide_render
            obj.hide_render = True

    imported_obj = None
    successful = 0

    try:
        imported_obj = import_stl(filepath)
        imported_obj.hide_render = False

        # Give the imported mesh a neutral material if it has none.
        if len(imported_obj.data.materials) == 0:
            material = bpy.data.materials.new(
                name="STL_Dataset_Material"
            )
            material.diffuse_color = (0.35, 0.45, 0.55, 1)
            imported_obj.data.materials.append(material)

        total = IMAGES_PER_STL

        for index in range(total):
            split = choose_split(index, total)
            stem = f"{class_name}_{index + 1:04d}"

            image_path = (
                output_dir / "images" / split / f"{stem}.png"
            )
            label_path = (
                output_dir / "labels" / split / f"{stem}.txt"
            )

            success = False

            for attempt in range(20):
                prepare_object(imported_obj)
                position_camera(imported_obj, camera, light)
                set_random_world_background()

                bbox = get_yolo_bbox(imported_obj, camera)
                if bbox is None:
                    continue

                bpy.context.scene.render.filepath = str(image_path)

                try:
                    bpy.ops.render.render(write_still=True)
                except Exception as exc:
                    log(f"Render failed for {stem}: {exc}")
                    continue

                if not image_path.exists():
                    continue

                x, y, w, h = bbox
                label_path.write_text(
                    f"{class_id} {x:.6f} {y:.6f} "
                    f"{w:.6f} {h:.6f}\n",
                    encoding="utf-8",
                )

                successful += 1
                success = True
                break

            if not success:
                log(
                    f"WARNING: Could not render {stem} "
                    "after 20 attempts."
                )

        log(
            f"Finished {filepath.name}: "
            f"{successful}/{IMAGES_PER_STL} images"
        )

    finally:
        if imported_obj is not None:
            mesh_data = imported_obj.data
            bpy.data.objects.remove(imported_obj, do_unlink=True)

            if mesh_data and mesh_data.users == 0:
                bpy.data.meshes.remove(mesh_data)

        # Restore visibility of meshes that were already in the scene.
        for name, hidden in original_states.items():
            obj = bpy.data.objects.get(name)
            if obj is not None:
                obj.hide_render = hidden


def write_yolo_config(output_dir, class_names):
    (output_dir / "classes.txt").write_text(
        "\n".join(class_names) + "\n",
        encoding="utf-8",
    )

    names_yaml = "\n".join(
        f"  {i}: '{name}'"
        for i, name in enumerate(class_names)
    )

    yaml_text = (
        f"path: '{output_dir.as_posix()}'\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(class_names)}\n"
        "names:\n"
        f"{names_yaml}\n"
    )

    (output_dir / "data.yaml").write_text(
        yaml_text, encoding="utf-8"
    )


def main():
    input_folder = find_newest_input_folder()
    stl_files = find_stl_files(input_folder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"{input_folder.name}_{timestamp}"

    create_output_dirs(output_dir)

    # Each STL stem becomes one class. Duplicate names in different
    # subfolders are disambiguated to prevent class collisions.
    class_names = []
    file_class_names = []

    for filepath in stl_files:
        base_name = safe_name(filepath.stem)

        class_name = base_name
        if class_name in class_names:
            relative = filepath.relative_to(input_folder).with_suffix("")
            class_name = safe_name(str(relative).replace("\\", "_").replace("/", "_"))

        suffix = 2
        original_name = class_name
        while class_name in class_names:
            class_name = f"{original_name}_{suffix}"
            suffix += 1

        class_names.append(class_name)
        file_class_names.append(class_name)

    write_yolo_config(output_dir, class_names)

    setup_render()
    camera = ensure_camera()
    light = ensure_light()

    log("=" * 60)
    log(f"Input folder: {input_folder}")
    log(f"STL files found: {len(stl_files)}")
    log(f"Images per STL: {IMAGES_PER_STL}")
    log(f"Expected images: {len(stl_files) * IMAGES_PER_STL}")
    log(f"Output folder: {output_dir}")
    log("=" * 60)

    for class_id, (filepath, class_name) in enumerate(
        zip(stl_files, file_class_names)
    ):
        try:
            render_stl(
                filepath=filepath,
                class_id=class_id,
                class_name=class_name,
                output_dir=output_dir,
                camera=camera,
                light=light,
            )
        except Exception as exc:
            log(f"ERROR processing {filepath}: {exc}")

    log("=" * 60)
    log("DATASET GENERATION FINISHED")
    log(f"Output: {output_dir}")
    log("=" * 60)


if __name__ == "__main__":
    main()