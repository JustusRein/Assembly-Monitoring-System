
import ast
import re
import sys
from pathlib import Path

def replace_once(text, pattern, replacement, label):
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.M | re.S)
    if count != 1:
        raise RuntimeError("Patch failed: " + label)
    return new_text

def replace_constant(text, name, value):
    return replace_once(
        text,
        rf"^{re.escape(name)}\s*=.*$",
        f"{name} = {value}",
        name,
    )

def replace_function(text, name, new_function):
    pattern = rf"^def {re.escape(name)}\(.*?(?=^def |\Z)"
    return replace_once(text, pattern, new_function.rstrip() + "\n\n", "function " + name)

def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python apply_v18_generator_patch.py <REAL_V17_GENERATOR.py>")

    src = Path(sys.argv[1]).expanduser().resolve()
    if not src.exists() or src.suffix.lower() != ".py":
        raise SystemExit("Generator .py not found: " + str(src))

    original = src.read_text(encoding="utf-8")

    required = [
        "IMAGES_PER_PART",
        "IMAGES_PER_STAGE",
        "def choose_split(index):",
        "def choose_target_fraction():",
        "def render_one(class_name, object_name, index):",
        "def stage_view_for_index(index):",
        "def stage_target_fraction(view_name):",
        "def stage_camera_angles_for_index(index):",
        "def final_production_dataset_audit():",
    ]

    missing = [item for item in required if item not in original]
    if missing:
        raise SystemExit(
            "ABORTED. The original file was NOT changed.\nMissing anchors:\n- "
            + "\n- ".join(missing)
        )

    text = original

    text = replace_constant(text, "OCCLUSION_PROBABILITY", "0.0")
    text = replace_constant(text, "STAGE_ELEVATION_BANDS_DEG", "(58.0, 68.0, 76.0, 83.0, 89.0)")
    text = replace_constant(text, "STAGE_ELEVATION_JITTER_DEG", "3.0")
    text = replace_constant(text, "CAM_ELEV_LOW_MIN", "52.0")
    text = replace_constant(text, "CAM_ELEV_LOW_MAX", "70.0")
    text = replace_constant(text, "CAM_ELEV_HIGH_MIN", "70.0")
    text = replace_constant(text, "CAM_ELEV_HIGH_MAX", "89.0")
    text = replace_constant(
        text,
        "MIN_BOX_PIXELS",
        '{"CLOSE": 8, "MEDIUM": 6, "FAR": 4, "TINY": 4}',
    )

    if "IMAGES_PER_CLASS = {" not in text:
        block = (
            "IMAGES_PER_STAGE = 200\n\n"
            "# Class-weighted single-part generation for the weak real-camera classes.\n"
            "IMAGES_PER_CLASS = {\n"
            '    "nut": 500,\n'
            '    "threaded_pin": 500,\n'
            '    "bushing": 400,\n'
            '    "block_pin": 300,\n'
            '    "knob": 250,\n'
            '    "body_a": 200,\n'
            '    "body_b": 200,\n'
            "}"
        )
        text = replace_once(
            text,
            r"^IMAGES_PER_STAGE\s*=\s*200\s*$",
            block,
            "IMAGES_PER_CLASS insertion",
        )

    text = replace_function(
        text,
        "choose_target_fraction",
        """def choose_target_fraction(class_name=None):
    name = str(class_name or "").strip()
    small = name in {"nut", "bushing", "threaded_pin", "block_pin"}
    r = random.random()

    if small:
        if r < 0.40:
            return random.uniform(0.015, 0.05)
        if r < 0.75:
            return random.uniform(0.05, 0.14)
        if r < 0.95:
            return random.uniform(0.14, 0.30)
        return random.uniform(0.30, 0.55)

    if r < 0.20:
        return random.uniform(0.05, 0.14)
    if r < 0.60:
        return random.uniform(0.14, 0.30)
    if r < 0.95:
        return random.uniform(0.30, 0.55)
    return random.uniform(0.55, 0.75)
""",
    )

    text = text.replace(
        "target_fraction = choose_target_fraction()",
        "target_fraction = choose_target_fraction(class_name)",
        1,
    )

    text = replace_function(
        text,
        "choose_camera_elevation",
        """def choose_camera_elevation():
    r = random.random()
    if r < 0.20:
        return random.uniform(CAM_ELEV_LOW_MIN, CAM_ELEV_LOW_MAX)
    if r < 0.80:
        return random.uniform(CAM_ELEV_LOW_MAX, CAM_ELEV_HIGH_MIN)
    return random.uniform(CAM_ELEV_HIGH_MIN, CAM_ELEV_HIGH_MAX)
""",
    )

    text = replace_function(
        text,
        "choose_split",
        """def choose_split(index, total_count):
    train_end = int(total_count * SPLIT_TRAIN)
    val_end = train_end + int(total_count * SPLIT_VAL)

    if index <= train_end:
        return "train"
    if index <= val_end:
        return "val"
    return "test"
""",
    )

    text = text.replace(
        "split = choose_split(index)",
        "split = choose_split(index, IMAGES_PER_PART)",
        1,
    )

    text = replace_constant(
        text,
        "STAGE_VIEW_CYCLE",
        """(
    "TINY", "FAR", "MEDIUM", "FAR", "TINY",
    "MEDIUM", "FAR", "TINY", "MEDIUM", "CLOSE",
    "TINY", "FAR", "MEDIUM", "FAR", "TINY",
    "MEDIUM", "FAR", "TINY", "MEDIUM", "CLOSE",
)""",
    )

    text = replace_function(
        text,
        "stage_view_for_index",
        """def stage_view_for_index(index):
    if SMOKE_TEST:
        return ["TINY", "FAR", "MEDIUM"][min(index - 1, 2)]
    return STAGE_VIEW_CYCLE[(index - 1) % len(STAGE_VIEW_CYCLE)]
""",
    )

    text = replace_function(
        text,
        "stage_target_fraction",
        """def stage_target_fraction(view_name):
    # Whole-stage scale is deliberately separate from single-part scale.
    if view_name == "TINY":
        return random.uniform(0.07, 0.11)
    if view_name == "FAR":
        return random.uniform(0.10, 0.16)
    if view_name == "MEDIUM":
        return random.uniform(0.15, 0.22)
    return random.uniform(0.22, 0.32)
""",
    )

    text = replace_function(
        text,
        "stage_camera_angles_for_index",
        """def stage_camera_angles_for_index(index):
    zero = index - 1
    golden_angle = 137.508

    azimuth = (zero * golden_angle) % 360.0
    azimuth += random.uniform(
        -CAMERA_RELATIVE_VIEW_JITTER_DEG,
        CAMERA_RELATIVE_VIEW_JITTER_DEG,
    )

    elevation = STAGE_ELEVATION_BANDS_DEG[zero % len(STAGE_ELEVATION_BANDS_DEG)]
    elevation += random.uniform(
        -STAGE_ELEVATION_JITTER_DEG,
        STAGE_ELEVATION_JITTER_DEG,
    )

    elevation = max(CAM_ELEV_LOW_MIN, min(CAM_ELEV_HIGH_MAX, elevation))
    return azimuth, elevation
""",
    )

    loop_pattern = (
        r"    for class_name in CLASS_NAMES:\s*"
        r"object_name = resolved_sources\[class_name\]\s*"
        r"for index in range\(1, IMAGES_PER_PART \+ 1\):\s*"
        r"ok = render_one\(class_name, object_name, index\)\s*"
        r"if ok:\s*"
        r"total_ok \+= 1\s*"
        r"else:\s*"
        r"total_fail \+= 1"
    )

    new_loop = """    for class_name in CLASS_NAMES:
        object_name = resolved_sources[class_name]
        class_count = IMAGES_PER_CLASS.get(class_name, IMAGES_PER_PART)

        original_class_total = IMAGES_PER_PART
        globals()["IMAGES_PER_PART"] = class_count
        try:
            for index in range(1, class_count + 1):
                ok = render_one(class_name, object_name, index)
                if ok:
                    total_ok += 1
                else:
                    total_fail += 1
        finally:
            globals()["IMAGES_PER_PART"] = original_class_total
"""

    text = replace_once(
        text,
        loop_pattern,
        new_loop,
        "single-part per-class loop",
    )

    text = replace_function(
        text,
        "final_production_dataset_audit",
        """def final_production_dataset_audit():
    failures = []

    for class_name in CLASS_NAMES:
        expected_part = IMAGES_PER_CLASS.get(class_name, IMAGES_PER_PART)
        train_n = int(expected_part * SPLIT_TRAIN)
        val_n = int(expected_part * SPLIT_VAL)

        for split in ("train", "val", "test"):
            folder = IMAGES_DIR / split
            count = len(list(folder.glob(f"{class_name}_*.png")))
            expected = (
                train_n
                if split == "train"
                else val_n
                if split == "val"
                else expected_part - train_n - val_n
            )

            if count != expected:
                failures.append(
                    f"single-part {class_name}/{split}: {count} != {expected}"
                )

    expected_stage = IMAGES_PER_STAGE
    train_stage = int(expected_stage * SPLIT_TRAIN)
    val_stage = int(expected_stage * SPLIT_VAL)

    for stage_name in STAGE_COLLECTIONS:
        stem = stage_name.lower()

        for split in ("train", "val", "test"):
            folder = SUBASSEMBLY_OUTPUT_DIR / "images" / split
            count = len(list(folder.glob(f"{stem}_*.png")))
            expected = (
                train_stage
                if split == "train"
                else val_stage
                if split == "val"
                else expected_stage - train_stage - val_stage
            )

            if count != expected:
                failures.append(
                    f"subassembly {stage_name}/{split}: {count} != {expected}"
                )

    if failures:
        raise RuntimeError(
            "FINAL DATASET AUDIT FAILED:\\n  - "
            + "\\n  - ".join(failures)
        )

    total_single = sum(
        IMAGES_PER_CLASS.get(name, IMAGES_PER_PART)
        for name in CLASS_NAMES
    )
    total_subassembly = len(STAGE_COLLECTIONS) * IMAGES_PER_STAGE

    log("=" * 70)
    log("FINAL DATASET AUDIT PASSED")
    log(f"Single-part total: {total_single}")
    log(f"Subassembly total: {total_subassembly}")
    log(f"TOTAL: {total_single + total_subassembly} images")
    log("Train/Val/Test split verified on disk.")
    log("=" * 70)
""",
    )

    text = text.replace(
        "PRODUCTION: 200 images/class + 200 images/stage",
        "PRODUCTION: class-weighted single-part images + 200 images/stage",
        1,
    )

    ast.parse(text)

    out = src.with_name(src.stem + "_V18_deployment_matched.py")
    out.write_text(text, encoding="utf-8")

    print("PATCH SUCCESS")
    print("Original:", src)
    print("New:", out)

if __name__ == "__main__":
    main()
