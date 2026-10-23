# `synthetic_data_generator/` - CAD to labelled training data

This folder holds the Blender pipeline that produced the training data for the detector in this repository. It is the answer to the last clause of the thesis research question: *"particularly with limited datasets."*

**The problem it solves.** Object detection needs labelled images. For specialised one-off components none exist, and hand-annotating every new product variant does not scale - it costs effort per image, produces inconsistent labels, and has to be repeated for every product, which defeats the point of a transferable method.

**What it does instead.** Components are placed in a parameterised Blender scene with randomised pose, camera position, lighting and background. The renderer produces the image and the label together, so every bounding box is computed from known 3-D geometry and a known camera. **Exact by construction, with no box drawn by hand.**

---

## Important: the CAD geometry is not included

The original scripts embedded two meshes directly in the source as base64 binary STL - the Stage-01 body (5,835 triangles) and a bushing. That geometry is derived from a STEP file supplied by an industrial partner and **is not published.**

Both payloads have been replaced with `None`, under a comment explaining why. The scripts already contained a fallback path, so they still run: when the embedded payload fails to decode, the loader reads the corresponding STL from disk instead.

### What you have to supply

Place your own CAD under `SUBASSEMBLY_ASSETS_DIR` (set near the top of each generator):

```
<SUBASSEMBLY_ASSETS_DIR>/
└── stage_01/
    ├── 1_body_a.stl      (or body_a.stl)
    └── 4_bushing.stl     (or bushing_A.stl)
```

If neither file is present the generator raises a clear error naming what is missing. It does not fail silently.

Everything else - the scene construction, randomisation, visibility-aware labelling, split logic and YOLO export - is unchanged and complete.

---

## The generators

Four versions are included because they produce different parts of the final dataset, and the lineage matters for reproducing the reported results.

| Script | Produces | Note |
| --- | --- | --- |
| `generate_dataset_YOLO_v17_LIGHTFIX.py` | `images/`, `subassembly/` | Base single-part and ordered subassembly scenes |
| `generate_dataset_YOLO_v18_3_PRODUCTION.py` | adds `addon_single_fullrot/`, `subassembly_addon_fullrot/` | Full-rotation coverage |
| `generate_dataset_YOLO_v19_1_VARIANTS_PREVIEWS.py` | adds `variants_fullrot/` | **Produces all five sources used by the deployed checkpoint** |
| `generate_dataset_YOLO_V24_FINAL_DIVERSE.py` | `images/`, `subassembly/` | Later diversity-focused variant |
| `apply_v18_generator_patch.py` | - | Patch helper for the v18 generator |

**If you only want to reproduce the thesis dataset, use `v19_1_VARIANTS_PREVIEWS`.**

## What gets randomised

| Parameter | Purpose |
| --- | --- |
| Part orientation | Recognition is not tied to one fixed orientation |
| Camera azimuth | Horizontal viewpoint variation |
| Camera elevation | Upper and side viewing direction |
| Camera distance and framing | Variation in object size - close, medium, far |
| Object placement on the work surface | Different spatial positions, realistic contact |
| Lighting energy and position | Reduces dependence on one illumination condition |
| Background photograph | Realistic workshop and workstation backgrounds |
| Visibility-aware labelling | Hidden geometry is not labelled when it is not visible to the camera |

**Why both single-part and subassembly scenes.** Single-part images expose the shape of a component without scene complexity. Subassembly images place the same objects in the context in which they actually occur during the task, as ordered stages rather than unrelated collections of objects.

---

## The dataset this produced

| Source folder | Train | Val | Test | Total |
| --- | --- | --- | --- | --- |
| `images/` | 840 | 280 | 280 | 1,400 |
| `subassembly/` | 840 | 280 | 280 | 1,400 |
| `addon_single_fullrot/` | 810 | 277 | 277 | 1,364 |
| `subassembly_addon_fullrot/` | 802 | 271 | 274 | 1,347 |
| `variants_fullrot/` | 624 | 208 | 208 | 1,040 |
| **Total** | **3,916** | **1,316** | **1,319** | **6,551** |

Annotated object instances: **8,714** in training, **2,938** in validation, **2,922** in testing.

Seven classes, listed in `classes.txt`: `body_a`, `body_b`, `block_pin`, `bushing`, `nut`, `knob`, `threaded_pin`. Seven classes for eight physical parts, because the two bushings are the same object category - the downstream state machine counts instances rather than treating the class as a yes/no flag.

The dataset itself is not in this repository because of its size. `data_combined_v18.yaml` is the five-source training configuration; set its `path` to your own generated output.

> An earlier `combined/` dataset of 2,800 images exists but is **not** one of the five sources and was not used for the deployed checkpoint.

---

## Running it

The generators are Blender scripts, not standalone Python. They need Blender with its Python API available.

```bash
blender --background <your_scene>.blend --python generators/generate_dataset_YOLO_v19_1_VARIANTS_PREVIEWS.py
```

Before the first run, set these near the top of the script:

| Variable | Meaning |
| --- | --- |
| `OUTPUT_DIR` | Where the rendered dataset is written |
| `SUBASSEMBLY_ASSETS_DIR` | Where your stage CAD lives (see above) |

The absolute paths from the development machine have been left in place as examples; change them to your own.

---

## Honest limitations

- **Synthetic-to-real gap.** Rendered images cannot reproduce surface finish, illumination gradients, sensor response, motion blur, shadows or hand occlusion. Detector metrics obtained on this data describe the synthetic distribution, not a physical workstation. This is the first threat to validity named in the thesis.
- **Colour is part of the learned distribution.** The appearance was deliberately kept fixed - body and knob green, fasteners metallic. The model should not be expected to reject a geometrically identical component in a different colour, because that case lies outside the defined training domain.
- **This does not remove the need for real validation.** Synthetic data reduces the effort of creating labelled training data. It does not remove the requirement for independent testing on real images, which is the second item of future work in the thesis.

See the [main README](../README.md) for the detector results and the full evidence boundary.
