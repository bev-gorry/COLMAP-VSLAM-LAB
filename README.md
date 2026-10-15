# COLMAP for VSLAM-LAB

The `colmap` baseline of [VSLAM-LAB](https://github.com/VSLAM-LAB/VSLAM-LAB): a Python wrapper that runs
[COLMAP](https://colmap.github.io/) 4.2.1 (conda-forge package, CUDA build on linux-64 / win-64) on the
frames of a VSLAM-LAB sequence and writes the camera trajectory in VSLAM-LAB's format, optionally followed
by COLMAP's dense reconstruction and meshing. VSLAM-LAB clones this repository into `Baselines/colmap`
(`pixi run install-baseline colmap`); the pixi environment and the `colmap` binary come from the main
repository's `pixi.toml`.

## Pipeline

`vslamlab_colmap.py` is the entry point (`pixi run -e colmap execute-mono ...`, built by VSLAM-LAB from an
experiment's `Parameters:`). Stages, each in its own module:

| Stage | Module | COLMAP commands |
|---|---|---|
| extraction | `colmap_matcher.py` | `database_creator`, `feature_extractor` (frames of the experiment's `rgb_exp.csv`, intrinsics from `calibration_exp.yaml`, optional masks) |
| matching | `colmap_matcher.py` | `exhaustive_matcher` or `sequential_matcher` with loop detection |
| reconstruction | `colmap_mapper.py` | `mapper` (incremental) or `global_mapper` (GLOMAP); the sub-model with most registered images is kept; `model_converter` to TXT |
| trajectory | `colmap_to_vslamlab.py` | `images.txt` -> `<exp_id>_KeyFrameTrajectory.csv` (camera-to-world, timestamps from the rgb csv) |
| dense (optional) | `colmap_dense.py` | `image_undistorter`, `patch_match_stereo`, `stereo_fusion`, then `delaunay_mesher` / `poisson_mesher` / `advancing_front_mesher` |

Shared helpers live in `colmap_utilities.py` (COLMAP subprocess runner, GPU / thread detection, camera-model
mapping, settings forwarding, profiler); `get_calibration.py`, `create_colmap_image_list.py` and
`create_colmap_mask_dir.py` read the VSLAM-LAB inputs.

## Parameters

Experiment `Parameters:` keys understood by the baseline (the `rgb_*`, `segmentation`, `refraction`,
`calibration` keys are handled by VSLAM-LAB itself):

| Key | Default | Values |
|---|---|---|
| `matcher_type` | `exhaustive` | `exhaustive` (all pairs) · `sequential` (neighbouring frames + loop detection with COLMAP's default vocabulary tree for the feature type) |
| `matching_type` | `sift_bruteforce` | `sift_bruteforce` · `sift_lightglue` · `aliked_bruteforce` · `aliked_lightglue` · `loma_b_bruteforce` · `loma_b_loma` · `loma_b_loma_r` · `loma_b_loma_l` · `loma_b_loma_g` · `loma_b128_bruteforce` · `loma_b128_loma` |
| `mapper_type` | `colmap` | `colmap` (incremental mapper) · `glomap` (global mapper) |
| `optimize_intrinsics` | `1` | `1` refines focal length and distortion in bundle adjustment (principal point fixed); `0` keeps the calibration as given; forced to `1` for the `unknown` camera model |
| `use_mask` | `0` | `1` restricts feature extraction to the `path_mask_<i>` masks VSLAM-LAB provides (`segmentation: mask2former`, `refraction: refrax`, datasets shipping masks) |
| `dense` | `0` | `1` runs the dense stage after the trajectory is written (needs the CUDA build; skipped with a warning otherwise) |
| `dense_max_image_size` | `1600` | longest image side for undistortion, patch match and fusion (`1000` low, `1600` medium, `-1` full resolution) |
| `mesher` | `none` | `none` · `delaunay` · `poisson` · `advancing_front` |
| `verbose` | `1` | `1` opens `colmap gui` at the end on the sparse model, or on the fused cloud when `dense: 1`; the mesh can be loaded in that window via *File > Import model from...* |

Naming of `matching_type` is `<extractor>_<matcher>`. SIFT and ALIKED (`ALIKED_N16ROT`, 128-dim) match with
brute force or LightGlue. LoMa comes as `LOMA_B` (256-dim, DINOv2 + convolutional features) with the
dedicated matchers `loma` (B, LightGlue-sized), `loma_r` (rotation-augmented), `loma_l` and `loma_g`
(larger, slower, more accurate), and as `LOMA_B128` (128-dim) with its own `loma` matcher; `bruteforce`
(cosine similarity) works with both. Indicative cost on 200 frames with the sequential matcher (RTX 4000):
extraction 1.6 min (LoMa B) / 0.5 min (LoMa B128) / seconds (SIFT, ALIKED); matching 0.2-0.4 min brute
force, 2 min B / R / B128 / LightGlue, 4 min L, 10 min G.

Example:

```yaml
exp_colmap_loma_dense:
  Config: config_vslamlab.yaml
  NumRuns: 1
  Parameters: {verbose: 0, rgb_max: 200, rgb_step: 3, matcher_type: sequential, matching_type: loma_b_loma,
               mapper_type: colmap, dense: 1, dense_max_image_size: 1000, mesher: poisson}
  Module: colmap
```

## Camera models

VSLAM-LAB calibration -> COLMAP camera model (`colmap_utilities.colmap_camera`; the number of distortion
coefficients is checked):

| VSLAM-LAB | COLMAP | Parameters passed |
|---|---|---|
| `pinhole` | `PINHOLE` | fx fy cx cy |
| `radtan4` | `OPENCV` | fx fy cx cy k1 k2 p1 p2 |
| `radtan5` | `FULL_OPENCV` | fx fy cx cy k1 k2 p1 p2 k3 0 0 0 |
| `radtan8` | `FULL_OPENCV` | fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6 |
| `equid4` | `OPENCV_FISHEYE` | fx fy cx cy k1 k2 k3 k4 |
| `unknown` | `OPENCV` | none: COLMAP starts from its own guess and always refines the intrinsics |

With `optimize_intrinsics: 1` every distortion coefficient of the model is refined; on short sequences the
eight coefficients of `radtan8` can drift, so `optimize_intrinsics: 0` is the safer choice when a trusted
calibration exists.

## Settings yaml

`vslamlab_colmap_settings.yaml` holds COLMAP options by stage (`feature_extractor`, `matcher`, `mapper`,
`dense`), keys written as `<Section>_<param>` and forwarded as `--<Section>.<param> <value>` to the commands
of that stage. Each key is filtered against the command's own option list, so `Mapper_*` keys reach the
incremental mapper and `GlobalMapper_*` keys GLOMAP; a key no command of the stage knows is reported and
ignored. Options that the experiment parameters control (`ba_refine_*`, `use_gpu`, `gpu_index`,
`num_threads`, `loop_detection`, `vocab_tree_path`, the dense image sizes) are not read from the file. The
shipped values are COLMAP 4.2.1 defaults; any other COLMAP option can be added as a new key. This is the
file a VSLAM-LAB `Ablation:` csv edits per run (`<section>.<key>` columns).

## Outputs per run

Inside the experiment's sequence folder (`<evaluation>/<experiment>/<DATASET>/<sequence>/`):

| File | Content |
|---|---|
| `<exp_id>_KeyFrameTrajectory.csv` | the trajectory VSLAM-LAB evaluates (`ts (ns), tx, ty, tz, qx, qy, qz, qw`) |
| `<exp_id>_profiling.csv` | one row per COLMAP command (`stage, step, start (s), duration (s), status, peak RAM (GB), peak GPU (GB)`) plus per-stage totals; a stage summary is also printed at the end of `system_output_<exp_id>.txt` |
| `<exp_id>_run_report.yaml` | the run in one readable file (`run_report.py`): configuration (VPR, features, thresholds, verification, mapper), input images and resolution, pair counts, registration coverage, runtimes, peak memory, hardware |
| `<exp_id>_dense.ply`, `<exp_id>_mesh.ply` | fused point cloud and mesh when `dense: 1` |
| `colmap_<exp_id>/` | database, image list, sub-models `0/`, `1/`, ... (`best_model` names the one used), TXT export, `dense/` workspace (`fused.ply`, `meshed-*.ply`, `sparse_fused/` for the gui) |

## Model cache and offline nodes

COLMAP downloads the ONNX models (ALIKED, LightGlue, LoMa) and the FAISS vocabulary trees for sequential
loop detection on first use into `~/.cache/colmap` (the LoMa descriptor alone is 1.2 GB). On a machine
without internet, pre-fill the cache beforehand with the same HOME the jobs will see:

```
pixi run download-colmap-models            # everything the installed colmap may fetch (bf16 variants excluded)
pixi run download-colmap-models --list     # inventory and cache state only
```

`download_colmap_models.py` reads the resource list out of the installed `colmap` binary, so it follows the
installed version.

## Requirements and notes

- COLMAP 4.2.1 from conda-forge; on linux-64 the CUDA build plus `libcudnn` (the ONNX CUDA provider needs it)
  and `libopenimageio` 3.1 (linked by the package without being declared). All of this is in the main
  repository's `pixi.toml`.
- Dense reconstruction (`patch_match_stereo`) needs the CUDA build; the macOS package is CPU-only and skips it.
- Known issue: with GLOMAP the ALIKED + LightGlue runs on eth `table_3` register every image but join two
  internally accurate segments with a wrong transform
  ([VSLAM-LAB#164](https://github.com/VSLAM-LAB/VSLAM-LAB/issues/164)); the incremental mapper is unaffected.
