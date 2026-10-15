"""
Module: VSLAM-LAB - Baselines - colmap - colmap_matcher.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5.1)
- Version: 2.0
- Created: 2024-07-12
- Updated: 2026-10-03
- License: GPLv3 License

Feature extraction and matching stage of the colmap pipeline (Python port of colmap_matcher.sh):
creates the COLMAP database, extracts features for the frames listed in the experiment's rgb csv
(optionally masked) and matches them exhaustively or sequentially. Called by vslamlab_colmap.py.

matcher_type 'hybrid': exhaustive SIFT matching for every pair, then hybrid_matcher.py adds
learned-feature matches (hybrid_features, hybrid_matching) for VPR-selected cross-session pairs (run in the colmap-lightglue env), then
`colmap geometric_verifier` verifies the combined matches (see run_hybrid).
"""

import json
from pathlib import Path

from colmap_utilities import COLMAP_DIR, PROFILER, Settings, colmap_camera, detect_gpus, run, settings_args
from create_colmap_image_list import create_colmap_image_list
from create_colmap_mask_dir import create_colmap_mask_dir

# matching_type -> (FeatureExtraction.type, FeatureMatching.type), named <extractor>_<matcher>.
# ALIKED and LoMa are ONNX models that COLMAP downloads once into ~/.cache/colmap. LoMa (ECCV26,
# DeDoDe architecture): LOMA_B is a 256-dim descriptor with dedicated matchers B (LightGlue-sized),
# R (rotation-augmented), L and G (larger, slower, more accurate); LOMA_B128 is a lighter 128-dim
# descriptor with its own B128 matcher; LOMA_BRUTEFORCE (cosine similarity) takes either.
MATCHING_TYPES: dict[str, tuple[str, str]] = {
    'sift_bruteforce': ('SIFT', 'SIFT_BRUTEFORCE'),
    'sift_lightglue': ('SIFT', 'SIFT_LIGHTGLUE'),
    'aliked_bruteforce': ('ALIKED_N16ROT', 'ALIKED_BRUTEFORCE'),
    'aliked_lightglue': ('ALIKED_N16ROT', 'ALIKED_LIGHTGLUE'),
    'loma_b_bruteforce': ('LOMA_B', 'LOMA_BRUTEFORCE'),
    'loma_b_loma': ('LOMA_B', 'LOMA_B'),
    'loma_b_loma_r': ('LOMA_B', 'LOMA_R'),
    'loma_b_loma_l': ('LOMA_B', 'LOMA_L'),
    'loma_b_loma_g': ('LOMA_B', 'LOMA_G'),
    'loma_b128_bruteforce': ('LOMA_B128', 'LOMA_BRUTEFORCE'),
    'loma_b128_loma': ('LOMA_B128', 'LOMA_B128'),
}


def run_matcher(sequence_path: Path, exp_folder_colmap: Path, rgb_path: Path, rgb_csv: Path, calibration_yaml: Path,
                camera_name: str, matcher_type: str, matching_type: str, use_gpu: int, use_mask: int,
                settings: Settings | None = None, single_camera_per_folder: int = 1,
                hybrid: dict | None = None, camera_model: str = "") -> Path:
    """Build <exp_folder_colmap>/colmap_database.db with features and matches; returns its path.
    settings: the [feature_extractor] / [matcher] sections of the settings yaml (colmap_utilities.load_settings).
    single_camera_per_folder: 1 -> one camera per image folder (COLMAP resolves symlinks, so combined multi-session
    sequences get one camera per session); 0 -> one camera for every image (the original hybrid pipeline's setting).
    hybrid: options of matcher_type 'hybrid' (see run_hybrid).
    camera_model: COLMAP camera model for an 'unknown' calibration ('' -> OPENCV); a known calibration sets its own."""
    settings = settings or {}
    print("\nExecuting colmap_matcher ...")
    PROFILER.set_stage("extraction")

    if matching_type not in MATCHING_TYPES:
        raise SystemExit(f"Unknown matching_type: {matching_type}")
    feature_extraction_type, feature_matching_type = MATCHING_TYPES[matching_type]

    gpu_index_list, num_threads = detect_gpus(use_gpu)

    calibration_model, colmap_camera_model, camera_params = colmap_camera(calibration_yaml, camera_name)
    if camera_model:
        if calibration_model != "unknown":
            raise SystemExit(f"camera_model {camera_model} needs an 'unknown' calibration; {calibration_yaml} has {calibration_model}")
        colmap_camera_model = camera_model

    colmap_image_list = exp_folder_colmap / "colmap_image_list.txt"
    create_colmap_image_list(str(rgb_csv), str(colmap_image_list), camera_name)

    database = exp_folder_colmap / "colmap_database.db"
    database.unlink(missing_ok=True)
    run(["colmap", "database_creator", "--database_path", str(database)])

    print(f"    colmap feature_extractor ({feature_extraction_type}) ...")
    print(f"        gpu_index: {gpu_index_list}")
    print(f"        camera model : {calibration_model} (colmap: {colmap_camera_model})")
    camera_params_args: list[str] = []
    if camera_params:
        print(f"        camera params: {camera_params}")
        camera_params_args = ["--ImageReader.camera_params", camera_params]

    # Masks (use_mask=1): the rgb csv's path_mask_<i> column (1 = usable pixel, 0 = masked out - no
    # features are extracted where the mask is 0). One shared mask (refrax) goes through
    # --ImageReader.camera_mask_path, per-frame masks (mask2former, datasets shipping masks) through
    # a directory of <image name>.png symlinks and --ImageReader.mask_path.
    mask_args: list[str] = []
    if use_mask == 1:
        mask_spec = create_colmap_mask_dir(str(rgb_csv), camera_name, str(sequence_path), str(exp_folder_colmap / "colmap_masks"))
        if mask_spec.startswith("camera_mask:"):
            mask_args = ["--ImageReader.camera_mask_path", mask_spec[len("camera_mask:"):]]
            print(f"        mask: shared camera mask {mask_spec[len('camera_mask:'):]}")
        elif mask_spec.startswith("mask_dir:"):
            mask_args = ["--ImageReader.mask_path", mask_spec[len("mask_dir:"):]]
            print(f"        mask: per-frame masks in {mask_spec[len('mask_dir:'):]}")
        else:
            print(f"        mask: none available for {camera_name} in {rgb_csv}")
    else:
        print(f"        mask: disabled (use_mask={use_mask})")

    extractor_explicit = ["ImageReader.camera_model", "ImageReader.single_camera", "ImageReader.single_camera_per_folder",
                          "ImageReader.camera_params", "ImageReader.camera_mask_path", "ImageReader.mask_path",
                          "FeatureExtraction.type", "FeatureExtraction.use_gpu", "FeatureExtraction.gpu_index",
                          "FeatureExtraction.num_threads"]
    run(["colmap", "feature_extractor",
         "--database_path", str(database),
         "--image_path", str(rgb_path),
         "--image_list_path", str(colmap_image_list),
         "--ImageReader.camera_model", colmap_camera_model,
         "--ImageReader.single_camera", "1",
         "--ImageReader.single_camera_per_folder", str(single_camera_per_folder),
         "--FeatureExtraction.type", feature_extraction_type,
         "--FeatureExtraction.use_gpu", str(use_gpu),
         "--FeatureExtraction.gpu_index", gpu_index_list,
         "--FeatureExtraction.num_threads", str(num_threads),
         *camera_params_args, *mask_args,
         *settings_args(settings, "feature_extractor", "feature_extractor", extractor_explicit)])

    matching_args = ["--database_path", str(database),
                     "--FeatureMatching.type", feature_matching_type,
                     "--FeatureMatching.use_gpu", str(use_gpu),
                     "--FeatureMatching.gpu_index", gpu_index_list,
                     "--FeatureMatching.num_threads", str(num_threads)]
    matcher_explicit = ["FeatureMatching.type", "FeatureMatching.use_gpu", "FeatureMatching.gpu_index", "FeatureMatching.num_threads",
                        "SequentialMatching.loop_detection", "SequentialMatching.vocab_tree_path"]

    PROFILER.set_stage("matching")
    if matcher_type == "hybrid" and feature_matching_type != "SIFT_BRUTEFORCE":
        raise SystemExit(f"matcher_type: hybrid needs matching_type: sift_bruteforce (got {matching_type})")
    if matcher_type in ("exhaustive", "hybrid"):
        print(f"    colmap exhaustive_matcher ({feature_matching_type}) ...")
        print(f"        gpu_index: {gpu_index_list}")
        run(["colmap", "exhaustive_matcher", *matching_args,
             *settings_args(settings, "matcher", "exhaustive_matcher", matcher_explicit)])
    elif matcher_type == "sequential":
        # Loop detection uses COLMAP's default vocabulary tree for the feature type (a FAISS index
        # per SIFT / ALIKED / LoMa, auto-downloaded once into ~/.cache/colmap). The legacy
        # flickr100K FLANN trees are rejected by COLMAP >= 3.12 ("Failed to read faiss index").
        print(f"    colmap sequential_matcher ({feature_matching_type}) ...")
        print(f"        Vocabulary Tree: COLMAP default for {feature_extraction_type} (cached in ~/.cache/colmap)")
        print(f"        gpu_index: {gpu_index_list}")
        run(["colmap", "sequential_matcher", *matching_args,
             "--SequentialMatching.loop_detection", "1",
             *settings_args(settings, "matcher", "sequential_matcher", matcher_explicit)])
    else:
        raise SystemExit(f"Unknown matcher_type: {matcher_type}")

    if matcher_type == "hybrid":
        run_hybrid(database, sequence_path, rgb_path, rgb_csv, camera_name, num_threads, hybrid or {}, settings)

    return database


# Two-view verification profiles for matcher_type 'hybrid'. 'glomap' mirrors the relative-pose stage of standalone
# GLOMAP 1.2.0 (the original hybrid pipeline's mapper): essential-matrix RANSAC with a 1 px error, 50k trials, 0.9999
# confidence, then pairs with < 30 inliers or an inlier ratio < 0.25 dropped (glomap relpose_estimation.h,
# relpose_filter.h, poselib RansacOptions defaults). 'colmap' is COLMAP 4.2.1's default verification.
VERIFICATION_PROFILES: dict[str, dict[str, str]] = {
    "colmap": {},
    "glomap": {"TwoViewGeometry.max_error": "1", "TwoViewGeometry.max_num_trials": "50000",
               "TwoViewGeometry.confidence": "0.9999", "TwoViewGeometry.min_num_inliers": "30",
               "TwoViewGeometry.min_inlier_ratio": "0.25"},
}


def resolve_sequence_file(sequence_path: Path, value: str, default: str) -> Path:
    """An experiment path parameter: '' -> the default inside the sequence folder, relative -> inside the
    sequence folder, absolute -> as given."""
    path = Path(value) if value else Path(default)
    return path if path.is_absolute() else sequence_path / path


def run_hybrid(database: Path, sequence_path: Path, rgb_path: Path, rgb_csv: Path, camera_name: str, num_threads: int,
               hybrid: dict, settings: Settings) -> None:
    """Hybrid matching on a database holding exhaustive SIFT features + raw matches:
    1. hybrid_matcher.py (colmap-lightglue env): hybrid_features matched with hybrid_matching on the selected pairs,
       combined with raw SIFT.
    2. verification 'colmap' / 'glomap': `colmap geometric_verifier` with that profile. 'none' keeps the original
       pipeline's all-inlier geometries, which COLMAP 4.x's global mapper (mapper_type: glomap) cannot handle (it no longer
       re-estimates relative poses, so outliers would reach rotation averaging); use it with the incremental mapper."""
    verification = hybrid.get("verification", "colmap")
    if verification not in ("colmap", "glomap", "none"):
        raise SystemExit(f"Unknown verification: {verification}")
    vpr_matrix = resolve_sequence_file(sequence_path, hybrid.get("vpr_matrix", ""), "vpr-lab/D.npy")
    vpr_rgb_csv = resolve_sequence_file(sequence_path, hybrid.get("vpr_rgb_csv", ""), "rgb.csv")

    features, matching = hybrid.get("hybrid_features", "superpoint"), hybrid.get("hybrid_matching", "lightglue")
    print(f"    hybrid matcher ({features} + {matching}) ...")
    print(f"        pair_selection: {hybrid.get('pair_selection', 'vpr')}, vpr_threshold: {hybrid.get('vpr_threshold', 0.5)}")
    print(f"        vpr_matrix: {vpr_matrix}  (rows: {vpr_rgb_csv})")
    print(f"        lg_min_matches: {hybrid.get('lg_min_matches', 40)}, min_matches: {hybrid.get('min_matches', 10)}, verification: {verification}")

    # The matcher runs in its own env (torch + LightGlue at pinned versions); the colmap env has no torch.
    pixi = ["pixi", "run", "--frozen", "--manifest-path", str(COLMAP_DIR.parents[1] / "pixi.toml"), "-e", "colmap-lightglue"]
    run([*pixi, "install-lightglue"], profile=False)
    run([*pixi, "python", str(COLMAP_DIR / "hybrid_matcher.py"),
         "--database", str(database), "--rgb_path", str(rgb_path), "--rgb_csv", str(rgb_csv), "--camera_name", camera_name,
         "--features", features, "--matching", matching, "--pair_selection", str(hybrid.get("pair_selection", "vpr")),
         "--vpr_matrix", str(vpr_matrix), "--vpr_rgb_csv", str(vpr_rgb_csv),
         "--vpr_threshold", str(hybrid.get("vpr_threshold", 0.5)),
         "--lg_min_matches", str(hybrid.get("lg_min_matches", 40)), "--min_matches", str(hybrid.get("min_matches", 10)),
         "--verification", verification], step="hybrid_matcher")
    # Profile only the matcher's processing time (time_total_s), not pixi start-up and the torch / LightGlue imports
    stats_json = database.parent / "hybrid_matcher_stats.json"
    if stats_json.is_file():
        PROFILER.set_last_duration(json.loads(stats_json.read_text())["time_total_s"])

    if verification == "none":
        print("        verification: none (all matches stored as inliers, as in the original pipeline)")
        return
    profile = VERIFICATION_PROFILES[verification]
    seed = int(hybrid.get("random_seed", -1))
    explicit = ["TwoViewGeometry.random_seed", "num_threads", *profile]
    print(f"    colmap geometric_verifier (profile: {verification}) ...")
    run(["colmap", "geometric_verifier", "--database_path", str(database), "--num_threads", str(num_threads),
         "--TwoViewGeometry.random_seed", str(seed),
         *[tok for option, value in profile.items() for tok in (f"--{option}", value)],
         *settings_args(settings, "matcher", "geometric_verifier", explicit)])
