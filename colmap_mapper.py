"""
Module: VSLAM-LAB - Baselines - colmap - colmap_mapper.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5.1)
- Version: 2.0
- Created: 2024-07-12
- Updated: 2026-10-03
- License: GPLv3 License

Mapping stage of the colmap pipeline (Python port of colmap_mapper.sh): runs COLMAP's incremental
mapper or GLOMAP's global mapper on the database, picks the sub-model with the most registered
images and exports it as TXT into the colmap folder. Called by vslamlab_colmap.py.
init_pair 'cross_session' starts the incremental mapper from the cross-session image pair with the most verified
matches (cross_session_init_pair).
"""

import os
import re
import sqlite3
from pathlib import Path

import pandas as pd

from colmap_utilities import PROFILER, Settings, colmap_camera, run, settings_args

MAX_NUM_IMAGES = 2147483647  # COLMAP's pair_id = image_id1 * MAX_NUM_IMAGES + image_id2


def registered_images(model_dir: Path) -> int:
    out = run(["colmap", "model_analyzer", "--path", str(model_dir)], capture=True)
    match = re.search(r"Registered images: (\d+)", out)
    return int(match.group(1)) if match else 0


def select_best_model(exp_folder_colmap: Path) -> tuple[Path, int]:
    """COLMAP may split the scene into several sub-models (<exp_folder_colmap>/0, /1, ...), and the
    largest one is not necessarily /0. Returns the sub-model with the most registered images and
    records the choice in <exp_folder_colmap>/best_model for the rest of the pipeline (gui, ...)."""
    best_model, best_num_images = None, -1
    for model_dir in sorted(p for p in exp_folder_colmap.iterdir() if p.is_dir() and p.name.isdigit()):
        if not (model_dir / "images.bin").is_file():
            continue
        num_images = registered_images(model_dir)
        print(f"        sub-model {model_dir.name}: {num_images} registered images")
        if num_images > best_num_images:
            best_model, best_num_images = model_dir, num_images
    if best_model is None or best_num_images == 0:  # glomap writes an empty /0 when its pose graph is empty
        raise SystemExit(f"    no reconstruction produced (no sub-model with registered images in {exp_folder_colmap})")
    (exp_folder_colmap / "best_model").write_text(f"{best_model}\n")
    return best_model, best_num_images


def cross_session_init_pair(database: Path, rgb_csv: Path, camera_name: str) -> tuple[int, int, int]:
    """(image_id1, image_id2, inliers) of the verified image pair from two different sessions (rgb csv's sequence_name)
    with the most inlier matches. Why: in a hybrid run the learned keypoints are matched only across sessions, so they
    are triangulated only once a model holds two sessions; a model started inside one session cannot register the
    other sessions through them and the reconstruction fragments."""
    df = pd.read_csv(rgb_csv)
    if "sequence_name" not in df.columns:
        raise SystemExit(f"    init_pair cross_session: {rgb_csv} has no sequence_name column")
    session = dict(zip((os.path.basename(str(p)) for p in df[f"path_{camera_name}"]), df["sequence_name"].astype(str)))
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as conn:
        names = {i: os.path.basename(n) for i, n in conn.execute("SELECT image_id, name FROM images")}
        best = (0, 0, 0)
        for pair_id, rows in conn.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0"):
            id2 = pair_id % MAX_NUM_IMAGES
            id1 = (pair_id - id2) // MAX_NUM_IMAGES
            if session[names[id1]] != session[names[id2]] and rows > best[2]:
                best = (id1, id2, rows)
    if best[2] == 0:
        raise SystemExit("    init_pair cross_session: no verified cross-session image pair in the database")
    return best


def run_mapper(exp_folder_colmap: Path, rgb_path: Path, calibration_yaml: Path, camera_name: str,
               mapper_type: str, optimize_intrinsics: int, settings: Settings | None = None,
               random_seed: int = -1, init_pair: str = "colmap", rgb_csv: Path | None = None) -> Path:
    """Reconstruct from <exp_folder_colmap>/colmap_database.db; returns the best sub-model folder.
    settings: the [mapper] section of the settings yaml (Mapper_* keys reach mapper, GlobalMapper_* keys glomap).
    init_pair: 'colmap' (COLMAP chooses the initial pair) or 'cross_session' (incremental mapper only: the
    cross-session pair with the most verified matches, see cross_session_init_pair; needs rgb_csv)."""
    print("Executing colmap_mapper ...")
    settings = settings or {}
    PROFILER.set_stage("reconstruction")

    calibration_model, _, _ = colmap_camera(calibration_yaml, camera_name)
    print(f"        camera model : {calibration_model}")

    # optimize_intrinsics (0/1, default 1): whether bundle adjustment refines the camera intrinsics.
    # With 1, focal length and distortion (extra params) are refined and the principal point stays
    # fixed, matching COLMAP's own defaults; with 0 the intrinsics from the calibration yaml are kept
    # as given. An 'unknown' calibration model has no intrinsics to keep (the matcher started COLMAP
    # from a guess), so it always refines regardless of the flag.
    if calibration_model == "unknown" and optimize_intrinsics != 1:
        print(f"        WARNING: optimize_intrinsics={optimize_intrinsics} ignored: camera model is 'unknown' (no intrinsics to keep fixed), refining intrinsics")
        optimize_intrinsics = 1
    refine = "1" if optimize_intrinsics == 1 else "0"
    ba_refine_focal_length, ba_refine_principal_point, ba_refine_extra_params = refine, "0", refine
    print(f"        optimize_intrinsics: {optimize_intrinsics} (ba_refine_focal_length={ba_refine_focal_length}, "
          f"ba_refine_principal_point={ba_refine_principal_point}, ba_refine_extra_params={ba_refine_extra_params})")

    database = exp_folder_colmap / "colmap_database.db"
    if mapper_type == "glomap":
        print("    global mapper (GLOMAP) ...")
        command, prefix = "global_mapper", "GlobalMapper"
    else:
        print("    colmap mapper (COLMAP) ...")
        command, prefix = "mapper", "Mapper"
    # random_seed: -1 keeps COLMAP's default (non-deterministic); >= 0 makes reruns repeatable
    explicit = [f"{prefix}.ba_refine_focal_length", f"{prefix}.ba_refine_principal_point", f"{prefix}.ba_refine_extra_params",
                f"{prefix}.random_seed"]
    init_args: list[str] = []
    if init_pair == "cross_session" and mapper_type != "glomap":
        id1, id2, inliers = cross_session_init_pair(database, rgb_csv, camera_name)
        print(f"        initial pair: images {id1} and {id2} (cross-session, {inliers} verified matches)")
        init_args = [f"--{prefix}.init_image_id1", str(id1), f"--{prefix}.init_image_id2", str(id2)]
        explicit += [f"{prefix}.init_image_id1", f"{prefix}.init_image_id2"]
    run(["colmap", command,
         "--database_path", str(database),
         "--image_path", str(rgb_path),
         "--output_path", str(exp_folder_colmap),
         f"--{prefix}.ba_refine_focal_length", ba_refine_focal_length,
         f"--{prefix}.ba_refine_principal_point", ba_refine_principal_point,
         f"--{prefix}.ba_refine_extra_params", ba_refine_extra_params,
         f"--{prefix}.random_seed", str(random_seed),
         *init_args,
         *settings_args(settings, "mapper", command, explicit)])

    best_model, best_num_images = select_best_model(exp_folder_colmap)

    print(f"    colmap model_converter (sub-model {best_model.name}, {best_num_images} images) ...")
    run(["colmap", "model_converter", "--input_path", str(best_model), "--output_path", str(exp_folder_colmap), "--output_type", "TXT"])
    return best_model
