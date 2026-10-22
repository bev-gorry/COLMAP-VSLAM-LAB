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
import shutil
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
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


TWO_VIEW_CALIBRATED = 2  # COLMAP TwoViewGeometry config with an essential matrix (known intrinsics)
INIT_ANGLE_MARGIN = 1.25  # initial pair: median triangulation angle >= 1.25 x Mapper.init_min_tri_angle


def undistorted_rays(uv: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Unit viewing rays of COLMAP keypoints (pixel-centre convention, +0.5) for an OPENCV/PINHOLE camera
    (f, fy, cx, cy[, k1, k2, p1, p2]); the radial-tangential distortion is inverted by fixed-point iteration."""
    f, fy, cx, cy = params[:4]
    k1, k2, p1, p2 = (list(params[4:8]) + [0.0] * 4)[:4]
    xd, yd = (uv[:, 0] - 0.5 - cx) / f, (uv[:, 1] - 0.5 - cy) / fy
    x, y = xd.copy(), yd.copy()
    for _ in range(20):
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2
        x = (xd - (2 * p1 * x * y + p2 * (r2 + 2 * x * x))) / radial
        y = (yd - (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)) / radial
    rays = np.c_[x, y, np.ones_like(x)]
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def median_triangulation_angle(E: np.ndarray, rays1: np.ndarray, rays2: np.ndarray) -> float:
    """Median angle (deg) between the two viewing rays of each match, with the relative pose decomposed from the
    pair's essential matrix (the decomposition with most points in front of both cameras). This is the parallax
    COLMAP's initialisation requires to be at least Mapper.init_min_tri_angle."""
    U, _, Vt = np.linalg.svd(E)
    U, Vt = U * np.sign(np.linalg.det(U)), Vt * np.sign(np.linalg.det(Vt))
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    best = (-1, 0.0)
    for R in (U @ W @ Vt, U @ W.T @ Vt):
        for t in (U[:, 2], -U[:, 2]):
            # camera 2 = R X + t; camera 2's centre and rays in camera 1's frame
            c2, d2 = -R.T @ t, rays2 @ R
            # depths s1, s2 of the closest points along both rays (midpoint triangulation)
            a12 = (rays1 * d2).sum(1)
            denom = 1 - a12 ** 2
            ok = denom > 1e-12
            s1 = np.where(ok, (rays1 @ c2 - a12 * (d2 @ c2)) / np.where(ok, denom, 1), 0)
            s2 = np.where(ok, (a12 * (rays1 @ c2) - d2 @ c2) / np.where(ok, denom, 1), 0)
            front = (s1 > 0) & (s2 > 0)
            if front.sum() > best[0]:
                X = s1[:, None] * rays1
                v2 = X - c2
                cos = (rays1 * v2).sum(1) / np.maximum(np.linalg.norm(v2, axis=1), 1e-12)
                angles = np.degrees(np.arccos(np.clip(cos, -1, 1)))[front]
                best = (int(front.sum()), float(np.median(angles)) if angles.size else 0.0)
    return best[1]


def cross_session_init_pair(database: Path, rgb_csv: Path, camera_name: str, min_tri_angle: float = 16.0,
                            min_num_inliers: int = 100) -> tuple[int, int, int, float] | None:
    """(image_id1, image_id2, inliers, median triangulation angle) of the initial pair: of the verified pairs from two
    different sessions (rgb csv's sequence_name) with an essential matrix, at least min_num_inliers inliers and a
    median triangulation angle of at least min_tri_angle (COLMAP's own initialisation thresholds), the one with the most
    inliers. Why cross-session: in a hybrid run the learned keypoints are matched only across sessions, so they are
    triangulated only once a model holds two sessions; a model started inside one session cannot register the other
    sessions through them and the reconstruction fragments. Why the angle: the pairs with the most matches are often
    near-identical viewpoints, which COLMAP rejects (or accepts only after relaxing its thresholds).
    None if no pair qualifies."""
    df = pd.read_csv(rgb_csv)
    if "sequence_name" not in df.columns:
        raise SystemExit(f"    init_pair cross_session: {rgb_csv} has no sequence_name column")
    session = dict(zip((os.path.basename(str(p)) for p in df[f"path_{camera_name}"]), df["sequence_name"].astype(str)))
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as conn:
        names = {i: os.path.basename(n) for i, n in conn.execute("SELECT image_id, name FROM images")}
        camera_of = dict(conn.execute("SELECT image_id, camera_id FROM images"))
        params = {c: np.frombuffer(p, np.float64) for c, p in conn.execute("SELECT camera_id, params FROM cameras")}
        candidates = []
        for pair_id, rows, config in conn.execute("SELECT pair_id, rows, config FROM two_view_geometries WHERE rows > 0"):
            id2 = pair_id % MAX_NUM_IMAGES
            id1 = (pair_id - id2) // MAX_NUM_IMAGES
            if session[names[id1]] != session[names[id2]] and config == TWO_VIEW_CALIBRATED and rows >= min_num_inliers:
                candidates.append((rows, pair_id, id1, id2))
        keypoints: dict[int, np.ndarray] = {}

        def keypoints_of(image_id):
            if image_id not in keypoints:
                r, c, d = conn.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (image_id,)).fetchone()
                keypoints[image_id] = np.frombuffer(d, np.float32).reshape(r, c)[:, :2].astype(np.float64)
            return keypoints[image_id]

        for rows, pair_id, id1, id2 in sorted(candidates, reverse=True):
            data, E = conn.execute("SELECT data, E FROM two_view_geometries WHERE pair_id = ?", (pair_id,)).fetchone()
            matches = np.frombuffer(data, np.uint32).reshape(rows, 2)
            rays1 = undistorted_rays(keypoints_of(id1)[matches[:, 0]], params[camera_of[id1]])
            rays2 = undistorted_rays(keypoints_of(id2)[matches[:, 1]], params[camera_of[id2]])
            angle = median_triangulation_angle(np.frombuffer(E, np.float64).reshape(3, 3), rays1, rays2)
            if angle >= min_tri_angle:
                return id1, id2, rows, angle
    return None


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
        mapper_settings = settings.get("mapper", {})
        # 1.25 x COLMAP's own minimum: the angle here is estimated from the stored essential matrix and differs from
        # COLMAP's by up to ~3 deg, and a pair COLMAP rejects is retried only after relaxing its thresholds
        min_tri_angle = INIT_ANGLE_MARGIN * float(mapper_settings.get("Mapper.init_min_tri_angle", 16))
        pair = cross_session_init_pair(database, rgb_csv, camera_name, min_tri_angle,
                                       int(mapper_settings.get("Mapper.init_min_num_inliers", 100)))
        if pair is None:
            print(f"        WARNING: no cross-session pair with a median triangulation angle >= {min_tri_angle:.0f} deg; "
                  f"COLMAP chooses the initial pair")
        else:
            id1, id2, inliers, angle = pair
            print(f"        initial pair: images {id1} and {id2} (cross-session, {inliers} verified matches, "
                  f"median triangulation angle {angle:.1f} deg)")
            init_args = [f"--{prefix}.init_image_id1", str(id1), f"--{prefix}.init_image_id2", str(id2)]
            explicit += [f"{prefix}.init_image_id1", f"{prefix}.init_image_id2"]
    base = ["colmap", command,
            "--database_path", str(database),
            "--image_path", str(rgb_path),
            "--output_path", str(exp_folder_colmap),
            f"--{prefix}.ba_refine_focal_length", ba_refine_focal_length,
            f"--{prefix}.ba_refine_principal_point", ba_refine_principal_point,
            f"--{prefix}.ba_refine_extra_params", ba_refine_extra_params,
            f"--{prefix}.random_seed", str(random_seed)]
    try:
        run([*base, *init_args, *settings_args(settings, "mapper", command, explicit)], exit_on_error=not init_args)
    except subprocess.CalledProcessError:
        # Only with a given initial pair: COLMAP could not start from it, so start again from COLMAP's own choice
        print("        WARNING: the mapper failed from the given initial pair; retrying with COLMAP's own choice")
        for model_dir in (p for p in exp_folder_colmap.iterdir() if p.is_dir() and p.name.isdigit()):
            shutil.rmtree(model_dir)
        run([*base, *settings_args(settings, "mapper", command, explicit[:4])])

    best_model, best_num_images = select_best_model(exp_folder_colmap)

    print(f"    colmap model_converter (sub-model {best_model.name}, {best_num_images} images) ...")
    run(["colmap", "model_converter", "--input_path", str(best_model), "--output_path", str(exp_folder_colmap), "--output_type", "TXT"])
    return best_model
