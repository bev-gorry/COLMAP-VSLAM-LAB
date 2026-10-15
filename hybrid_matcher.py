"""
Module: VSLAM-LAB - Baselines - colmap - hybrid_matcher.py
- Author: Beverley Gorry (port of feature_matcher.py from the original colmap-vslamlab fork @ d5f8541)
- Assisted by: Claude (Opus 5.5)
- Version: 1.0
- Created: 2026-10-09
- Updated: 2026-10-09
- License: GPLv3 License

Hybrid matcher (matcher_type: hybrid, see colmap_matcher.run_matcher). Adds learned-feature cross-session
correspondences to a COLMAP database that already holds exhaustive SIFT features and raw SIFT matches.
Runs in the colmap-lightglue pixi env (torch + LightGlue) as a subprocess of the colmap env.

Steps - identical to the original feature_matcher.py unless marked [port]:
  1. Read the SIFT keypoints and the raw (unverified) SIFT matches written by `colmap exhaustive_matcher`.
  2. Pair selection (--pair_selection):
       vpr            MegaLoc distance matrix, min-max normalised over the whole matrix, self and
                      same-session pairs masked, kept if normalised distance <= vpr_threshold
                      (the original scripts/clean_distance_matrix.py + apply_distance_threshold.py)
       cross_session  every pair of images from different sessions       ("SIFT + LightGlue (different sessions)")
       all            every pair                                          ("SIFT + LightGlue (all)", the original no-VPR setting)
     [port] matrix rows are looked up by image name instead of by position in the database.
  3. Features (--features, 2048 keypoints) of every image: SuperPoint (original) or [port] ALIKED at 0/90/180/270
     degree rotations, as neither is rotation invariant; [port] SIFT once, as it is.
  4. Matching (--matching: LightGlue, or [port] brute force) on each selected pair: best relative rotation plus the
     3 consistent rotation pairs, kept only when it returns more than lg_min_matches matches.
  5. Rewrite the database: keypoints = SIFT then un-rotated hybrid keypoints; matches = raw SIFT + hybrid matches,
     for every pair with at least min_matches matches.
  6. Two-view geometries (--verification):
       none           every match written as an inlier with config CALIBRATED and no geometry, exactly as
                      the original pipeline did (its GLOMAP 1.2.0 mapper then re-estimated relative poses with RANSAC)
       colmap/glomap  [port] the table is cleared and colmap_matcher.py runs `colmap geometric_verifier`
                      afterwards, because COLMAP 4.x's global mapper no longer re-estimates relative poses
Writes hybrid_matcher_stats.json (counts, timings, parameters, feature settings, versions) and vpr_pairs.npy next to the database.
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from kornia.feature import match_mnn, match_smnn
from tqdm import tqdm

from lightglue import ALIKED, SIFT, LightGlue, SuperPoint
from lightglue.utils import load_image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_matcher_utilities import feature_matching, lightglue_matching, unrotate_kps_W  # noqa: E402

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "

# COLMAP's kMaxNumImages (util/types.h): pair_id = min_id * MAX_NUM_IMAGES + max_id
MAX_NUM_IMAGES = 2147483647
# COLMAP TwoViewGeometry::CALIBRATED (scene/two_view_geometry.h), what the original pipeline wrote for every pair
CONFIG_CALIBRATED = 2
# COLMAP FeatureExtractorType::UNDEFINED (feature/types.h): these descriptors are placeholders, never matched
DESCRIPTOR_TYPE_UNDEFINED = -1
SIFT_DESCRIPTOR_DIM = 128
MAX_NUM_KEYPOINTS = 2048  # per rotation, as in the original pipeline
# --features -> (LightGlue extractor, rotations k of k * 90 degrees). Rotation-invariant features are extracted once.
FEATURES = {"superpoint": (SuperPoint, [0, 1, 2, 3]), "aliked": (ALIKED, [0, 1, 2, 3]), "sift": (SIFT, [0])}


def pair_id_of(image_id1: int, image_id2: int) -> int:
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_NUM_IMAGES + image_id2


# ---- Database I/O ----------------------------------------------------------------------------------------------------

def read_images(cursor) -> dict[int, str]:
    cursor.execute("SELECT image_id, name FROM images")
    return {image_id: name for image_id, name in cursor.fetchall()}


def read_keypoints(cursor) -> dict[int, np.ndarray]:
    cursor.execute("SELECT image_id, rows, cols, data FROM keypoints")
    keypoints = {}
    for image_id, rows, cols, data in cursor.fetchall():
        keypoints[image_id] = np.frombuffer(data, dtype=np.float32).reshape((rows, cols)) if rows > 0 \
            else np.zeros((0, cols), dtype=np.float32)
    return keypoints


def read_raw_matches(cursor) -> dict[int, np.ndarray]:
    """Raw (unverified) matches per pair_id, as the original pipeline read them (the `matches` table, not `two_view_geometries`)."""
    cursor.execute("SELECT pair_id, rows, data FROM matches")
    matches = {}
    for pair_id, rows, data in cursor.fetchall():
        matches[pair_id] = np.frombuffer(data, dtype=np.uint32).reshape(-1, 2) if (data is not None and rows > 0) \
            else np.zeros((0, 2), dtype=np.uint32)
    return matches


def insert_keypoints(cursor, image_id: int, keypoints: np.ndarray) -> None:
    keypoints = np.ascontiguousarray(keypoints, dtype=np.float32)
    cursor.execute("INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?)",
                   (image_id, keypoints.shape[0], keypoints.shape[1], keypoints.tobytes()))
    # Placeholder descriptors: matching is finished, nothing reads them. [port] uint8 so the blob size is what
    # COLMAP 4.x expects (the original pipeline wrote float32 zeros, which COLMAP 4.x would reject if it ever read them).
    descriptors = np.zeros((keypoints.shape[0], SIFT_DESCRIPTOR_DIM), dtype=np.uint8)
    cursor.execute("INSERT INTO descriptors(image_id, type, rows, cols, data) VALUES(?, ?, ?, ?, ?)",
                   (image_id, DESCRIPTOR_TYPE_UNDEFINED, descriptors.shape[0], descriptors.shape[1], descriptors.tobytes()))


def insert_matches(cursor, image_id1: int, image_id2: int, matches: np.ndarray) -> None:
    matches = np.ascontiguousarray(matches, dtype=np.uint32)
    cursor.execute("INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?)",
                   (pair_id_of(image_id1, image_id2), matches.shape[0], matches.shape[1], matches.tobytes()))


def insert_all_inlier_two_view_geometry(cursor, image_id1: int, image_id2: int, matches: np.ndarray) -> None:
    """The original insert_all_inlier_two_view_geometry: every match is an inlier, config CALIBRATED, no F/E/H/pose."""
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
        matches = matches[:, [1, 0]]
    matches = np.ascontiguousarray(matches, dtype=np.uint32)
    cursor.execute("INSERT OR REPLACE INTO two_view_geometries(pair_id, rows, cols, data, config) VALUES (?, ?, ?, ?, ?)",
                   (pair_id_of(image_id1, image_id2), matches.shape[0], matches.shape[1], matches.tobytes(), CONFIG_CALIBRATED))


# ---- Pair selection --------------------------------------------------------------------------------------------------

def vpr_pair_matrix(distance_matrix: np.ndarray, session_names: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean matrix of VPR candidate pairs, with exactly the original rule and float32 arithmetic:
    min-max normalise over the whole matrix (diagonal and same-session entries included), then drop the
    diagonal and same-session entries and keep normalised distances <= threshold."""
    D = distance_matrix.astype(np.float32, copy=True)
    D = (D - np.min(D)) / (np.max(D) - np.min(D))
    D[np.eye(D.shape[0], dtype=bool)] = np.inf
    D[session_names[:, None] == session_names[None, :]] = np.inf
    return np.isfinite(D) & (D <= threshold)


def image_rows_by_name(csv_path: Path, camera_name: str, need_sessions: bool) -> tuple[dict[str, int], np.ndarray | None]:
    """Row index of every frame in a VSLAM-LAB rgb csv, keyed by file name, plus the session of each row
    (the csv's sequence_name column; required when need_sessions)."""
    df = pd.read_csv(csv_path)
    names = [os.path.basename(str(p)) for p in df[f"path_{camera_name}"]]
    if len(set(names)) != len(names):
        raise SystemExit(f"{SCRIPT_LABEL}image file names are not unique in {csv_path}; cannot index images by name")
    if "sequence_name" not in df.columns:
        if need_sessions:
            raise SystemExit(f"{SCRIPT_LABEL}{csv_path} has no sequence_name column; same-session pairs cannot be identified")
        return {name: row for row, name in enumerate(names)}, None
    return {name: row for row, name in enumerate(names)}, df["sequence_name"].astype(str).to_numpy()


# ---- Feature handling ------------------------------------------------------------------------------------------------

def extract_rotated(extractor, path: str, rotations: list[int], device: str):
    """hybrid_matcher_utilities.extract_keypoints for any LightGlue extractor (loaded once, not per image): features of
    the image rotated by k * 90 degrees for each k, plus every keypoint in rotated coordinates and the k it came from.
    Gives the same SuperPoint features as extract_keypoints."""
    image = load_image(path).to(device)
    _, h, w = image.shape
    feats = {k: extractor.extract(torch.rot90(image, k, dims=(1, 2))) for k in rotations}
    keypoints = torch.cat([f["keypoints"] for f in feats.values()], dim=1)
    rotation_of = torch.cat([torch.full(f["keypoints"].shape[:2], k, device=device) for k, f in feats.items()], dim=1)
    return {"keypoints": keypoints, "rotations": rotation_of}, feats, h, w


class BruteForce(torch.nn.Module):
    """Mutual nearest neighbours on descriptor distance, with Lowe's ratio test when ratio is set. Called with and
    returning the same dicts as LightGlue, so the rotation matching code runs either matcher unchanged."""

    def __init__(self, ratio: float | None = None) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, data: dict) -> dict:
        desc0, desc1 = data["image0"]["descriptors"][0], data["image1"]["descriptors"][0]
        if len(desc0) < 2 or len(desc1) < 2:
            return {"matches": [torch.zeros((0, 2), dtype=torch.long, device=desc0.device)]}
        _, matches = match_mnn(desc0, desc1) if self.ratio is None else match_smnn(desc0, desc1, th=self.ratio)
        return {"matches": [matches]}


def match_rotated(feats1: dict, feats2: dict, matcher) -> np.ndarray | None:
    """Matches between two images' features: over the four rotations as in the original pipeline
    (hybrid_matcher_utilities.feature_matching), or a single match for features extracted once."""
    if len(feats1) == 4:
        return feature_matching(feats1, feats2, matcher=matcher, exhaustive=True)
    matches = lightglue_matching(feats1[0], feats2[0], matcher=matcher)
    return matches.cpu().numpy().astype(np.uint32) if len(matches) else None


def feats_to(feats: dict, device: str) -> dict:
    """Move one image's per-rotation features (dict rotation -> feature dict) to a device.
    Values are unchanged; only where they live. Lets large datasets keep features in CPU memory."""
    return {k: {key: (val.to(device, non_blocking=True) if torch.is_tensor(val) else val) for key, val in f.items()}
            for k, f in feats.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid SIFT + learned-feature matcher")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--rgb_path", type=Path, required=True, help="folder COLMAP read the images from (--image_path)")
    parser.add_argument("--camera_name", type=str, default="rgb_0")
    parser.add_argument("--features", type=str, default="superpoint", choices=list(FEATURES),
                        help="features matched on the selected pairs")
    parser.add_argument("--matching", type=str, default="lightglue", choices=["lightglue", "bruteforce"])
    parser.add_argument("--pair_selection", type=str, default="vpr", choices=["vpr", "cross_session", "all"])
    parser.add_argument("--rgb_csv", type=Path, required=True, help="the experiment's rgb_exp.csv (sessions for cross_session)")
    parser.add_argument("--vpr_matrix", type=Path, default=None, help="MegaLoc distance matrix D.npy (pair_selection: vpr)")
    parser.add_argument("--vpr_rgb_csv", type=Path, default=None, help="rgb csv whose rows the VPR matrix follows")
    parser.add_argument("--vpr_threshold", type=float, default=0.5)
    parser.add_argument("--lg_min_matches", type=int, default=40, help="hybrid matches kept only if MORE than this")
    parser.add_argument("--min_matches", type=int, default=10, help="pairs with fewer SIFT + hybrid matches are dropped")
    parser.add_argument("--verification", type=str, default="colmap", choices=["colmap", "glomap", "none"])
    parser.add_argument("--features_on_gpu", type=str, default="auto", choices=["auto", "yes", "no"])
    args = parser.parse_args()

    t_start = time.perf_counter()
    stats: dict = {"parameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{SCRIPT_LABEL}torch {torch.__version__} (cuda {torch.version.cuda}), device: {device}")

    conn = sqlite3.connect(args.database)
    cursor = conn.cursor()
    images = read_images(cursor)
    image_ids = sorted(images.keys())
    image_names = {image_id: os.path.basename(images[image_id]) for image_id in image_ids}
    if len(set(image_names.values())) != len(image_ids):
        raise SystemExit(f"{SCRIPT_LABEL}image file names in the database are not unique")

    # 1. SIFT keypoints and raw SIFT matches written by colmap exhaustive_matcher
    sift_keypoints = read_keypoints(cursor)
    sift_matches = read_raw_matches(cursor)
    sift_cols = {kp.shape[1] for kp in sift_keypoints.values()}
    if sift_cols != {6}:
        raise SystemExit(f"{SCRIPT_LABEL}expected 6-column SIFT keypoints, found {sift_cols}")
    print(f"{SCRIPT_LABEL}{len(image_ids)} images, {sum(len(k) for k in sift_keypoints.values())} SIFT keypoints, "
          f"{len(sift_matches)} SIFT match pairs")

    # 2. Pair selection, as a boolean matrix over the database's image order
    num_images = len(image_ids)
    if args.pair_selection == "vpr":
        if args.vpr_matrix is None or not args.vpr_matrix.is_file():
            raise SystemExit(f"{SCRIPT_LABEL}VPR distance matrix not found: {args.vpr_matrix}")
        vpr_rgb_csv = args.vpr_rgb_csv or args.rgb_csv
        vpr_rows, vpr_sessions = image_rows_by_name(vpr_rgb_csv, args.camera_name, need_sessions=True)
        D = np.load(args.vpr_matrix)
        if D.shape != (len(vpr_rows), len(vpr_rows)):
            raise SystemExit(f"{SCRIPT_LABEL}VPR matrix {D.shape} does not match {len(vpr_rows)} rows of {vpr_rgb_csv}")
        missing = [n for n in image_names.values() if n not in vpr_rows]
        if missing:
            raise SystemExit(f"{SCRIPT_LABEL}{len(missing)} database images are not rows of {vpr_rgb_csv} (e.g. {missing[:3]})")
        selected_full = vpr_pair_matrix(D, vpr_sessions, args.vpr_threshold)
        rows = np.array([vpr_rows[image_names[image_id]] for image_id in image_ids])
        # The original pipeline used D[i, j] with i < j in database order (upper triangle); keep that orientation.
        selected = selected_full[np.ix_(rows, rows)]
        stats["vpr"] = {"matrix": str(args.vpr_matrix), "rgb_csv": str(vpr_rgb_csv), "matrix_shape": list(D.shape),
                        "threshold": args.vpr_threshold,
                        "database_order_equals_matrix_order": bool(np.array_equal(rows, np.arange(num_images)))}
    elif args.pair_selection == "cross_session":
        exp_rows, exp_sessions = image_rows_by_name(args.rgb_csv, args.camera_name, need_sessions=True)
        sessions = np.array([exp_sessions[exp_rows[image_names[image_id]]] for image_id in image_ids])
        selected = sessions[:, None] != sessions[None, :]
    else:
        selected = ~np.eye(num_images, dtype=bool)
    upper = np.triu_indices(num_images, 1)
    num_selected = int(selected[upper].sum())
    stats["selected_pairs"] = num_selected
    np.save(args.database.parent / "vpr_pairs.npy",
            np.array([(image_ids[i], image_ids[j]) for i, j in zip(*upper) if selected[i, j]], dtype=np.int64).reshape(-1, 2))
    print(f"{SCRIPT_LABEL}pair selection '{args.pair_selection}': {num_selected} pairs for {args.matching}")

    # 3. Features of every image, at the rotations the feature type needs.
    # Features stay on the GPU when all of them fit (as originally); otherwise each image's features go to CPU
    # memory right after extraction and are moved back per pair. Same values either way, only where they live.
    extractor_class, rotations = FEATURES[args.features]
    extractor = extractor_class(max_num_keypoints=MAX_NUM_KEYPOINTS).eval().to(device)
    t0 = time.perf_counter()
    feats, keypoints_rot, rotations_of, image_sizes = {}, {}, {}, {}
    keep_on_gpu, feat_bytes = device == "cuda", 0
    for n_done, image_id in enumerate(tqdm(image_ids, desc="Feature Extraction")):
        path = os.path.join(args.rgb_path, images[image_id])
        merged, feats_image, h, w = extract_rotated(extractor, path, rotations, device)
        bytes_image = sum(v.element_size() * v.nelement() for d in feats_image.values() for v in d.values() if torch.is_tensor(v))
        feat_bytes += bytes_image
        if n_done == 0 and device == "cuda":
            estimate = bytes_image * num_images
            keep_on_gpu = args.features_on_gpu == "yes" or \
                (args.features_on_gpu == "auto" and estimate < 0.6 * torch.cuda.mem_get_info()[0])
            print(f"{SCRIPT_LABEL}{args.features} features: ~{estimate / 1e9:.2f} GB, kept on {'GPU' if keep_on_gpu else 'CPU'}")
        feats[image_id] = feats_image if keep_on_gpu else feats_to(feats_image, "cpu")
        keypoints_rot[image_id] = merged["keypoints"].squeeze(0).cpu().numpy().astype(np.float32)
        rotations_of[image_id] = merged["rotations"].squeeze(0).cpu().numpy().astype(np.float32)
        image_sizes[image_id] = (h, w)
        del merged
    stats["time_extraction_s"] = time.perf_counter() - t0
    stats["features_gb"], stats["features_on_gpu"] = feat_bytes / 1e9, keep_on_gpu

    # 5a. Keypoints: SIFT first (so raw SIFT match indices stay valid), then the un-rotated hybrid keypoints.
    # The extra columns (1, 0, 1, 0) are the original placeholder shape values; COLMAP only uses x, y here.
    for table in ["keypoints", "descriptors", "matches", "two_view_geometries"]:
        cursor.execute(f"DELETE FROM {table};")
    num_hybrid_keypoints = 0
    for image_id in image_ids:
        h, w = image_sizes[image_id]
        kpts = unrotate_kps_W(keypoints_rot[image_id], rotations_of[image_id], h, w)
        n = kpts.shape[0]
        kpts_mod = np.hstack([kpts.astype(np.float32), np.ones((n, 1), np.float32), np.zeros((n, 1), np.float32),
                              np.ones((n, 1), np.float32), np.zeros((n, 1), np.float32)])
        insert_keypoints(cursor, image_id, np.vstack([sift_keypoints[image_id], kpts_mod]))
        num_hybrid_keypoints += n
    conn.commit()
    stats["sift_keypoints"] = int(sum(len(k) for k in sift_keypoints.values()))
    stats["hybrid_keypoints"] = num_hybrid_keypoints

    # 4 + 5b. Match the selected pairs; combine with raw SIFT matches
    if args.matching == "lightglue":
        matcher = LightGlue(features=args.features, depth_confidence=-1, width_confidence=-1, flash=True)
        matcher_settings = {k: v for k, v in vars(matcher.conf).items() if isinstance(v, (int, float, str, bool))}
    else:  # SIFT with Lowe's ratio test (COLMAP's 0.8); learned descriptors with plain mutual nearest neighbours
        matcher = BruteForce(ratio=0.8 if args.features == "sift" else None)
        matcher_settings = {"mutual_nearest_neighbours": True, "ratio_test": matcher.ratio}
    matcher = matcher.eval().to(device)
    stats["features"] = {"extractor": args.features, "max_num_keypoints": MAX_NUM_KEYPOINTS,
                         "rotations_deg": [90 * k for k in rotations]}
    stats["matcher"] = {"type": args.matching, **matcher_settings}
    match_calls = pairs_kept = pairs_written = total_sift = total_hybrid = 0
    t_match = 0.0
    t0 = time.perf_counter()
    for i in tqdm(range(num_images), desc="Feature Matching"):
        id1 = image_ids[i]
        feats1 = feats[id1] if keep_on_gpu else None
        for j in range(i + 1, num_images):
            id2 = image_ids[j]
            matches_sift = sift_matches.get(pair_id_of(id1, id2), np.zeros((0, 2), dtype=np.uint32))

            matches_hybrid = None
            if selected[i, j]:
                if feats1 is None:
                    feats1 = feats_to(feats[id1], device)
                feats2 = feats[id2] if keep_on_gpu else feats_to(feats[id2], device)
                start = time.perf_counter()
                matches_hybrid = match_rotated(feats1, feats2, matcher)
                t_match += time.perf_counter() - start
                match_calls += 1

            if matches_hybrid is not None and len(matches_hybrid) > args.lg_min_matches:
                matches_hybrid[:, 0] += sift_keypoints[id1].shape[0]
                matches_hybrid[:, 1] += sift_keypoints[id2].shape[0]
                pairs_kept += 1
            else:
                matches_hybrid = np.zeros((0, 2), dtype=np.uint32)

            matches = np.vstack([matches_sift, matches_hybrid])
            if len(matches) < args.min_matches:
                continue
            insert_matches(cursor, id1, id2, matches)
            if args.verification == "none":
                insert_all_inlier_two_view_geometry(cursor, id1, id2, matches)
            pairs_written += 1
            total_sift += len(matches_sift)
            total_hybrid += len(matches_hybrid)
    conn.commit()
    conn.close()

    stats.update({"match_calls": match_calls, "hybrid_pairs_kept": pairs_kept, "pairs_written": pairs_written,
                  "sift_matches_written": total_sift, "hybrid_matches_written": total_hybrid,
                  "time_matching_s": t_match, "time_matching_loop_s": time.perf_counter() - t0,
                  "time_total_s": time.perf_counter() - t_start,
                  "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0,
                  "versions": {"torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__,
                               "lightglue_commit": git_commit(Path(sys.modules["lightglue"].__file__).resolve().parents[1])}})
    with open(args.database.parent / "hybrid_matcher_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"{SCRIPT_LABEL}{args.matching} calls: {match_calls} ({pairs_kept} kept), pairs written: {pairs_written}, "
          f"matching time: {t_match:.1f} s, total: {stats['time_total_s']:.1f} s")


def git_commit(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


if __name__ == "__main__":
    main()
