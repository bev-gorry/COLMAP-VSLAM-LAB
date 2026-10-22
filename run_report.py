"""
Module: VSLAM-LAB - Baselines - colmap - run_report.py
- Author: Beverley Gorry
- Assisted by: Claude (Opus 5.5)
- Version: 1.0
- Created: 2026-10-14
- Updated: 2026-10-14
- License: GPLv3 License

Run report of one colmap run: <exp_folder>/<exp_id>_run_report.yaml, written by vslamlab_colmap.py at the end of
every run (also a failed one, with what exists by then). One file holds what a paper table needs:
  configuration   VPR method + threshold, COLMAP features + matching thresholds, hybrid (SuperPoint / ALIKED / SIFT + LightGlue / brute force)
                  features + thresholds, geometric verification, mapper
  input           images, sessions, image resolution
  results         candidate pairs (VPR), matched and verified pairs (all / cross-session), registration coverage
  runtime         total and per stage / step (VPR is computed before the run; its time comes from vpr_info.yaml)
  memory          peak RAM and GPU memory per stage (colmap_utilities.MemorySampler)
  hardware        CPU, threads used, RAM, GPUs, COLMAP / torch versions
Sources: the experiment arguments, the settings yaml, the COLMAP database and exported TXT model, the profiler,
hybrid_matcher_stats.json and the VPR matrix's vpr_info.yaml.
"""

import argparse
import json
import os
import platform
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from colmap_matcher import MATCHING_TYPES, VERIFICATION_PROFILES, resolve_sequence_file
from colmap_utilities import PROFILER, Settings, shell_output

# COLMAP's kMaxNumImages (util/types.h): pair_id = min_id * MAX_NUM_IMAGES + max_id
MAX_NUM_IMAGES = 2147483647
# COLMAP TwoViewGeometry configs >= 2 (CALIBRATED, UNCALIBRATED, PLANAR, ...) passed verification; 0/1 did not
FIRST_VERIFIED_CONFIG = 2
# COLMAP 4.2.1 defaults of the verification options, used when the settings yaml does not set one
VERIFICATION_DEFAULTS = {"max_error": "4", "min_num_inliers": "15", "min_inlier_ratio": "0",
                         "confidence": "0.999", "max_num_trials": "10000"}


def write_run_report(args: argparse.Namespace, exp_id: str, exp_folder_colmap: Path, settings: Settings) -> Path:
    report = {
        "experiment": experiment_section(args, exp_id),
        "configuration": configuration_section(args, exp_folder_colmap, settings),
        "input": input_section(args, exp_folder_colmap),
        "results": results_section(args, exp_folder_colmap),
        "runtime_s": runtime_section(args, exp_folder_colmap),
        "peak_memory_gb": memory_section(exp_folder_colmap),
        "hardware": hardware_section(exp_folder_colmap),
    }
    path = args.exp_folder / f"{exp_id}_run_report.yaml"
    with open(path, "w") as f:
        f.write(f"# Run report of {args.exp_folder.parent.parent.name} / {args.sequence_path.name} (run {exp_id}), "
                f"see Baselines/colmap/run_report.py\n")
        yaml.safe_dump(report, f, sort_keys=False, width=120, allow_unicode=True)
    print(f"\n    run report: {path}")
    return path


# ---- Sections --------------------------------------------------------------------------------------------------------

def experiment_section(args, exp_id: str) -> dict:
    return {"name": args.exp_folder.parent.parent.name, "sequence": args.sequence_path.name, "run": exp_id,
            "date": datetime.now().isoformat(timespec="seconds"), "exp_folder": str(args.exp_folder)}


def configuration_section(args, exp_folder_colmap: Path, settings: Settings) -> dict:
    extraction_type, matching_type = MATCHING_TYPES[args.matching_type]
    hybrid = args.matcher_type == "hybrid"
    config: dict = {}

    # VPR pair selection (hybrid only)
    if hybrid and args.pair_selection == "vpr":
        matrix = resolve_sequence_file(args.sequence_path, args.vpr_matrix, "vpr-lab/D.npy")
        info = read_yaml(matrix.parent / "vpr_info.yaml")
        config["vpr"] = {
            "method": info.get("method", "unknown (no vpr_info.yaml next to the matrix)"),
            "rotations_deg": info.get("rotations_deg", "unknown"),
            "threshold": args.vpr_threshold,
            "threshold_rule": "distances min-max normalised over the whole matrix; self and same-session pairs "
                              "excluded; pair is a candidate if normalised distance <= threshold",
            "matrix": str(matrix),
        }
    elif hybrid:
        config["pair_selection"] = {"cross_session": "every pair of images from different sessions",
                                    "all": "every image pair"}[args.pair_selection]

    # COLMAP features: in a hybrid run these are matched exhaustively over ALL pairs, before the hybrid matcher
    config["colmap_features"] = {
        "extractor": extraction_type,
        "matcher": f"{matching_type} ({args.matcher_type if not hybrid else 'exhaustive'}"
                   f"{', all image pairs incl. cross-session' if args.matcher_type != 'sequential' else ''})",
        **selected(settings.get("feature_extractor", {}), EXTRACTOR_KEYS.get(extraction_type.split("_")[0], ())),
        **selected(settings.get("matcher", {}), MATCHER_KEYS.get(matching_type, ()) + ("FeatureMatching.max_num_matches",)),
    }

    # Hybrid (cross-session) features
    if hybrid:
        stats = read_json(exp_folder_colmap / "hybrid_matcher_stats.json")
        features, matcher = stats.get("features", {}), stats.get("matcher", {})
        matcher_keys = {"lightglue": ("filter_threshold", "depth_confidence", "width_confidence"),
                        "bruteforce": ("mutual_nearest_neighbours", "ratio_test")}.get(matcher.get("type"), ())
        config["hybrid_features"] = {
            "extractor": features.get("extractor", args.hybrid_features),
            "max_keypoints_per_rotation": features.get("max_num_keypoints", "unknown"),
            "rotations_deg": features.get("rotations_deg", "unknown"),
            "matcher": matcher.get("type", args.hybrid_matching),
            **{k: matcher[k] for k in matcher_keys if k in matcher},
            "keep_pair_if_hybrid_matches_more_than": args.lg_min_matches,
            "drop_pair_if_total_matches_below": args.min_matches,
        }

    # Geometric verification
    matcher_settings = settings.get("matcher", {})
    verification_options = {k: matcher_settings.get(f"TwoViewGeometry.{k}", v) for k, v in VERIFICATION_DEFAULTS.items()}
    if hybrid and args.verification == "none":
        config["geometric_verification"] = {"method": "none (all matches stored as inliers)"}
    else:
        if hybrid:
            profile = VERIFICATION_PROFILES[args.verification]
            verification_options.update({k.split(".", 1)[1]: v for k, v in profile.items()})
            method = f"colmap geometric_verifier on SIFT + hybrid matches (profile: {args.verification})"
        else:
            method = f"colmap {args.matcher_type}_matcher built-in two-view verification"
        config["geometric_verification"] = {"method": method,
                                            **{k: number(v) for k, v in verification_options.items()}}

    # Mapper
    config["mapper"] = {
        "type": "COLMAP incremental mapper" if args.mapper_type == "colmap" else "COLMAP global mapper (GLOMAP)",
        "optimize_intrinsics": bool(args.optimize_intrinsics),
        # COLMAP groups cameras by the folder in the image name: a combined sequence's single rgb folder means one
        # shared camera unless per_session_calibration extracts each session with its own camera
        "cameras": ("one camera per session, intrinsics from each session's calibration.yaml" if args.per_session_calibration
                    else "one shared camera" if args.single_camera_per_folder == 0 else "one camera per image folder"),
        "random_seed": args.random_seed,
        "initial_pair": ("cross-session pair with the most verified matches and >= 1.25 x init_min_tri_angle parallax" if getattr(args, "init_pair", "colmap") == "cross_session"
                         else "chosen by COLMAP"),
    }
    return config


def input_section(args, exp_folder_colmap: Path) -> dict:
    sessions = session_of_image(args)
    section: dict = {"images": len(sessions), "images_per_session": count_by_session(sessions.values())}
    database = exp_folder_colmap / "colmap_database.db"
    if database.is_file():
        with sqlite3.connect(database) as conn:
            sizes = conn.execute("SELECT DISTINCT width, height FROM cameras").fetchall()
        section["image_resolution_px"] = [f"{w} x {h}" for w, h in sizes]
    section["calibration_yaml"] = str(args.calibration_yaml)
    return section


def results_section(args, exp_folder_colmap: Path) -> dict:
    results: dict = {}
    sessions = session_of_image(args)
    num_images = len(sessions)

    pairs: dict = {"possible_pairs": num_images * (num_images - 1) // 2}
    stats = read_json(exp_folder_colmap / "hybrid_matcher_stats.json")
    if args.matcher_type == "hybrid" and stats:
        pairs["hybrid_candidate_pairs"] = stats.get("selected_pairs")  # VPR candidates when pair_selection: vpr
        pairs["hybrid_pairs_kept"] = stats.get("hybrid_pairs_kept")
    database = exp_folder_colmap / "colmap_database.db"
    if database.is_file():
        with sqlite3.connect(database) as conn:
            names = {i: os.path.basename(n) for i, n in conn.execute("SELECT image_id, name FROM images")}
            matched = conn.execute("SELECT COUNT(*) FROM matches WHERE rows > 0").fetchone()[0]
            verified = [(pid, rows) for pid, rows, config in
                        conn.execute("SELECT pair_id, rows, config FROM two_view_geometries")
                        if rows > 0 and config >= FIRST_VERIFIED_CONFIG]
        cross = [rows for pid, rows in verified
                 if sessions.get(names.get(pid // MAX_NUM_IMAGES)) != sessions.get(names.get(pid % MAX_NUM_IMAGES))]
        pairs.update({"matched_pairs": matched, "verified_pairs": len(verified),
                      "verified_cross_session_pairs": len(cross),
                      "verified_cross_session_inlier_matches": int(sum(cross))})
    results["pairs"] = pairs

    registered = registered_images(exp_folder_colmap)
    if registered is not None:
        reg_sessions = [sessions.get(name, "unknown") for name in registered]
        total_by_session = count_by_session(sessions.values())
        reg_by_session = count_by_session(reg_sessions)
        results["registration"] = {
            "registered_images": len(registered),
            "input_images": num_images,
            "coverage_percent": round(100 * len(registered) / num_images, 2) if num_images else 0.0,
            "per_session": {s: f"{reg_by_session.get(s, 0)} / {n}" for s, n in total_by_session.items()},
            "sub_models": len([p for p in exp_folder_colmap.iterdir() if p.is_dir() and p.name.isdigit()]),
            "best_sub_model": read_text(exp_folder_colmap / "best_model"),
            "points_3d": count_points(exp_folder_colmap / "points3D.txt"),
            "refined_cameras": read_cameras(exp_folder_colmap / "cameras.txt"),
        }
    else:
        results["registration"] = "no model exported (run failed before mapping finished)"
    return results


def runtime_section(args, exp_folder_colmap: Path) -> dict:
    """Processing time only: total_colmap_pipeline is the sum of the profiled steps (see colmap_utilities.Profiler)."""
    stage_totals = dict(PROFILER.stage_totals())
    total = sum(d for s, d in stage_totals.items() if s != "gui")
    steps: dict = {}
    for stage, step, _, duration, *_ in PROFILER.rows:
        steps[step] = steps.get(step, 0.0) + duration
    runtime: dict = {"total_colmap_pipeline": round(total, 1),
                     "per_stage": {s: round(d, 1) for s, d in stage_totals.items()},
                     "key_steps": {}}
    key = runtime["key_steps"]
    stats = read_json(exp_folder_colmap / "hybrid_matcher_stats.json")
    for step, label in (("feature_extractor", "colmap_feature_extraction"), ("exhaustive_matcher", "colmap_exhaustive_matching"),
                        ("sequential_matcher", "colmap_sequential_matching"), ("hybrid_matcher", "hybrid_matcher"),
                        ("geometric_verifier", "geometric_verification"),
                        ("mapper", "incremental_mapping"), ("global_mapper", "global_mapping")):
        if step not in steps:
            continue
        key[label] = round(steps[step], 1)
        if step == "hybrid_matcher" and stats:  # its parts, from hybrid_matcher_stats.json
            extraction, matching = stats.get("time_extraction_s", 0.0), stats.get("time_matching_s", 0.0)
            key[label] = {"total": round(steps[step], 1), "feature_extraction": round(extraction, 1),
                          "matching": round(matching, 1),
                          "database_and_pair_selection": round(steps[step] - extraction - matching, 1)}
    if args.matcher_type == "hybrid" and args.pair_selection == "vpr":
        matrix = resolve_sequence_file(args.sequence_path, args.vpr_matrix, "vpr-lab/D.npy")
        vpr = read_yaml(matrix.parent / "vpr_info.yaml").get("runtime_s")
        runtime["vpr_matrix_computed_before_run"] = vpr if vpr is not None else "unknown (no vpr_info.yaml)"
        if vpr is not None:
            runtime["total_including_vpr"] = round(total + vpr, 1)
    return runtime


def memory_section(exp_folder_colmap: Path) -> dict:
    """Peak RAM / GPU memory of the commands of each stage (process-tree measurements, see MemorySampler)."""
    memory: dict = {"per_stage": {}}
    for stage, (ram, gpu) in PROFILER.stage_peaks().items():
        if ram is None and gpu is None:
            continue
        memory["per_stage"][stage] = {"ram": round(ram, 2) if ram is not None else "n/a",
                                      "gpu": round(gpu, 2) if gpu is not None else "n/a"}
    rams = [r for r, _ in PROFILER.stage_peaks().values() if r is not None]
    gpus = [g for _, g in PROFILER.stage_peaks().values() if g is not None]
    memory["overall"] = {"ram": round(max(rams), 2) if rams else "n/a", "gpu": round(max(gpus), 2) if gpus else "n/a"}
    stats = read_json(exp_folder_colmap / "hybrid_matcher_stats.json")
    if stats:
        memory["hybrid_matcher_torch_allocated_gpu"] = round(stats.get("peak_gpu_memory_gb", 0.0), 2)
    return memory


def hardware_section(exp_folder_colmap: Path) -> dict:
    cpu = next((line.split(":", 1)[1].strip() for line in read_text("/proc/cpuinfo", "").splitlines()
                if line.startswith("model name")), platform.processor() or "unknown")
    mem_kb = next((int(line.split()[1]) for line in read_text("/proc/meminfo", "").splitlines()
                   if line.startswith("MemTotal:")), 0)
    gpus = [line.strip() for line in shell_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                                   "--format=csv,noheader"]).splitlines() if "," in line]
    stats = read_json(exp_folder_colmap / "hybrid_matcher_stats.json")
    colmap_banner = [line for line in shell_output(["colmap", "help"]).splitlines() if line.startswith("COLMAP")]
    return {
        "cpu": cpu,
        "cpu_threads_used": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "ram_total_gb": round(mem_kb / 1024**2, 1) if mem_kb else "unknown",
        "gpus": gpus or ["none detected"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
        "host": platform.node(),
        "os": platform.platform(),
        "colmap": colmap_banner[0] if colmap_banner else "unknown",
        "torch_hybrid_matcher": stats.get("versions", {}).get("torch", "n/a"),
    }


# ---- Helpers ---------------------------------------------------------------------------------------------------------

def session_of_image(args) -> dict[str, str]:
    """image file name -> session (rgb_exp.csv's sequence_name column; the sequence name if there is none)."""
    df = pd.read_csv(args.rgb_csv)
    names = [os.path.basename(str(p)) for p in df[f"path_{args.camera_name}"]]
    sessions = df["sequence_name"].astype(str) if "sequence_name" in df.columns else [args.sequence_path.name] * len(names)
    return dict(zip(names, sessions))


def count_by_session(sessions) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in sessions:
        counts[s] = counts.get(s, 0) + 1
    return counts


def registered_images(exp_folder_colmap: Path) -> list[str] | None:
    """Image names of the exported best sub-model (images.txt: two lines per image, the first ends with the name)."""
    path = exp_folder_colmap / "images.txt"
    if not path.is_file():
        return None
    lines = [line for line in open(path) if not line.startswith("#")]
    return [os.path.basename(lines[i].split()[-1]) for i in range(0, len(lines), 2) if lines[i].strip()]


def count_points(path: Path) -> int | None:
    return sum(1 for line in open(path) if line.strip() and not line.startswith("#")) if path.is_file() else None


def read_cameras(path: Path) -> list[str]:
    return [line.strip() for line in open(path) if line.strip() and not line.startswith("#")] if path.is_file() else []


# The settings that set how many features are found and which matches are accepted (values from the settings yaml)
EXTRACTOR_KEYS = {"SIFT": ("SiftExtraction.max_num_features", "SiftExtraction.peak_threshold", "SiftExtraction.edge_threshold"),
                  "ALIKED": ("AlikedExtraction.max_num_features", "AlikedExtraction.min_score"),
                  "LOMA": ("LomaExtraction.max_num_features", "LomaExtraction.min_score")}
MATCHER_KEYS = {"SIFT_BRUTEFORCE": ("SiftMatching.max_ratio", "SiftMatching.max_distance", "SiftMatching.cross_check"),
                "SIFT_LIGHTGLUE": ("SiftMatching.lightglue_min_score",),
                "ALIKED_BRUTEFORCE": (), "ALIKED_LIGHTGLUE": ("AlikedMatching.lightglue_min_score",),
                **{t: ("LomaMatching.min_score",) for t in ("LOMA_BRUTEFORCE", "LOMA_B", "LOMA_R", "LOMA_L", "LOMA_G", "LOMA_B128")}}


def selected(entries: dict[str, str], keys) -> dict:
    return {k.split(".", 1)[1]: number(entries[k]) for k in keys if k in entries}


def number(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except (TypeError, ValueError):
            pass
    return value


def read_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def read_yaml(path: Path) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


def read_text(path, default: str = "") -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default
