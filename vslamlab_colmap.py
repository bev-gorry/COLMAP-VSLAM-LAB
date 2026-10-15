"""
Module: VSLAM-LAB - Baselines - colmap - vslamlab_colmap.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5.1)
- Version: 2.0
- Created: 2024-07-12
- Updated: 2026-10-03
- License: GPLv3 License

VSLAM-LAB entry point for the colmap baseline (Python port of colmap_reconstruction.sh), run by
the pixi task execute-mono with the arguments BaselineVSLAMLAB.build_execute_command passes for
command_style 'python' (--key value). Stages, each in its own module:
  colmap_matcher.run_matcher      database, feature extraction (masks optional), exhaustive / sequential matching
  colmap_mapper.run_mapper        COLMAP incremental or GLOMAP global mapping, best sub-model, TXT export
  colmap_to_vslamlab              <exp_folder>/<exp_id>_KeyFrameTrajectory.csv from images.txt
  colmap_dense.run_dense          (dense=1) undistort, patch match, fusion -> <exp_id>_dense.ply, optional mesher -> <exp_id>_mesh.ply
With verbose=1 the best sub-model (or the fused cloud with dense=1) is opened in the colmap gui afterwards.
Every COLMAP command is timed into <exp_folder>/<exp_id>_profiling.csv by stage (extraction, matching,
reconstruction, trajectory, dense), with its peak RAM / GPU memory, see colmap_utilities.Profiler.
<exp_folder>/<exp_id>_run_report.yaml summarises the run's configuration, pair counts, registration coverage,
runtimes, peak memory and hardware (run_report.py), also when the run fails.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from colmap_dense import MESHERS, run_dense  # noqa: E402
from colmap_matcher import MATCHING_TYPES, run_matcher  # noqa: E402
from colmap_mapper import run_mapper  # noqa: E402
from colmap_to_vslamlab import colmap_to_vslamlab  # noqa: E402
from colmap_utilities import PROFILER, load_settings, rgb_path_from_csv, run  # noqa: E402
from run_report import write_run_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VSLAM-LAB colmap baseline")
    # Fixed per-run arguments (BaselineVSLAMLAB.build_execute_command)
    parser.add_argument("--sequence_path", type=Path, required=True)
    parser.add_argument("--calibration_yaml", type=Path, required=True)
    parser.add_argument("--rgb_csv", type=Path, required=True)
    parser.add_argument("--exp_folder", type=Path, required=True)
    parser.add_argument("--exp_it", type=int, required=True)
    parser.add_argument("--settings_yaml", type=Path, default=None)
    # Baseline parameters (baseline_colmap.py default_parameters, overridable per experiment)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--mode", type=str, default="mono")
    parser.add_argument("--matcher_type", type=str, default="exhaustive", choices=["exhaustive", "sequential", "hybrid"])
    parser.add_argument("--matching_type", type=str, default="sift_bruteforce", choices=list(MATCHING_TYPES))
    parser.add_argument("--mapper_type", type=str, default="colmap", choices=["colmap", "glomap"])
    parser.add_argument("--rgb_max", type=int, default=None, help="consumed by the run pipeline; accepted here because it is forwarded")
    parser.add_argument("--use_mask", type=int, default=0)
    parser.add_argument("--optimize_intrinsics", type=int, default=1)
    parser.add_argument("--camera_model", type=str, default="", help="COLMAP camera model for an 'unknown' calibration")
    parser.add_argument("--dense", type=int, default=0)
    parser.add_argument("--dense_max_image_size", type=int, default=1600)
    parser.add_argument("--mesher", type=str, default="none", choices=list(MESHERS))
    # Camera grouping, repeatability, and matcher_type 'hybrid' (colmap_matcher.run_hybrid)
    parser.add_argument("--single_camera_per_folder", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=-1)
    parser.add_argument("--hybrid_features", type=str, default="superpoint", choices=["superpoint", "aliked", "sift"])
    parser.add_argument("--hybrid_matching", type=str, default="lightglue", choices=["lightglue", "bruteforce"])
    parser.add_argument("--pair_selection", type=str, default="vpr", choices=["vpr", "cross_session", "all"])
    parser.add_argument("--vpr_matrix", type=str, default="")
    parser.add_argument("--vpr_rgb_csv", type=str, default="")
    parser.add_argument("--vpr_threshold", type=float, default=0.5)
    parser.add_argument("--lg_min_matches", type=int, default=40)
    parser.add_argument("--min_matches", type=int, default=10)
    parser.add_argument("--verification", type=str, default="colmap", choices=["colmap", "glomap", "none"])
    # Not exposed as baseline parameters
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--camera_name", type=str, default="rgb_0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_id = f"{args.exp_it:05d}"

    print("\n================= Experiment Configuration =================")
    print(f"  Sequence Path     : {args.sequence_path}")
    print(f"  Experiment Folder : {args.exp_folder}")
    print(f"  Experiment ID     : {exp_id}")
    print(f"  Verbose           : {args.verbose}")
    print(f"  Matcher Type      : {args.matcher_type}")
    print(f"  Matching Type     : {args.matching_type}")
    print(f"  Mapper Type       : {args.mapper_type}")
    print(f"  Use GPU           : {args.use_gpu}")
    print(f"  Use Mask          : {args.use_mask}")
    print(f"  Optimize Intrins. : {args.optimize_intrinsics}")
    print(f"  Camera model      : {args.camera_model or 'from calibration (OPENCV if unknown)'}")
    print(f"  Dense             : {args.dense}")
    print(f"  Dense Max Img Size: {args.dense_max_image_size}")
    print(f"  Mesher            : {args.mesher}")
    print(f"  Camera per folder : {args.single_camera_per_folder}")
    print(f"  Random seed       : {args.random_seed}")
    if args.matcher_type == "hybrid":
        print(f"  Hybrid features   : {args.hybrid_features} + {args.hybrid_matching}")
        print(f"  Pair selection    : {args.pair_selection} (vpr_threshold {args.vpr_threshold})")
        print(f"  VPR matrix        : {args.vpr_matrix or '<sequence>/vpr-lab/D.npy'} (rows: {args.vpr_rgb_csv or '<sequence>/rgb.csv'})")
        print(f"  Hybrid min        : {args.lg_min_matches}, min matches: {args.min_matches}")
        print(f"  Verification      : {args.verification}")
    print(f"  Settings YAML     : {args.settings_yaml}")
    print(f"  Calibration YAML  : {args.calibration_yaml}")
    print(f"  RGB CSV           : {args.rgb_csv}")
    print(f"  Camera Name       : {args.camera_name}")
    print("============================================================")

    # Folder for the colmap files of this run
    exp_folder_colmap = args.exp_folder / f"colmap_{exp_id}"
    shutil.rmtree(exp_folder_colmap, ignore_errors=True)
    exp_folder_colmap.mkdir(parents=True)

    # Per-stage timing -> <exp_folder>/<exp_id>_profiling.csv (summary printed at the end, also on failure)
    PROFILER.start(args.exp_folder / f"{exp_id}_profiling.csv")
    try:
        run_pipeline(args, exp_id, exp_folder_colmap)
    finally:
        PROFILER.summary()
        try:  # the report must never turn a finished run into a failed one
            write_run_report(args, exp_id, exp_folder_colmap, load_settings(args.settings_yaml))
        except Exception as e:  # noqa: BLE001
            print(f"    WARNING: run report not written: {e!r}")


def run_pipeline(args: argparse.Namespace, exp_id: str, exp_folder_colmap: Path) -> None:

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = f"{conda_prefix}/plugins/platforms"

    rgb_path = rgb_path_from_csv(args.sequence_path, args.rgb_csv, args.camera_name)

    # COLMAP options from the settings yaml (forwarded per stage, see colmap_utilities.settings_args)
    settings = load_settings(args.settings_yaml)

    hybrid = {key: getattr(args, key) for key in ("hybrid_features", "hybrid_matching", "pair_selection", "vpr_matrix",
                                                  "vpr_rgb_csv", "vpr_threshold", "lg_min_matches", "min_matches",
                                                  "verification", "random_seed")}
    if args.matcher_type == "hybrid" and args.verification == "none" and args.mapper_type == "glomap":
        print("    WARNING: verification 'none' with mapper_type 'glomap': COLMAP 4.x's global mapper does not re-estimate "
              "relative poses, so unverified matches reach rotation averaging (see colmap_matcher.run_hybrid)")
    database = run_matcher(args.sequence_path, exp_folder_colmap, rgb_path, args.rgb_csv, args.calibration_yaml, args.camera_name,
                           args.matcher_type, args.matching_type, args.use_gpu, args.use_mask, settings,
                           single_camera_per_folder=args.single_camera_per_folder, hybrid=hybrid,
                           camera_model=args.camera_model)
    best_model = run_mapper(exp_folder_colmap, rgb_path, args.calibration_yaml, args.camera_name, args.mapper_type,
                            args.optimize_intrinsics, settings, random_seed=args.random_seed)

    # Convert COLMAP outputs to a format suitable for VSLAM-LAB
    PROFILER.set_stage("trajectory")
    with PROFILER.step("colmap_to_vslamlab"):
        colmap_to_vslamlab(args.exp_folder, exp_id, args.rgb_csv, args.camera_name)

    # Dense reconstruction (optional, after the trajectory so a dense problem never costs it)
    gui_model, gui_images = best_model, rgb_path
    if args.dense == 1:
        dense_gui_model = run_dense(args.exp_folder, exp_id, exp_folder_colmap, best_model, rgb_path, args.use_gpu,
                                    args.dense_max_image_size, args.mesher, gui_model=(args.verbose == 1), settings=settings)
        if dense_gui_model is not None:  # show the fused cloud instead of the sparse points
            gui_model, gui_images = dense_gui_model, exp_folder_colmap / "dense" / "images"

    # Visualization with colmap gui
    if args.verbose == 1:
        mesh_ply = args.exp_folder / f"{exp_id}_mesh.ply"
        if mesh_ply.is_file():
            print(f"\n    colmap gui: to see the mesh, use File > Import model from... (or drag it in): {mesh_ply}")
        PROFILER.set_stage("gui")
        run(["colmap", "gui", "--import_path", str(gui_model), "--database_path", str(database), "--image_path", str(gui_images)])


if __name__ == "__main__":
    main()
