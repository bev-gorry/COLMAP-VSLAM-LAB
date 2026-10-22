"""
Re-run only the mapping stage of finished colmap runs, starting the incremental mapper from the cross-session image
pair with the most verified matches that has enough parallax for COLMAP (init_pair cross_session, see
colmap_mapper.cross_session_init_pair). Feature
extraction, matching and verification are deterministic, so the run's own database is reused as it is.

  check   (default) per run: the sessions of the initial pair COLMAP chose (from the system output), sub-models and
          registration; 'REMAP' marks runs that started inside one session and fragmented (more than one sub-model)
  --remap re-map the given runs: the previous mapping outputs move to colmap_<id>/before_remap/ (sub-models, TXT model,
          best_model, trajectory, run report, profiling), the extraction and matching rows of the profiling csv are
          kept, and the trajectory and run report are rewritten as the pipeline does

The run's original arguments are read from its system output (the vslamlab_colmap.py command line), so the re-run uses
the same settings, cameras and seed.

usage (colmap env, from the VSLAM-LAB root):
  pixi run -e colmap python Baselines/colmap/remap_colmap.py <run folder> ...            # check
  pixi run -e colmap python Baselines/colmap/remap_colmap.py <run folder> ... --remap    # re-map
<run folder> = <evaluation>/<exp>/<DATASET>/<sequence>
"""

import argparse
import csv
import re
import shlex
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vslamlab_colmap  # noqa: E402
from colmap_mapper import run_mapper  # noqa: E402
from colmap_to_vslamlab import colmap_to_vslamlab  # noqa: E402
from colmap_utilities import PROFILER, load_settings, rgb_path_from_csv  # noqa: E402
from run_report import write_run_report  # noqa: E402

# Outputs of the mapping stage and after it (moved aside before re-mapping)
MODEL_FILES = ("best_model", "cameras.txt", "images.txt", "points3D.txt", "frames.txt", "rigs.txt")
KEPT_STAGES = ("setup", "extraction", "matching")


def original_args(run: Path, exp_id: str) -> argparse.Namespace:
    """The run's vslamlab_colmap.py arguments, parsed from the first line of its system output that calls it."""
    log = run / f"system_output_{exp_id}.txt"
    line = next((text for text in open(log, errors="replace") if "vslamlab_colmap.py" in text), None)
    if line is None:
        raise SystemExit(f"{log}: no vslamlab_colmap.py command line")
    tokens = shlex.split(line[line.index("vslamlab_colmap.py"):])[1:]
    # An empty value (e.g. --camera_model '') is lost in the logged line: restore it between two flags
    tokens = [x for i, tok in enumerate(tokens)
              for x in ([tok, ""] if tok.startswith("--") and (i + 1 == len(tokens) or tokens[i + 1].startswith("--")) else [tok])]
    saved, sys.argv = sys.argv, ["vslamlab_colmap.py", *tokens]
    try:
        return vslamlab_colmap.parse_args()
    finally:
        sys.argv = saved


def initial_pair_sessions(run: Path, exp_id: str, colmap_dir: Path, rgb_csv: Path) -> str:
    """Sessions of the first initial pair the mapper logged ('ssk17-ssk17'), or '?'."""
    text = (run / f"system_output_{exp_id}.txt").read_text(errors="replace")
    match = re.search(r"Registering initial image pair #(\d+) and #(\d+)", text)
    if not match:
        return "?"
    df = pd.read_csv(rgb_csv)
    if "sequence_name" not in df.columns:
        return "single session"
    session = dict(zip((Path(str(p)).name for p in df["path_rgb_0"]), df["sequence_name"].astype(str)))
    with sqlite3.connect(f"file:{colmap_dir / 'colmap_database.db'}?mode=ro", uri=True) as conn:
        names = dict(conn.execute("SELECT image_id, name FROM images"))
    return "-".join(session[Path(names[int(i)]).name] for i in match.groups())


def check(run: Path, exp_id: str) -> None:
    if not (run / f"{exp_id}_run_report.yaml").is_file():
        print(f"       {run.parents[1].name:45s} {run.name:12s} no run report (unfinished run), skipped")
        return
    report = yaml.safe_load(open(run / f"{exp_id}_run_report.yaml"))
    registration = report["results"]["registration"]
    colmap_dir = run / f"colmap_{exp_id}"
    remapped = (colmap_dir / "before_remap").exists()
    # After a re-map the system output still logs the original initial pair
    pair = "cross-session" if remapped else initial_pair_sessions(run, exp_id, colmap_dir, run / "rgb_exp.csv")
    sessions = pair.split("-")
    fragmented = registration["sub_models"] > 1
    flag = "REMAP" if len(sessions) == 2 and sessions[0] == sessions[1] and fragmented else ""
    remapped = " (re-mapped)" if remapped else ""
    print(f"{flag:5s}  {run.parents[1].name:45s} {run.name:12s} initial pair {pair:12s} sub-models {registration['sub_models']}  "
          f"registered {registration['coverage_percent']}%  {registration['per_session']}{remapped}")


def load_profiling(path: Path) -> None:
    """Start PROFILER from the run's profiling csv, keeping only the stages before mapping."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["stage"] in KEPT_STAGES and r["step"] != "total":
                rows.append((r["stage"], r["step"], float(r["start (s)"]), float(r["duration (s)"]), r["status"],
                             float(r["peak RAM (GB)"]) if r["peak RAM (GB)"] else None,
                             float(r["peak GPU (GB)"]) if r["peak GPU (GB)"] else None))
    end = max((start + duration for _, _, start, duration, *_ in rows), default=0.0)
    PROFILER.path, PROFILER.rows = path, rows
    PROFILER.t0 = time.perf_counter() - end  # new rows continue after the kept ones


def remap(run: Path, exp_id: str) -> None:
    args = original_args(run, exp_id)
    args.init_pair = "cross_session"
    # The run's own files from the folder given, not the logged paths (the run may have been moved or copied back)
    args.exp_folder, args.rgb_csv, args.calibration_yaml = run, run / args.rgb_csv.name, run / args.calibration_yaml.name
    colmap_dir = args.exp_folder / f"colmap_{exp_id}"
    backup = colmap_dir / "before_remap"
    if backup.exists():
        raise SystemExit(f"{backup} exists: this run was already re-mapped (move it away to re-map again)")
    backup.mkdir()
    for p in colmap_dir.iterdir():
        if (p.is_dir() and p.name.isdigit()) or p.name in MODEL_FILES:
            shutil.move(str(p), backup / p.name)
    for name in (f"{exp_id}_KeyFrameTrajectory.csv", f"{exp_id}_run_report.yaml", f"{exp_id}_profiling.csv"):
        if (args.exp_folder / name).exists():
            shutil.copy2(args.exp_folder / name, backup / name)

    load_profiling(args.exp_folder / f"{exp_id}_profiling.csv")
    settings = load_settings(args.settings_yaml)
    rgb_path = rgb_path_from_csv(args.sequence_path, args.rgb_csv, args.camera_name)
    try:
        run_mapper(colmap_dir, rgb_path, args.calibration_yaml, args.camera_name, args.mapper_type,
                   args.optimize_intrinsics, settings, random_seed=args.random_seed,
                   init_pair=args.init_pair, rgb_csv=args.rgb_csv)
        PROFILER.set_stage("trajectory")
        with PROFILER.step("colmap_to_vslamlab"):
            colmap_to_vslamlab(args.exp_folder, exp_id, args.rgb_csv, args.camera_name)
    finally:
        PROFILER.summary()
        write_run_report(args, exp_id, colmap_dir, settings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="<evaluation>/<exp>/<DATASET>/<sequence> run folders")
    parser.add_argument("--exp_it", type=int, default=0, help="run number (colmap_<id>)")
    parser.add_argument("--remap", action="store_true", help="re-map the runs (default: only check them)")
    args = parser.parse_args()
    exp_id = f"{args.exp_it:05d}"
    for run in args.runs:
        if args.remap:
            print(f"\n=== re-mapping {run} (run {exp_id}) ===")
            remap(run.resolve(), exp_id)
        check(run.resolve(), exp_id)


if __name__ == "__main__":
    main()
