"""
Module: VSLAM-LAB - Baselines - colmap - colmap_utilities.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5.1)
- Version: 1.0
- Created: 2026-10-03
- Updated: 2026-10-03
- License: GPLv3 License

Helpers shared by the colmap pipeline stages (vslamlab_colmap.py, colmap_matcher.py,
colmap_mapper.py): running colmap commands, GPU / thread detection, and mapping a VSLAM-LAB
calibration onto a COLMAP camera model.
"""

import contextlib
import csv
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from get_calibration import get_camera_intrinsics

COLMAP_DIR = Path(__file__).resolve().parent


class Profiler:
    """Per-run timing of the pipeline, written to <exp_folder>/<exp_id>_profiling.csv.

    Every command that goes through run() is one row (its stage, the COLMAP sub-command, start offset,
    duration, status, peak RAM and GPU memory of the command's process tree, see MemorySampler); Python-side
    work is recorded with step() (no memory columns). Stage functions call
    PROFILER.set_stage('extraction') etc. at their boundaries. The csv is rewritten after every step, so a run that dies
    half-way keeps the rows up to the failure; it ends with one 'total' row per stage and an 'all'
    total (the interactive 'gui' stage is listed but excluded from 'all'). summary() prints the stage
    table at the end of the system output."""

    COLUMNS = ("stage", "step", "start (s)", "duration (s)", "status", "peak RAM (GB)", "peak GPU (GB)")

    def __init__(self) -> None:
        self.path: Path | None = None
        self.t0 = time.perf_counter()
        # (stage, step, start, duration, status, peak RAM GB or None, peak GPU GB or None)
        self.rows: list[tuple[str, str, float, float, str, float | None, float | None]] = []
        self.current_stage = "setup"

    def start(self, path: Path) -> None:
        self.path, self.t0, self.rows = path, time.perf_counter(), []
        self.write()

    def set_stage(self, name: str) -> None:
        """Stage the following steps belong to (extraction, matching, reconstruction, trajectory, dense, gui)."""
        self.current_stage = name

    def record(self, step: str, start: float, status: str = "ok",
               peak_ram_gb: float | None = None, peak_gpu_gb: float | None = None, end: float | None = None) -> None:
        end = time.perf_counter() if end is None else end
        self.rows.append((self.current_stage, step, start - self.t0, end - start, status, peak_ram_gb, peak_gpu_gb))
        self.write()

    @contextlib.contextmanager
    def step(self, name: str):
        start, status = time.perf_counter(), "ok"
        try:
            yield
        except BaseException:
            status = "failed"
            raise
        finally:
            self.record(name, start, status)

    def set_last_duration(self, duration: float) -> None:
        """Replace the last row's duration, for a command that reports its own processing time (hybrid matcher)."""
        stage, step, start, _, status, ram, gpu = self.rows[-1]
        self.rows[-1] = (stage, step, start, duration, status, ram, gpu)
        self.write()

    def skipped(self, step: str) -> None:
        self.rows.append((self.current_stage, step, time.perf_counter() - self.t0, 0.0, "skipped", None, None))
        self.write()

    def stage_totals(self) -> list[tuple[str, float]]:
        totals: dict[str, float] = {}
        for stage, _, _, duration, *_ in self.rows:
            totals[stage] = totals.get(stage, 0.0) + duration
        return list(totals.items())

    def stage_peaks(self) -> dict[str, tuple[float | None, float | None]]:
        """stage -> (peak RAM GB, peak GPU GB) over the stage's commands (None if nothing was measured)."""
        peaks: dict[str, tuple[float | None, float | None]] = {}
        for stage, _, _, _, _, ram, gpu in self.rows:
            old_ram, old_gpu = peaks.get(stage, (None, None))
            peaks[stage] = (_max_or_none(old_ram, ram), _max_or_none(old_gpu, gpu))
        return peaks

    def write(self) -> None:
        if self.path is None:
            return
        totals = self.stage_totals()
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)
            for stage, step, start, duration, status, ram, gpu in self.rows:
                writer.writerow([stage, step, f"{start:.3f}", f"{duration:.3f}", status, _fmt_gb(ram), _fmt_gb(gpu)])
            for stage, duration in totals:
                writer.writerow([stage, "total", "", f"{duration:.3f}", ""])
            writer.writerow(["all", "total", "", f"{sum(d for s, d in totals if s != 'gui'):.3f}", ""])

    def summary(self) -> None:
        totals = self.stage_totals()
        overall = sum(d for s, d in totals if s != "gui") or 1.0
        print("\n================= Profiling =================")
        for stage, duration in totals:
            share = "" if stage == "gui" else f"{100 * duration / overall:5.1f} %"
            print(f"  {stage:15s} {duration:9.1f} s  {share}")
        print(f"  {'all':15s} {overall:9.1f} s")
        if self.path is not None:
            print(f"  written to {self.path}")
        print("=============================================")


PROFILER = Profiler()


def _max_or_none(a: float | None, b: float | None) -> float | None:
    return b if a is None else a if b is None else max(a, b)


def _fmt_gb(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


class MemorySampler:
    """Peak memory of one command's process tree (the command and every process it starts, e.g. pixi -> python),
    sampled every `interval` s while it runs:
      RAM: sum of VmRSS over the tree (/proc, Linux only);
      GPU: sum of the tree's entries in `nvidia-smi --query-compute-apps` (per process, so other users' jobs on a
           shared HPC node are not counted). Stays None when nvidia-smi lists none of the tree's processes.
    run() also takes the kernel's ru_maxrss of the tree, which catches RAM peaks shorter than the interval."""

    def __init__(self, pid: int, interval: float = 1.0) -> None:
        self.pid, self.interval = pid, interval
        self.peak_ram_gb: float | None = None
        self.peak_gpu_gb: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _tree(self) -> list[int]:
        pids, stack = [], [self.pid]
        while stack:
            pid = stack.pop()
            pids.append(pid)
            try:
                for task in os.listdir(f"/proc/{pid}/task"):
                    with open(f"/proc/{pid}/task/{task}/children") as f:
                        stack.extend(int(c) for c in f.read().split())
            except OSError:
                pass
        return pids

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            pids = self._tree()
            ram_kb = 0
            for pid in pids:
                try:
                    with open(f"/proc/{pid}/status") as f:
                        ram_kb += next((int(line.split()[1]) for line in f if line.startswith("VmRSS:")), 0)
                except OSError:
                    pass
            if ram_kb:
                self.peak_ram_gb = _max_or_none(self.peak_ram_gb, ram_kb / 1024**2)
            out = shell_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
            gpu_mb = [int(m) for p, m in re.findall(r"^\s*(\d+),\s*(\d+)", out, re.MULTILINE) if int(p) in pids]
            if gpu_mb:
                self.peak_gpu_gb = _max_or_none(self.peak_gpu_gb, sum(gpu_mb) / 1024)


def run(cmd: list[str], capture: bool = False, exit_on_error: bool = True, step: str | None = None,
        profile: bool = True) -> str:
    """Run a command; on failure exit with its return code (default) or raise CalledProcessError
    (exit_on_error=False, for optional stages). Returns the merged output when capture=True.
    Each call is one row of the profile (step = the colmap sub-command, or `step` when given), with the peak RAM /
    GPU memory of the command's process tree (MemorySampler). The duration ends when the command exits, before the
    sampler is stopped. profile=False runs it without a row (setup commands that are not part of the processing)."""
    start = time.perf_counter()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE if capture else None,
                               stderr=subprocess.STDOUT if capture else None, text=True)
    sampler = MemorySampler(process.pid)
    sampler.start()
    output = process.stdout.read() if capture else ""
    if hasattr(os, "wait4"):  # wait4 also returns the tree's resource usage (ru_maxrss: KB on Linux, bytes on macOS)
        _, wait_status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(wait_status)
        sampler.peak_ram_gb = _max_or_none(sampler.peak_ram_gb,
                                           usage.ru_maxrss / (1024**3 if sys.platform == "darwin" else 1024**2))
    else:
        process.wait()
    end = time.perf_counter()
    sampler.stop()
    if capture:
        process.stdout.close()
    if step is None:
        step = cmd[1] if len(cmd) > 1 and cmd[0] == "colmap" else os.path.basename(cmd[0])
    if profile:
        PROFILER.record(step, start, "ok" if process.returncode == 0 else "failed", sampler.peak_ram_gb,
                        sampler.peak_gpu_gb, end=end)
    if process.returncode != 0:
        print(f"    ERROR: '{' '.join(cmd[:2])}' failed with return code {process.returncode}")
        if capture:
            print(output)
        if not exit_on_error:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        sys.exit(process.returncode)
    return output


def colmap_has_cuda() -> bool:
    """True if the colmap binary was built with CUDA (its banner reads 'COLMAP x.y.z (... with CUDA ...)')."""
    return "with CUDA" in shell_output(["colmap", "help"]).splitlines()[0]


# ---- Settings yaml (vslamlab_colmap_settings.yaml) -> COLMAP command-line options ----
# The yaml has one section per pipeline stage (feature_extractor, matcher, mapper, dense) and keys of
# the form <Section>_<param> (e.g. SiftExtraction_peak_threshold), the shape Run/ablations.py edits
# per run for an experiment's Ablation csv. Each key is forwarded as --<Section>.<param> <value> to the
# COLMAP commands of its stage, filtered against that command's own option list (`colmap <cmd> -h`),
# so Mapper_* keys reach the incremental mapper, GlobalMapper_* keys reach glomap, and a key a command
# does not know (typo, option removed in a newer COLMAP) is a warning rather than a failed command.
# Options the pipeline sets itself (ba_refine_*, GPU / thread options, ...) keep precedence: a yaml key
# naming one of them is dropped with a note.
Settings = dict[str, dict[str, str]]  # section -> {"Section.param": "value"}

_command_options_cache: dict[str, set[str]] = {}


def load_settings(settings_yaml: Path | None) -> Settings:
    if settings_yaml is None or str(settings_yaml) in ("", "None"):
        return {}
    if not Path(settings_yaml).is_file():
        print(f"    WARNING: settings yaml {settings_yaml} not found; running COLMAP on its defaults")
        return {}
    import yaml  # local import: only this helper needs it
    with open(settings_yaml) as f:
        data = yaml.safe_load(f) or {}
    settings: Settings = {}
    for section, entries in data.items():
        if not isinstance(entries, dict):
            continue
        settings[section] = {}
        for key, value in entries.items():
            if "_" not in key:
                print(f"    WARNING: settings key '{key}' in [{section}] is not <Section>_<param>; ignored")
                continue
            group, param = key.split("_", 1)
            if isinstance(value, bool):
                value = int(value)
            settings[section][f"{group}.{param}"] = str(value)
    return settings


def command_options(command: str) -> set[str]:
    """Option names (without --) that `colmap <command>` accepts, from its -h output; cached."""
    if command not in _command_options_cache:
        text = shell_output(["colmap", command, "-h"])
        _command_options_cache[command] = set(re.findall(r"--([A-Za-z0-9_.]+) arg", text))
    return _command_options_cache[command]


# Commands that may run for each settings section; a key that another command of the same stage
# accepts (SequentialMatching_* during an exhaustive run, GlobalMapper_* during an incremental run) is
# simply not applicable, only a key none of them knows is warned about.
STAGE_COMMANDS: dict[str, tuple[str, ...]] = {
    "feature_extractor": ("feature_extractor",),
    "matcher": ("exhaustive_matcher", "sequential_matcher"),
    "mapper": ("mapper", "global_mapper"),
    "dense": ("image_undistorter", "patch_match_stereo", "stereo_fusion", "poisson_mesher", "delaunay_mesher", "advancing_front_mesher"),
}


def settings_args(settings: Settings, section: str, command: str, explicit: Iterable[str] = ()) -> list[str]:
    """Command-line tokens for the settings of `section` that `colmap <command>` accepts and the pipeline
    does not set explicitly; prints what was forwarded or dropped, warns about keys unknown to the stage."""
    entries = settings.get(section, {})
    if not entries:
        return []
    known, explicit = command_options(command), set(explicit)
    known_in_stage = set().union(*(command_options(c) for c in STAGE_COMMANDS.get(section, (command,))))
    args: list[str] = []
    forwarded, dropped, unknown = [], [], []
    for option, value in entries.items():
        if option in explicit:
            dropped.append(option)
        elif option in known:
            args += [f"--{option}", value]
            forwarded.append(option)
        elif option not in known_in_stage:
            unknown.append(option)
    print(f"        settings [{section}] -> {command}: {len(forwarded)} option(s) forwarded")
    if dropped:
        print(f"        settings [{section}] -> {command}: set by the pipeline, yaml value ignored: {dropped}")
    if unknown:
        print(f"        WARNING: settings [{section}]: unknown to every {section} command ({', '.join(STAGE_COMMANDS.get(section, (command,)))}), ignored: {unknown}")
    return args


def shell_output(cmd: list[str]) -> str:
    """stdout+stderr of a command, '' if it cannot run (e.g. no nvidia-smi)."""
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    except FileNotFoundError as e:
        return str(e)


def indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.rstrip("\n").split("\n"))


def detect_gpus(use_gpu: int) -> tuple[str, int]:
    """GPU index list for COLMAP's gpu_index options and the CPU thread count.

    Extraction / matching run as a single job across all usable GPUs, since COLMAP spawns one
    worker per listed GPU index and splits the workload internally. When CUDA_VISIBLE_DEVICES is
    set (a scheduler restricted this job to a device set, often MIG slice UUIDs on HPC that
    nvidia-smi does not enumerate per job), CUDA remaps the listed devices to ordinals 0..N-1 inside
    the process and COLMAP's gpu_index is just cudaSetDevice(ordinal), so the ordinals are used
    rather than the real indices. The thread count comes from the scheduling affinity (what `nproc`
    reports) rather than COLMAP's num_threads=-1 auto-detection, which on cgroup-limited nodes often
    sees the whole node's cores.
    """
    print("    detecting GPUs ...")
    print("        nvidia-smi -L:")
    print(indent(shell_output(["nvidia-smi", "-L"]), " " * 12))
    print("        nvidia-smi --query-gpu=index,name,uuid --format=csv:")
    print(indent(shell_output(["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv"]), " " * 12))
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"        CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")

    if cuda_visible_devices:
        num_gpus = len([d for d in cuda_visible_devices.split(",") if d.strip()])
        gpu_ids = list(range(num_gpus))
    else:
        out = shell_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
        gpu_ids = [int(tok) for tok in out.split() if tok.isdigit()]
    if len(gpu_ids) < 1 or use_gpu == 0:
        gpu_ids = [0]
    gpu_index_list = ",".join(str(i) for i in gpu_ids)
    print(f"        use_gpu: {use_gpu}")
    print(f"        detected gpu_ids: {' '.join(str(i) for i in gpu_ids)}")
    print(f"        gpu_index_list passed to colmap: {gpu_index_list}")

    num_threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    print(f"        num_threads: {num_threads}")
    return gpu_index_list, num_threads


def rgb_path_from_csv(sequence_path: Path, rgb_csv: Path, camera_name: str) -> Path:
    """Frame folder of camera_name as the csv names it: the run pipeline may point path_rgb_0 at a
    generated folder (refrax_0 for 'refraction: refrax') rather than rgb_0."""
    df = pd.read_csv(rgb_csv, nrows=1)
    rgb_dir = str(df[f"path_{camera_name}"].iloc[0]).split("/")[0]
    return sequence_path / rgb_dir


# VSLAM-LAB calibration model -> (COLMAP camera model, number of distortion coefficients expected
# after fx fy cx cy, zeros appended to reach COLMAP's parameter count). radtan* are OpenCV's
# k1 k2 p1 p2 [k3 [k4 k5 k6]] in COLMAP's OPENCV / FULL_OPENCV order; equid4 is k1 k2 k3 k4.
_COLMAP_MODELS: dict[str, tuple[str, int, int]] = {
    "pinhole": ("PINHOLE", 0, 0),
    "radtan4": ("OPENCV", 4, 0),
    "radtan5": ("FULL_OPENCV", 5, 3),
    "radtan8": ("FULL_OPENCV", 8, 0),
    "equid4": ("OPENCV_FISHEYE", 4, 0),
}


def colmap_camera(calibration_yaml: Path, camera_name: str) -> tuple[str, str, str]:
    """(vslamlab calibration model, COLMAP camera model, comma-separated COLMAP params or '')."""
    calibration_model, params = get_camera_intrinsics(calibration_yaml, camera_name)
    if calibration_model == "unknown":  # no intrinsics to pass: COLMAP starts from its own guess
        return calibration_model, "OPENCV", ""
    if calibration_model not in _COLMAP_MODELS:
        print(f"Unknown calibration_model: {calibration_model} (expected one of {['unknown', *_COLMAP_MODELS]})")
        sys.exit(1)
    colmap_model, num_dist, num_zeros = _COLMAP_MODELS[calibration_model]
    if len(params) != 4 + num_dist:
        print(f"calibration_model {calibration_model} expects {num_dist} distortion coefficients, "
              f"{camera_name} in {calibration_yaml} has {len(params) - 4}")
        sys.exit(1)
    params_csv = ",".join(str(p) for p in params + [0.0] * num_zeros)
    return calibration_model, colmap_model, params_csv
