#!/usr/bin/env python3
"""Launch multiple `collect_from_mpc.py` workers across one or more GPUs."""

from __future__ import annotations

import argparse
import os
import re
import signal
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from kuafu.data.collection.prepare_bc_prior_sweep import (
    BcPriorPrepConfig,
    prepare_bc_prior_sweep,
)

COLLECT_SCRIPT = Path(__file__).resolve().parent / "collect_from_mpc.py"
COMBINE_HDF5_SCRIPT = Path(__file__).resolve().parents[1] / "datasets" / "combine_hdf5.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch parallel collect_from_mpc workers across GPUs."
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        required=True,
        help="GPU IDs to distribute workers across.",
    )
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=2,
        help="How many worker processes to launch on each GPU.",
    )
    parser.add_argument(
        "--num_demos_per_worker",
        type=int,
        default=100,
        help="How many demos each worker should collect.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=None,
        help=(
            "Optional base seed for collection workers. Worker N receives "
            "--seed BASE_SEED+N so parallel workers do not duplicate RNG streams."
        ),
    )
    parser.add_argument(
        "--cpu_threads_per_worker",
        type=int,
        default=None,
        help=(
            "Optional explicit CPU thread-pool cap for each worker. By default "
            "the cap is derived from CPU count and this launcher's worker count."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional external shared run directory. Defaults below KUAFU_ARTIFACT_ROOT.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="FR3-FinRay-Table-Stack-3",
        help="Task name used for the default output_root and forwarded to workers.",
    )
    parser.add_argument(
        "--robot_name",
        choices=["fr3_finray", "kinova", "panda"],
        default=None,
        help="Robot label for the compacted dataset. Defaults to inferring from task_name.",
    )
    parser.add_argument(
        "--stagger_sec",
        type=float,
        default=5.0,
        help="Seconds to wait between launching consecutive workers.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use for worker processes.",
    )
    parser.add_argument(
        "--no_compact",
        action="store_true",
        help="Do not combine per-worker HDF5 files after collection.",
    )
    parser.add_argument(
        "--compact_output_dir",
        type=str,
        default=None,
        help="External directory for the combined HDF5 file. Defaults below KUAFU_ARTIFACT_ROOT.",
    )
    parser.add_argument(
        "--compact_output",
        type=str,
        default=None,
        help=(
            "Exact combined HDF5 output path. Overrides --compact_output_dir. "
            "Defaults to <compact_output_dir>/<task_name>_<run_name>.hdf5."
        ),
    )
    parser.add_argument(
        "--compact_no_strict_length",
        action="store_true",
        help="Forward --no-strict-length to combine_hdf5.py.",
    )
    parser.add_argument(
        "--compact_skip_bad_files",
        action="store_true",
        help="Forward --skip-bad-files to combine_hdf5.py.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print worker commands without launching them.",
    )
    parser.add_argument(
        "--no_bc_prep",
        action="store_true",
        help="Do not generate BC prior YAML configs and sweep CSV after compaction.",
    )
    parser.add_argument(
        "--bc_chunk_sizes",
        type=int,
        nargs="+",
        default=[10, 15, 30],
        help="Action chunk sizes for generated BC prior configs.",
    )
    parser.add_argument(
        "--bc_chunk_strides",
        type=int,
        nargs="+",
        default=[1, 3],
        help="Action chunk strides for generated BC prior configs.",
    )
    parser.add_argument(
        "--bc_variants",
        nargs="+",
        default=["f", "ff", "fff", "f3", "ff3", "fff3"],
        help="BC model variants to generate. Defaults to: f ff fff f3 ff3 fff3.",
    )
    parser.add_argument(
        "--bc_obs_key",
        type=str,
        default="state",
        help="Observation key for generated BC configs.",
    )
    parser.add_argument(
        "--bc_action_mode",
        type=str,
        default=None,
        choices=["relative_ee", "relative_joint", "absolute_ee", "absolute_joint"],
        help="Action branch for generated BC configs. Defaults to forwarded --control_output_mode.",
    )
    parser.add_argument(
        "--bc_time_shift",
        type=int,
        default=0,
        help="Temporal shift for generated BC configs.",
    )
    parser.add_argument(
        "--bc_num_epochs",
        type=int,
        default=150,
        help="Training epochs for generated BC configs.",
    )
    parser.add_argument(
        "--bc_num_workers",
        type=int,
        default=0,
        help="DataLoader worker processes for generated BC configs.",
    )
    parser.add_argument(
        "--bc_pin_memory",
        action="store_true",
        help="Enable pinned-memory DataLoader batches in generated BC configs.",
    )
    parser.add_argument(
        "--bc_artifact_root",
        type=str,
        default=None,
        help="Override robot artifact root for generated BC configs and sweep CSV.",
    )
    parser.add_argument(
        "--bc_task_slug",
        type=str,
        default=None,
        help="Override inferred task slug for generated BC artifacts.",
    )
    parser.add_argument(
        "--bc_name_prefix",
        type=str,
        default=None,
        help="Override generated BC sweep row/config name prefix.",
    )
    parser.add_argument(
        "--bc_no_sample_weighting",
        action="store_true",
        help="Disable stage-based sample weighting in generated BC configs.",
    )
    parser.add_argument(
        "--bc_sample_weighting_no_validation",
        action="store_true",
        help="Use stage weights for training loss only, not validation/best-checkpoint selection.",
    )
    parser.add_argument(
        "--bc_overwrite",
        action="store_true",
        help="Overwrite generated BC configs and sweep CSV if they already exist.",
    )
    return parser.parse_known_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def external_artifact_root() -> Path:
    task_root = os.environ.get("KUAFU_TASK_ARTIFACT_ROOT", "").strip()
    if task_root:
        return Path(task_root).expanduser().resolve()
    configured = os.environ.get("KUAFU_ARTIFACT_ROOT", "").strip()
    base = (
        Path(configured).expanduser().resolve()
        if configured
        else repo_root().parent / "KuafuArtifacts"
    )
    return base / "standalone"


def require_external_output(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    root = repo_root().resolve()
    if resolved == root or root in resolved.parents:
        raise ValueError(f"{label} must be outside the source project ({root}); got {resolved}")
    return resolved


def resolve_external_output(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = external_artifact_root() / path
    return require_external_output(path, label=label)


def build_output_root(args: argparse.Namespace) -> Path:
    if args.output_root:
        return resolve_external_output(args.output_root, label="--output_root")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return require_external_output(
        external_artifact_root() / "collection" / _safe_filename(args.task_name) / timestamp,
        label="default collection output",
    )


def _safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return value or "dataset"


def infer_robot_name(task_name: str) -> str:
    tokens = [token.lower() for token in re.split(r"[-_]+", task_name) if token]
    if tokens and tokens[0] == "panda":
        return "panda"
    if "kinova" in tokens:
        return "kinova"
    if tokens and tokens[0] == "fr3_finray":
        return "fr3_finray"
    raise ValueError(
        f"Could not infer robot name from task_name={task_name!r}. "
        "Expected a FR3FinRay, Kinova, or Panda task name."
    )


def resolve_robot_name(args: argparse.Namespace) -> str:
    return args.robot_name or infer_robot_name(args.task_name)


def build_compact_output_path(args: argparse.Namespace, output_root: Path) -> Path:
    if args.compact_output:
        output_path = resolve_external_output(args.compact_output, label="--compact_output")
    else:
        output_dir = (
            resolve_external_output(args.compact_output_dir, label="--compact_output_dir")
            if args.compact_output_dir
            else require_external_output(
                external_artifact_root() / "datasets",
                label="default compact output directory",
            )
        )
        robot_name = _safe_filename(resolve_robot_name(args))
        task_name = _safe_filename(args.task_name)
        run_name = _safe_filename(output_root.name)
        output_path = output_dir / f"{robot_name}_{task_name}_{run_name}.hdf5"
    return require_external_output(output_path, label="compact output")


def _worker_dataset_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"mpc_demos_worker(\d+)\.hdf5$", path.name)
    if match is None:
        return (10**9, path.name)
    return (int(match.group(1)), path.name)


def discover_worker_datasets(datasets_dir: Path) -> list[Path]:
    return sorted(
        datasets_dir.glob("mpc_demos_worker*.hdf5"),
        key=_worker_dataset_sort_key,
    )


def compacted_dataset_lengths_are_consistent(hdf5_path: Path) -> bool:
    try:
        import h5py

        from kuafu.data.dataset_utils import fix_demo_length_mismatches

        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                return False
            if not any(key.startswith("demo_") for key in f["data"].keys()):
                return False
        fix_demo_length_mismatches(str(hdf5_path), strict=True)
    except ValueError:
        return False
    except Exception as exc:
        print(f"[compact] Could not verify compacted dataset after nonzero exit: {exc}", file=sys.stderr)
        return False
    return True


def _count_hdf5_demos(path: Path) -> int:
    import h5py

    with h5py.File(path, "r") as f:
        data = f.get("data")
        if data is None:
            return 0
        return sum(1 for key in data.keys() if key.startswith("demo_"))


def validate_worker_dataset_counts(
    args: argparse.Namespace,
    *,
    jobs: list[dict],
    datasets_dir: Path,
) -> list[dict]:
    incomplete: list[dict] = []
    expected = int(args.num_demos_per_worker)
    print()
    print("=== Validating collected worker datasets ===")
    for job in jobs:
        worker_id = int(job["worker_id"])
        path = datasets_dir / f"mpc_demos_worker{worker_id}.hdf5"
        job["dataset_path"] = path
        if not path.exists():
            job["dataset_count"] = 0
            job["dataset_error"] = "missing"
            incomplete.append(job)
            print(f"  [worker {worker_id}] demos: 0/{expected} - MISSING {path}")
            continue
        try:
            count = _count_hdf5_demos(path)
        except Exception as exc:  # noqa: BLE001 - surface corrupt/invalid HDF5 clearly.
            job["dataset_count"] = 0
            job["dataset_error"] = f"{type(exc).__name__}: {exc}"
            incomplete.append(job)
            print(f"  [worker {worker_id}] demos: ?/{expected} - ERROR {job['dataset_error']}")
            continue
        job["dataset_count"] = count
        if count < expected:
            incomplete.append(job)
            print(f"  [worker {worker_id}] demos: {count}/{expected} - INCOMPLETE")
        else:
            print(f"  [worker {worker_id}] demos: {count}/{expected} - OK")
    return incomplete


def preflight_compactor(args: argparse.Namespace, output_root: Path) -> None:
    output_path = build_compact_output_path(args, output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [args.python, str(COMBINE_HDF5_SCRIPT), "--help"]
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = result.stderr.strip()
        if details:
            details = f"\n{details}"
        raise RuntimeError(
            f"Cannot invoke combine_hdf5.py with {args.python}. "
            f"Use a Python environment with h5py installed, or pass --no_compact.{details}"
        )


def compact_worker_datasets(args: argparse.Namespace, output_root: Path, datasets_dir: Path) -> Path | None:
    input_files = discover_worker_datasets(datasets_dir)
    if not input_files:
        print("[compact] No per-worker HDF5 files found; skipping compaction.")
        return None

    output_path = build_compact_output_path(args, output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        str(COMBINE_HDF5_SCRIPT),
        *[str(path) for path in input_files],
        "-o",
        str(output_path),
        "--robot-name",
        resolve_robot_name(args),
    ]
    if args.compact_no_strict_length:
        cmd.append("--no-strict-length")
    if args.compact_skip_bad_files:
        cmd.append("--skip-bad-files")

    print()
    print("=== Compacting collected datasets ===")
    print(f"  Inputs:           {len(input_files)} file(s)")
    print(f"  Output:           {output_path}")
    print(f"  Command:          {shlex.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        if output_path.exists() and compacted_dataset_lengths_are_consistent(output_path):
            print(
                f"[compact] combine_hdf5.py returned {result.returncode} after fixing lengths; "
                "accepting verified compacted dataset."
            )
        else:
            raise RuntimeError(f"Dataset compaction failed with return code {result.returncode}.")

    return output_path


def sanitize_worker_args(worker_args: list[str]) -> list[str]:
    # The launcher owns these arguments and injects worker-specific values itself.
    owned_flags = {
        "--worker_id",
        "--num_demos",
        "--output_dir",
        "--device",
        "--task_name",
    }
    cleaned: list[str] = []
    i = 0
    while i < len(worker_args):
        arg = worker_args[i]
        key = arg.split("=", 1)[0]
        if key in owned_flags:
            if "=" not in arg and i + 1 < len(worker_args) and not worker_args[i + 1].startswith("--"):
                i += 2
            else:
                i += 1
            continue
        cleaned.append(arg)
        i += 1
    return cleaned


def _extract_forwarded_option(worker_args: list[str], flag: str) -> str | None:
    i = 0
    while i < len(worker_args):
        arg = worker_args[i]
        if arg == flag:
            if i + 1 < len(worker_args):
                return worker_args[i + 1]
            return None
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
        i += 1
    return None


def resolve_bc_action_mode(args: argparse.Namespace, worker_args: list[str]) -> str:
    return args.bc_action_mode or _extract_forwarded_option(worker_args, "--control_output_mode") or "absolute_ee"


def prepare_bc_artifacts(
    args: argparse.Namespace,
    *,
    output_root: Path,
    compacted_path: Path,
    worker_args: list[str],
):
    result = prepare_bc_prior_sweep(
        BcPriorPrepConfig(
            dataset_path=compacted_path,
            task_name=args.task_name,
            robot_name=resolve_robot_name(args),
            run_name=output_root.name,
            artifact_root=resolve_external_output(
                args.bc_artifact_root,
                label="--bc_artifact_root",
            )
            if args.bc_artifact_root
            else external_artifact_root() / "prior_artifacts",
            task_slug=args.bc_task_slug,
            name_prefix=args.bc_name_prefix,
            chunk_sizes=tuple(args.bc_chunk_sizes),
            action_chunk_strides=tuple(args.bc_chunk_strides),
            variants=tuple(args.bc_variants),
            obs_key=args.bc_obs_key,
            action_mode=resolve_bc_action_mode(args, worker_args),
            time_shift=args.bc_time_shift,
            num_workers=args.bc_num_workers,
            pin_memory=args.bc_pin_memory,
            num_epochs=args.bc_num_epochs,
            sample_weighting_enabled=not args.bc_no_sample_weighting,
            sample_weighting_validation=not args.bc_sample_weighting_no_validation,
            overwrite=args.bc_overwrite,
        )
    )
    print()
    print("=== BC Prior Prep ===")
    print(f"  Config dir:        {result.config_dir}")
    print(f"  Sweep CSV:         {result.csv_path}")
    print(f"  Generated YAMLs:   {len(result.generated)}")
    return result


def main() -> None:
    args, worker_args = parse_args()
    worker_args = sanitize_worker_args(worker_args)
    if any(arg == "--num_workers" or arg.startswith("--num_workers=") for arg in worker_args):
        raise ValueError("--num_workers was removed; use --workers_per_gpu as the only concurrency knob.")
    if args.base_seed is not None and any(
        arg == "--seed" or arg.startswith("--seed=") for arg in worker_args
    ):
        raise ValueError("Use either --base_seed or a forwarded --seed, not both.")

    if args.workers_per_gpu <= 0:
        raise ValueError("--workers_per_gpu must be positive.")
    if args.cpu_threads_per_worker is not None and args.cpu_threads_per_worker <= 0:
        raise ValueError("--cpu_threads_per_worker must be positive.")
    num_workers = len(args.gpus) * args.workers_per_gpu

    output_root = build_output_root(args)
    logs_dir = output_root / "logs"
    datasets_dir = output_root / "datasets"

    jobs: list[dict] = []
    for worker_id in range(num_workers):
        gpu_id = args.gpus[worker_id % len(args.gpus)]
        # IMPORTANT: do NOT use CUDA_VISIBLE_DEVICES to pin Isaac Sim workers.
        # Omniverse enumerates GPUs via Vulkan independently of CUDA; masking
        # CUDA causes "CUDA in bad state" warnings and silently falls back to
        # CPU PhysX. Instead, leave all GPUs visible and select the GPU per
        # subsystem explicitly via Kit args (passed through --kit_args).
        kit_args = " ".join(
            [
                f"--/physics/cudaDevice={gpu_id}",
                f"--/renderer/activeGpu={gpu_id}",
                "--/renderer/multiGpu/enabled=false",
            ]
        )
        cmd = [
            args.python,
            str(COLLECT_SCRIPT),
            "--task_name",
            args.task_name,
            "--worker_id",
            str(worker_id),
            "--num_demos",
            str(args.num_demos_per_worker),
            "--device",
            f"cuda:{gpu_id}",
            "--output_dir",
            str(output_root),
            "--headless",
            "--kit_args",
            kit_args,
        ]
        if args.base_seed is not None:
            cmd.extend(["--seed", str(args.base_seed + worker_id)])
        cmd.extend(worker_args)
        jobs.append(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "cmd": cmd,
                "log_path": logs_dir / f"worker_{worker_id:02d}.txt",
            }
        )

    print("=== MPC Collection Launch ===")
    print(f"  Script:           {COLLECT_SCRIPT}")
    print(f"  Task:             {args.task_name}")
    print(f"  GPUs:             {args.gpus}")
    print(f"  Workers:          {num_workers}")
    print(f"  Demos / worker:   {args.num_demos_per_worker}")
    print(f"  Output root:      {output_root}")
    print(f"  Robot:            {resolve_robot_name(args)}")
    print(f"  Auto compact:     {'no' if args.no_compact else 'yes'}")
    if not args.no_compact:
        print(f"  Compact output:   {build_compact_output_path(args, output_root)}")
    print(f"  BC prep:          {'no' if args.no_bc_prep or args.no_compact else 'yes'}")
    print(f"  Forwarded args:   {' '.join(worker_args) if worker_args else '(none)'}")
    print()

    if args.dry_run:
        for job in jobs:
            print(f"  [worker {job['worker_id']}] GPU={job['gpu_id']}")
            print(f"    {' '.join(job['cmd'])}")
            print()
        return

    if not args.no_compact:
        try:
            preflight_compactor(args, output_root)
        except (OSError, RuntimeError) as exc:
            print(f"[compact] {exc}", file=sys.stderr)
            sys.exit(1)

    logs_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    active: list[dict] = []
    interrupted = False

    def _handle_interrupt(signum, frame):
        del signum, frame
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    for job in jobs:
        if interrupted:
            break
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Intentionally NOT setting CUDA_VISIBLE_DEVICES (see comment in the
        # job-construction loop). Strip any inherited mask so workers see all
        # GPUs and CUDA/Omniverse enumerations stay aligned.
        env.pop("CUDA_VISIBLE_DEVICES", None)
        # Cap CPU thread pools per worker. The MPC solver (IPOPT/CasADi) and
        # numpy/BLAS default to using ALL cores via OpenMP/MKL; with N workers
        # in parallel that creates massive oversubscription on a many-core box.
        # Give each worker a fair share of the logical CPUs.
        n_cpu = os.cpu_count() or 8
        threads_per_worker = (
            int(args.cpu_threads_per_worker)
            if args.cpu_threads_per_worker is not None
            else max(1, n_cpu // max(1, num_workers) // 2)
        )
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
            "TBB_NUM_THREADS",
        ):
            env[var] = str(threads_per_worker)
        log_file = open(job["log_path"], "w")
        print(
            f"  [worker {job['worker_id']}] launching on GPU {job['gpu_id']} -> {job['log_path']}"
        )
        proc = subprocess.Popen(
            job["cmd"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        job["proc"] = proc
        job["log_file"] = log_file
        job["start_time"] = time.time()
        active.append(job)
        time.sleep(args.stagger_sec)

    while active:
        still_running: list[dict] = []
        for job in active:
            rc = job["proc"].poll()
            if rc is None:
                still_running.append(job)
                continue
            job["log_file"].close()
            elapsed = time.time() - job["start_time"]
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            job["returncode"] = rc
            print(f"  [worker {job['worker_id']}] finished - {status} ({elapsed:.0f}s)")
        active = still_running

        if interrupted and active:
            print("\n[interrupted] terminating remaining workers...")
            for job in active:
                job["proc"].terminate()
            break

        if active:
            time.sleep(5)

    for job in jobs:
        log_file = job.get("log_file")
        if log_file is not None and not log_file.closed:
            log_file.close()

    compacted_path: Path | None = None
    bc_prep_result = None
    failed_jobs = [
        job for job in jobs if job.get("proc") is not None and job.get("returncode") != 0
    ]
    launched_jobs = [job for job in jobs if job.get("proc") is not None]
    incomplete_dataset_jobs: list[dict] = []
    if (
        not args.no_compact
        and not interrupted
        and len(launched_jobs) == len(jobs)
        and not failed_jobs
    ):
        incomplete_dataset_jobs = validate_worker_dataset_counts(
            args,
            jobs=jobs,
            datasets_dir=datasets_dir,
        )

    if args.no_compact:
        print("[compact] Disabled by --no_compact.")
    elif interrupted:
        print("[compact] Skipped because collection was interrupted.")
    elif len(launched_jobs) != len(jobs):
        print("[compact] Skipped because not all workers were launched.")
    elif failed_jobs:
        failed_ids = ", ".join(str(job["worker_id"]) for job in failed_jobs)
        print(f"[compact] Skipped because worker(s) failed: {failed_ids}.")
    elif incomplete_dataset_jobs:
        failed_ids = ", ".join(
            f"{job['worker_id']}({job.get('dataset_count', 0)}/{args.num_demos_per_worker})"
            for job in incomplete_dataset_jobs
        )
        print(f"[compact] Skipped because worker dataset counts are incomplete: {failed_ids}.")
    else:
        try:
            compacted_path = compact_worker_datasets(args, output_root, datasets_dir)
        except (OSError, RuntimeError) as exc:
            print(f"[compact] {exc}", file=sys.stderr)
            sys.exit(1)

    if args.no_bc_prep:
        print("[bc prep] Disabled by --no_bc_prep.")
    elif compacted_path is None:
        print("[bc prep] Skipped because no compacted trajectory was produced.")
    else:
        try:
            bc_prep_result = prepare_bc_artifacts(
                args,
                output_root=output_root,
                compacted_path=compacted_path,
                worker_args=worker_args,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"[bc prep] {exc}", file=sys.stderr)
            sys.exit(1)

    print()
    print(f"Logs:        {logs_dir}")
    print(f"Datasets:    {datasets_dir}")
    if compacted_path is not None:
        print(f"Trajectories: {compacted_path}")
    if bc_prep_result is not None:
        print(f"BC configs:   {bc_prep_result.config_dir}")
        print(f"BC sweep CSV: {bc_prep_result.csv_path}")
    if interrupted:
        print("Launch interrupted before all workers completed.")


if __name__ == "__main__":
    main()
