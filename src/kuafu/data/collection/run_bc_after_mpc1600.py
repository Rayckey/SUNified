#!/usr/bin/env python3
"""Validate MPC1600 datasets, dry-run BC, then train and publish non-residual priors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py

from kuafu.paths import require_external_artifact_path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
PIPELINE = REPO_ROOT / "src" / "kuafu" / "bc" / "training" / "run_bc_prior_pipeline.py"
AUDITOR = SCRIPT_DIR / "audit_mpc1600.py"
DEFAULT_MANIFEST = REPO_ROOT / "assets/configs" / "campaigns" / "mpc_first_two_programs_1600.json"
ROBOT_IDS = {"kinova": "kinova_gen3", "fr3_finray": "fr3_finray"}
BC_VARIANT = "fff3"
ACTION_CHUNK_SIZE = 10
ACTION_CHUNK_STRIDE = 1
FFF3_HIDDEN_DIMS = [4096, 2048, 4096]
EXPECTED_SWEEP_ROWS = 1


def checkout_env() -> dict[str, str]:
    """Prefer the Kuafu package belonging to this runner's checkout."""
    env = os.environ.copy()
    source_root = REPO_ROOT / "src"
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), existing] if existing else [str(source_root)]
    )
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--provenance-output-root",
        type=Path,
        default=None,
        help=(
            "Original absolute MPC output root for a relocated, "
            "checksum-identical task tree. Defaults to --output-root."
        ),
    )
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--wait-for-pid", type=int, default=None)
    parser.add_argument(
        "--also-wait-for-pid",
        type=int,
        nargs="+",
        default=[],
        help="Additional local collector/offload PIDs that must exit before BC starts.",
    )
    parser.add_argument("--wait-poll-sec", type=float, default=30.0)
    parser.add_argument(
        "--run-name",
        default="first2_programs_mpc1600_fff3_c10_s1_20260818",
    )
    parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=0,
        help=(
            "Periodic checkpoint interval written into all generated configs. "
            "Zero retains only best.pth and final.pth (default: 0)."
        ),
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=20.0,
        help="Minimum free space on the BC log filesystem before each task.",
    )
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument(
        "--revalidate-completed-only",
        action="store_true",
        help=(
            "Read-only validation of already-completed BC artifacts. Runs a fresh "
            "strict MPC audit in memory and does not overwrite the immutable "
            "pre-training audit log recorded by completion.json."
        ),
    )
    return parser.parse_args()


def decoded(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_demos(path: Path) -> int:
    with h5py.File(path, "r") as dataset:
        data = dataset.get("data")
        if data is None:
            return 0
        return sum(name.startswith("demo_") for name in data.keys())


def validate_dataset(path: Path, task: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing MPC1600 dataset: {path}")
    count = count_demos(path)
    if count != 1600:
        raise RuntimeError(f"BC input has {count} demos, expected 1,600: {path}")
    with h5py.File(path, "r") as dataset:
        required_scalars = {
            "task_name": task["task_name"],
            "task_slug": task["slug"],
            "robot_name": task["robot_name"],
        }
        for key, expected in required_scalars.items():
            actual = decoded(dataset.attrs.get(key))
            if actual != expected:
                raise RuntimeError(
                    f"BC input metadata {key!r} is {actual!r}, expected {expected!r}: {path}"
                )
        try:
            per_program = json.loads(str(decoded(dataset.attrs["demos_per_learned_program_json"])))
            program_paths = json.loads(str(decoded(dataset.attrs["learned_program_paths_json"])))
        except (KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"BC input lacks valid learned-program provenance: {path}") from exc
        expected_paths = [program["path"] for program in task["programs"]]
        if per_program != [800, 800] or program_paths != expected_paths:
            raise RuntimeError(
                f"BC input has wrong learned-program provenance: counts={per_program}, "
                f"paths={program_paths}, expected counts=[800, 800], paths={expected_paths}"
            )


def pid_is_alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        return state != "Z"
    except FileNotFoundError:
        return False
    except PermissionError:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False


def wait_for_pid(pid: int, poll_sec: float) -> None:
    if poll_sec <= 0:
        raise ValueError("--wait-poll-sec must be positive")
    print(f"[wait] waiting for MPC1600 supervisor PID {pid}")
    last_report = 0.0
    while pid_is_alive(pid):
        now = time.monotonic()
        if now - last_report >= 300:
            print(f"[wait] MPC1600 supervisor PID {pid} is still active")
            last_report = now
        time.sleep(min(poll_sec, 60.0))
    print(f"[wait] MPC1600 supervisor PID {pid} exited")


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen[Any] | None = None

    def handle_signal(signum: int, frame: Any) -> None:
        del signum, frame
        if process is not None and process.poll() is None:
            process.terminate()
        raise KeyboardInterrupt

    previous_sigint = signal.signal(signal.SIGINT, handle_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, handle_signal)
    try:
        with log_path.open("w") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=checkout_env(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"Command failed with exit code {return_code}; inspect {log_path}"
            )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def enforce_checkpoint_retention(
    csv_path: Path,
    *,
    expected_dataset: Path,
    save_every_n_epochs: int,
    expected_rows: int = EXPECTED_SWEEP_ROWS,
) -> None:
    if save_every_n_epochs < 0:
        raise ValueError("--save-every-n-epochs must be non-negative")
    if not csv_path.is_file():
        raise FileNotFoundError(f"BC dry-run did not create its sweep CSV: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"BC dry-run created {len(rows)} rows, expected {expected_rows}: {csv_path}"
        )
    raw_configs = [(row.get("config") or "").strip() for row in rows]
    if any(not raw for raw in raw_configs) or len(set(raw_configs)) != expected_rows:
        raise RuntimeError(
            f"BC dry-run did not create {expected_rows} unique config paths: {csv_path}"
        )

    expected_dataset = expected_dataset.expanduser().resolve()
    retention_re = re.compile(
        r"^(?P<prefix>[ \t]*save_every_n_epochs:[ \t]*)\d+(?P<suffix>[ \t]*)$",
        re.MULTILINE,
    )
    for row, raw_config in zip(rows, raw_configs, strict=True):
        expected_row = {
            "variant": BC_VARIANT,
            "chunk_size": str(ACTION_CHUNK_SIZE),
            "action_chunk_stride": str(ACTION_CHUNK_STRIDE),
        }
        row_mismatches = {
            key: (str(row.get(key) or "").strip(), value)
            for key, value in expected_row.items()
            if str(row.get(key) or "").strip() != value
        }
        if row_mismatches:
            raise RuntimeError(
                f"Generated BC row is not the required FFF3/chunk-10/stride-1 "
                f"selection: {row_mismatches}: {csv_path}"
            )
        config = resolve_repo_path(raw_config)
        if not config.is_file():
            raise FileNotFoundError(f"Generated BC config is missing: {config}")
        text = config.read_text(encoding="utf-8")
        if (
            f'dataset: "{expected_dataset}"' not in text
            or 'action_mode: "absolute_ee"' not in text
            or not re.search(
                rf"^[ \t]*action_chunk_size:[ \t]*{ACTION_CHUNK_SIZE}[ \t]*$",
                text,
                re.MULTILINE,
            )
            or not re.search(
                rf"^[ \t]*action_chunk_stride:[ \t]*{ACTION_CHUNK_STRIDE}[ \t]*$",
                text,
                re.MULTILINE,
            )
            or f"hidden_dims: {FFF3_HIDDEN_DIMS}" not in text
            or not re.search(r"^[ \t]*num_epochs:[ \t]*150[ \t]*$", text, re.MULTILINE)
        ):
            raise RuntimeError(
                f"Generated BC config has wrong dataset/action mode/epoch count: {config}"
            )
        matches = list(retention_re.finditer(text))
        if len(matches) != 1:
            raise RuntimeError(
                f"Generated BC config has {len(matches)} retention fields, expected 1: {config}"
            )
        updated = retention_re.sub(
            lambda match: (
                f"{match.group('prefix')}{save_every_n_epochs}{match.group('suffix')}"
            ),
            text,
            count=1,
        )
        if updated != text:
            partial = config.with_name(f".{config.name}.partial.{os.getpid()}")
            try:
                partial.write_text(updated, encoding="utf-8")
                os.replace(partial, config)
            finally:
                if partial.exists():
                    partial.unlink()
        configured = config.read_text(encoding="utf-8")
        if not re.search(
            rf"^[ \t]*save_every_n_epochs:[ \t]*{save_every_n_epochs}[ \t]*$",
            configured,
            re.MULTILINE,
        ):
            raise RuntimeError(f"Failed to enforce checkpoint retention: {config}")
    print(
        f"[storage] configured {expected_rows} BC jobs with "
        f"save_every_n_epochs={save_every_n_epochs}"
    )


def require_training_headroom(path: Path, min_free_gib: float) -> int:
    if min_free_gib <= 0:
        raise ValueError("--min-free-gib must be positive")
    storage_path = path.expanduser().resolve()
    probe_path = storage_path
    while not probe_path.exists() and probe_path != probe_path.parent:
        probe_path = probe_path.parent
    free_bytes = shutil.disk_usage(probe_path).free
    required_bytes = int(min_free_gib * 1024**3)
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"BC filesystem has {free_bytes / 1024**3:.1f} GiB free, "
            f"below required {min_free_gib:.1f} GiB: {storage_path}"
        )
    print(
        f"[storage] BC filesystem free={free_bytes / 1024**3:.1f} GiB "
        f"required={min_free_gib:.1f} GiB"
    )
    return free_bytes


def resolve_final_checkpoint(row_name: str, csv_path: Path) -> Path:
    """Resolve a sweep row to the final checkpoint recorded by its job log."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", row_name):
        raise RuntimeError(f"Unsafe/invalid BC sweep row name: {row_name!r}")
    if len(csv_path.parents) < 3:
        raise RuntimeError(f"Cannot infer BC artifact root from sweep CSV: {csv_path}")
    artifact_root = csv_path.parents[2]
    sweep_root = artifact_root / "logs" / "train"
    job_logs = sorted(
        sweep_root.glob(f"*/{row_name}.txt"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for job_log in job_logs:
        matches = re.findall(
            r"^Run dir:[ \t]*(.+?)[ \t]*$",
            job_log.read_text(encoding="utf-8", errors="replace"),
            re.MULTILINE,
        )
        for raw_run_dir in reversed(matches):
            run_dir = Path(raw_run_dir).expanduser()
            if not run_dir.is_absolute():
                run_dir = REPO_ROOT / run_dir
            final_checkpoint = run_dir / "checkpoints" / "final.pth"
            if final_checkpoint.is_file() and final_checkpoint.stat().st_size > 0:
                return require_external_artifact_path(
                    final_checkpoint,
                    "BC final checkpoint",
                )
    raise FileNotFoundError(
        f"No non-empty final.pth is traceable from sweep logs for {row_name!r}"
    )


def validate_sweep_csv(
    csv_path: Path,
    *,
    expected_dataset: Path,
    expected_save_every_n_epochs: int,
    expected_rows: int = EXPECTED_SWEEP_ROWS,
) -> dict[str, Any]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"BC sweep CSV was not created: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"BC sweep has {len(rows)} rows, expected {expected_rows}: {csv_path}"
        )
    for key in ("name", "config", "best_checkpoint", "prior_path"):
        values = [(row.get(key) or "").strip() for row in rows]
        if any(not value for value in values):
            raise RuntimeError(f"BC sweep contains an empty {key!r} field: {csv_path}")
        if len(set(values)) != expected_rows:
            raise RuntimeError(f"BC sweep {key!r} values are not unique: {csv_path}")
    missing = [row.get("name", "<unnamed>") for row in rows if not (row.get("best_checkpoint") or "").strip()]
    if missing:
        raise RuntimeError(
            f"BC sweep has {len(missing)} rows without best_checkpoint: {missing[:5]}"
        )
    expected_dataset = expected_dataset.expanduser().resolve()
    config_paths: list[Path] = []
    checkpoint_paths: list[Path] = []
    final_checkpoint_paths: list[Path] = []
    published_paths: list[Path] = []
    for row in rows:
        row_name = (row.get("name") or "").strip()
        expected_row = {
            "variant": BC_VARIANT,
            "chunk_size": str(ACTION_CHUNK_SIZE),
            "action_chunk_stride": str(ACTION_CHUNK_STRIDE),
        }
        row_mismatches = {
            key: (str(row.get(key) or "").strip(), value)
            for key, value in expected_row.items()
            if str(row.get(key) or "").strip() != value
        }
        if row_mismatches:
            raise RuntimeError(
                f"BC sweep row {row_name!r} is not the required "
                f"FFF3/chunk-10/stride-1 selection: {row_mismatches}"
            )
        raw_dataset = (row.get("dataset") or "").strip()
        if not raw_dataset or Path(raw_dataset).expanduser().resolve() != expected_dataset:
            raise RuntimeError(
                f"BC sweep row {row.get('name')} targets {raw_dataset!r}, "
                f"expected {expected_dataset}: {csv_path}"
            )
        raw_config = (row.get("config") or "").strip()
        config = Path(raw_config).expanduser()
        if not config.is_absolute():
            config = REPO_ROOT / config
        if not config.is_file() or config.stat().st_size == 0:
            raise FileNotFoundError(
                f"BC configuration is missing or empty for {row.get('name')}: {config}"
            )
        config_text = config.read_text(encoding="utf-8")
        if (
            f'dataset: "{expected_dataset}"' not in config_text
            or 'action_mode: "absolute_ee"' not in config_text
            or not re.search(
                rf"^[ \t]*action_chunk_size:[ \t]*{ACTION_CHUNK_SIZE}[ \t]*$",
                config_text,
                re.MULTILINE,
            )
            or not re.search(
                rf"^[ \t]*action_chunk_stride:[ \t]*{ACTION_CHUNK_STRIDE}[ \t]*$",
                config_text,
                re.MULTILINE,
            )
            or f"hidden_dims: {FFF3_HIDDEN_DIMS}" not in config_text
            or not re.search(
                r"^[ \t]*num_epochs:[ \t]*150[ \t]*$",
                config_text,
                re.MULTILINE,
            )
            or not re.search(
                rf"^[ \t]*save_every_n_epochs:[ \t]*{expected_save_every_n_epochs}[ \t]*$",
                config_text,
                re.MULTILINE,
            )
        ):
            raise RuntimeError(
                f"BC configuration has the wrong dataset, action mode, or "
                f"checkpoint retention: {config}"
            )
        config_paths.append(config.resolve())

        raw_path = (row.get("best_checkpoint") or "").strip()
        checkpoint = Path(raw_path).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = REPO_ROOT / checkpoint
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(
                f"BC sweep checkpoint does not exist for {row.get('name')}: {checkpoint}"
            )
        checkpoint_paths.append(checkpoint.resolve())
        published_raw = (row.get("prior_path") or "").strip()
        if not published_raw:
            raise RuntimeError(f"BC sweep row was not published: {row.get('name')}")
        published = Path(published_raw).expanduser()
        if not published.is_absolute():
            published = REPO_ROOT / published
        if not published.is_file() or published.stat().st_size == 0:
            raise FileNotFoundError(
                f"Published BC prior does not exist for {row.get('name')}: {published}"
            )
        if checkpoint.resolve() != published.resolve():
            raise RuntimeError(
                f"BC best checkpoint was not published in-place for {row_name}: "
                f"checkpoint={checkpoint}, prior={published}"
            )
        published_paths.append(published.resolve())
        final_checkpoint_paths.append(resolve_final_checkpoint(row_name, csv_path))

    if len(set(final_checkpoint_paths)) != expected_rows:
        raise RuntimeError(
            f"BC sweep has {len(set(final_checkpoint_paths))} unique final checkpoints, "
            f"expected {expected_rows}: {csv_path}"
        )

    return {
        "sweep_csv_sha256": sha256_file(csv_path),
        "sweep_rows": len(rows),
        "unique_config_count": len(set(config_paths)),
        "checkpoint_count": len(set(checkpoint_paths)),
        "final_checkpoint_count": len(set(final_checkpoint_paths)),
        "final_checkpoint_bytes": sum(path.stat().st_size for path in final_checkpoint_paths),
        "published_prior_count": len(set(published_paths)),
        "bc_variant": BC_VARIANT,
        "action_chunk_size": ACTION_CHUNK_SIZE,
        "action_chunk_stride": ACTION_CHUNK_STRIDE,
        "hidden_dims": FFF3_HIDDEN_DIMS,
    }


def write_completion(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.expanduser().resolve().read_text())
    by_slug = {task["slug"]: task for task in manifest["tasks"]}
    unknown = [slug for slug in args.tasks if slug not in by_slug]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}; choices: {list(by_slug)}")
    if not args.gpus or any(gpu < 0 for gpu in args.gpus):
        raise ValueError(f"Invalid GPU list: {args.gpus}")
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be positive")
    if args.save_every_n_epochs < 0:
        raise ValueError("--save-every-n-epochs must be non-negative")
    if args.min_free_gib <= 0:
        raise ValueError("--min-free-gib must be positive")
    if args.dry_run_only and args.revalidate_completed_only:
        raise ValueError(
            "--dry-run-only and --revalidate-completed-only are mutually exclusive"
        )

    if not (args.dry_run_only or args.revalidate_completed_only):
        wait_pids = []
        if args.wait_for_pid is not None:
            wait_pids.append(args.wait_for_pid)
        wait_pids.extend(args.also_wait_for_pid)
        for pid in dict.fromkeys(wait_pids):
            wait_for_pid(pid, args.wait_poll_sec)

    output_root = require_external_artifact_path(
        args.output_root,
        "MPC and BC artifact root",
    )
    provenance_output_root = require_external_artifact_path(
        (
            args.provenance_output_root
            if args.provenance_output_root is not None
            else output_root
        ),
        "MPC provenance artifact root",
    )
    relocation_metadata = (
        {"mpc_provenance_output_root": str(provenance_output_root)}
        if args.provenance_output_root is not None
        else {}
    )
    audit_slug = "_".join(args.tasks)
    audit_log = output_root / "bc_training" / f"audit_mpc1600_{audit_slug}.log"
    print(
        f"[audit] validating exact 800+800 datasets and aggregate provenance: "
        f"{', '.join(args.tasks)}"
    )
    audit_command = [
        sys.executable,
        str(AUDITOR),
        "--manifest",
        str(args.manifest.expanduser().resolve()),
        "--output-root",
        str(output_root),
        "--tasks",
        *args.tasks,
    ]
    if args.provenance_output_root is not None:
        audit_command.extend(
            ["--provenance-output-root", str(provenance_output_root)]
        )
    if args.revalidate_completed_only:
        audit_process = subprocess.run(
            audit_command,
            cwd=REPO_ROOT,
            env=checkout_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if audit_process.returncode != 0:
            raise RuntimeError(
                "Read-only strict MPC revalidation failed with exit code "
                f"{audit_process.returncode}: {audit_process.stderr}"
            )
        audit_report = json.loads(audit_process.stdout)
    else:
        run_logged(audit_command, audit_log)
        audit_report = json.loads(audit_log.read_text())
    if set(audit_report) != set(args.tasks):
        raise RuntimeError(
            f"Strict audit report tasks {sorted(audit_report)} do not match "
            f"requested tasks {sorted(args.tasks)}: {audit_log}"
        )
    audit_log_sha256 = sha256_file(audit_log)

    for slug in args.tasks:
        task = by_slug[slug]
        robot_id = ROBOT_IDS[task["robot_name"]]
        dataset = (
            output_root
            / slug
            / f"{task['robot_name']}_{slug}_first2_programs_mpc1600.hdf5"
        )
        validate_dataset(dataset, task)

        bc_root = output_root / "bc_training" / slug
        completion_path = bc_root / "completion.json"
        artifact_root = bc_root / "prior_artifacts" / robot_id
        csv_path = (
            artifact_root
            / "prior_sweep_csv"
            / "generated"
            / f"{slug}_{args.run_name}.csv"
        )
        if completion_path.exists():
            sweep_summary = validate_sweep_csv(
                csv_path,
                expected_dataset=dataset,
                expected_save_every_n_epochs=args.save_every_n_epochs,
            )
            completion = json.loads(completion_path.read_text())
            expected_completion = {
                "task_name": task["task_name"],
                "task_slug": slug,
                "robot_id": robot_id,
                "dataset": str(dataset),
                "dataset_sha256": audit_report[slug]["sha256"],
                "demos_per_learned_program": [800, 800],
                "audit_log": str(audit_log),
                "audit_log_sha256": audit_log_sha256,
                "sweep_csv": str(csv_path),
                "mode": "train",
                "residual_launcher": False,
                "save_every_n_epochs": args.save_every_n_epochs,
                **relocation_metadata,
                **sweep_summary,
            }
            mismatches = {
                key: (completion.get(key), value)
                for key, value in expected_completion.items()
                if completion.get(key) != value
            }
            if mismatches:
                raise RuntimeError(
                    f"BC completion record disagrees with artifacts for {slug}: {mismatches}"
                )
            print(f"[skip] {slug}: validated completed BC sweep {csv_path}")
            continue

        if args.revalidate_completed_only:
            raise FileNotFoundError(
                f"BC completion record is required for read-only revalidation: "
                f"{completion_path}"
            )

        command = [
            sys.executable,
            str(PIPELINE),
            "--dataset",
            str(dataset),
            "--task_name",
            task["task_name"],
            "--robot_id",
            robot_id,
            "--task_slug",
            slug,
            "--mode",
            "train",
            "--run_name",
            args.run_name,
            "--artifact_root",
            str(artifact_root),
            "--priors_root",
            str(artifact_root / "published_priors"),
            "--action_mode",
            "absolute_ee",
            "--chunk_sizes",
            str(ACTION_CHUNK_SIZE),
            "--chunk_strides",
            str(ACTION_CHUNK_STRIDE),
            "--variants",
            BC_VARIANT,
            "--gpus",
            *(str(gpu) for gpu in args.gpus),
            "--workers_per_gpu",
            str(args.workers_per_gpu),
            "--python",
            sys.executable,
            "--no_residual_launcher",
        ]
        print(f"[bc-dry-run] {slug}: dataset validated at 800+800=1,600")
        run_logged([*command, "--dry_run"], bc_root / "pipeline_dry_run.log")
        enforce_checkpoint_retention(
            csv_path,
            expected_dataset=dataset,
            save_every_n_epochs=args.save_every_n_epochs,
        )
        if args.dry_run_only:
            continue
        free_bytes_before_training = require_training_headroom(
            artifact_root,
            args.min_free_gib,
        )
        print(
            f"[bc-train] {slug}: starting non-residual {BC_VARIANT.upper()} "
            f"with action chunk size {ACTION_CHUNK_SIZE} and stride "
            f"{ACTION_CHUNK_STRIDE}"
        )
        run_logged(command, bc_root / "pipeline_train.log")
        sweep_summary = validate_sweep_csv(
            csv_path,
            expected_dataset=dataset,
            expected_save_every_n_epochs=args.save_every_n_epochs,
        )
        write_completion(
            completion_path,
            {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "task_name": task["task_name"],
                "task_slug": slug,
                "robot_id": robot_id,
                "dataset": str(dataset),
                "dataset_sha256": audit_report[slug]["sha256"],
                "demos_per_learned_program": [800, 800],
                "audit_log": str(audit_log),
                "audit_log_sha256": audit_log_sha256,
                "sweep_csv": str(csv_path),
                "mode": "train",
                "residual_launcher": False,
                "save_every_n_epochs": args.save_every_n_epochs,
                "free_bytes_before_training": free_bytes_before_training,
                **relocation_metadata,
                **sweep_summary,
            },
        )
        print(
            f"[bc-complete] {slug}: {EXPECTED_SWEEP_ROWS}/{EXPECTED_SWEEP_ROWS} "
            "FFF3 chunk-10 checkpoint validated"
        )


if __name__ == "__main__":
    main()
