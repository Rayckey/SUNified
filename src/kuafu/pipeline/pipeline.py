"""End-to-end stage orchestration over the vendored research components."""

from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from kuafu.artifact_contracts import learned_program_identity, evaluation_identity, validate_program_metadata
from kuafu.skills.constraints.tolerance_policy import policy_metadata

from .config import ConfigError, PipelineConfig, project_root
from .runtime import CommandRunner, command_text, pipeline_environment
from .state import StateStore, utc_now


STAGES = ("learn-programs", "collect", "train-prior", "eval-prior", "train-residual")

RESIDUAL_TASK_SUFFIXES = {
    "can": "learned_can",
    "drawer": "learned_drawer",
    "drawer_stack": "learned_drawer_stack",
    "estop_close": "learned_estop_close",
    "estop_open": "learned_estop_open",
    "hang_cup": "learned_hang_cup",
    "pen_cup": "learned_pen_cup",
    "stack2": "learned_stack_2",
    "stack_2": "learned_stack_2",
    "stack3": "learned_stack_3",
    "stack_3": "learned_stack_3",
}
ROBOT_PREFIXES = {
    "fr3_finray": "fr3_finray",
    "franka_panda": "panda",
    "kinova_gen3": "kinova",
}


class PipelineError(RuntimeError):
    """Raised when a stage contract cannot be satisfied."""


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return token or "artifact"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_demos(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        import h5py

        with h5py.File(path, "r") as dataset:
            data = dataset.get("data")
            if data is None:
                return 0
            return sum(name.startswith("demo_") for name in data.keys())
    except (OSError, ImportError):
        return 0


def _as_bool(mapping: dict[str, Any], key: str, default: bool = False) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"Expected boolean for {key}, got {value!r}")
    return value


class Pipeline:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        dry_run: bool = False,
        force: bool = False,
        prepare_only: bool = False,
    ):
        self.config = config
        self.dry_run = dry_run
        self.force = force
        self.prepare_only = prepare_only
        self.runner = CommandRunner(config, dry_run=dry_run)
        self.state = StateStore(
            config.task_artifact_root / "pipeline_state.json",
            config.task_slug,
        )

    @property
    def isaaclab_sh(self) -> Path:
        return self.config.isaaclab_root / "isaaclab.sh"

    def run(self, stages: Iterable[str]) -> None:
        for stage in stages:
            self.run_stage(stage)

    def run_stage(self, stage: str) -> dict[str, Any]:
        if stage not in STAGES:
            raise PipelineError(f"Unknown stage {stage!r}; expected one of {STAGES}")
        if not self.dry_run and not self.force and self.state.stage_complete(stage):
            record = self.state.data["stages"][stage]
            try:
                files = self._stage_files(stage, record.get("artifacts", {}))
                inputs = self._stage_inputs(stage)
                if record.get("inputs") and record["inputs"] != inputs:
                    raise ConfigError(f"{stage} inputs changed since completion. Use a new task/run identity, or explicitly rerun with --force.")
                valid = bool(record.get("files")) and record.get("files") == files and record.get("inputs") == inputs
            except ConfigError:
                raise
            except (OSError, ValueError, PipelineError) as error:
                print(f"[resume] {stage}: {error}", flush=True)
                valid = False
            if valid:
                print(f"[skip] {stage}: verified inputs and output artifacts", flush=True)
                return dict(record.get("artifacts", {}))
            print(f"[resume] {stage}: completion record is stale; rechecking/running the stage", flush=True)

        implementation = {
            "learn-programs": self.learn_programs,
            "collect": self.collect,
            "train-prior": self.train_prior,
            "eval-prior": self.eval_prior,
            "train-residual": self.train_residual,
        }[stage]

        if not self.dry_run:
            for downstream in STAGES[STAGES.index(stage) + 1:]:
                record = self.state.data.get("stages", {}).get(downstream, {})
                if record.get("status") in {"complete", "prepared"}:
                    record["status"] = "stale"
            self.state.stage_started(stage)
        try:
            artifacts = implementation()
            files = self._stage_files(stage, artifacts) if not self.dry_run else {}
            inputs = self._stage_inputs(stage) if not self.dry_run else ""
        except BaseException as error:
            if not self.dry_run:
                self.state.stage_failed(stage, error)
            raise
        if not self.dry_run:
            status = "prepared" if stage == "train-residual" and not artifacts.get("residual_executed") else "complete"
            self.state.stage_finished(stage, artifacts=artifacts, status=status, inputs=inputs, files=files)
        return artifacts

    def _execution_program_path(self) -> Path:
        paths = self._configured_program_paths()
        explicit = str(self.config.section("eval").get("program_path", "")).strip()
        if explicit:
            selected = self.config.resolve_path(explicit)
            if selected not in paths:
                raise PipelineError("[eval].program_path must select one of the collected learned programs")
            return selected
        if len(paths) != 1:
            raise PipelineError("Multiple programs were collected: set [eval].program_path explicitly; residual training uses the same program.")
        return paths[0]

    def _csv_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise PipelineError(f"Empty prior sweep: {path}")
        return rows

    def _stage_inputs(self, stage: str) -> str:
        section_name = dict(zip(STAGES, ("learn_programs", "collect", "prior", "eval", "residual"), strict=True))[stage]
        payload: dict[str, Any] = {"task": self.config.section("task"), "settings": dict(self.config.section(section_name))}
        payload["settings"].pop("execute", None)
        if stage in {"learn-programs", "eval-prior", "train-residual"}:
            payload["automatic_tolerance_policy"] = policy_metadata()
        if stage != "learn-programs":
            payload["programs"] = [learned_program_identity(p) for p in self._configured_program_paths()]
        if stage in {"train-prior", "eval-prior", "train-residual"}:
            payload["dataset_sha256"] = _sha256(self._dataset_path())
        if stage in {"eval-prior", "train-residual"}:
            payload["execution_program"] = learned_program_identity(self._execution_program_path())
            payload["priors"] = [(r.get("name"), r.get("best_checkpoint"), _sha256(Path(r["best_checkpoint"])))
                                 for r in self._csv_rows(self._prior_csv())]
            configured_params = str(self.config.section("eval").get("params_path", "")).strip()
            if configured_params:
                payload["params_sha256"] = _sha256(self.config.resolve_artifact_path(configured_params, field="[eval].params_path"))
            if stage == "train-residual" or _as_bool(self.config.section("eval"), "acquire_weights", True):
                residual = self.config.section("residual")
                payload["weight_selection"] = {key: residual.get(key) for key in ("selection_policy", "success_threshold", "model_class_order")}
        if stage == "train-residual":
            payload["weights_sha256"] = _sha256(self._weights_path())
            payload["evaluation"] = self._csv_rows(self._prior_csv())
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _validate_dataset(self, path: Path) -> None:
        import h5py

        expected = [str(p) for p in self._configured_program_paths()]
        if _count_demos(path) <= 0:
            raise PipelineError(f"Missing or empty collected dataset: {path}")
        with h5py.File(path, "r") as dataset:
            actual = json.loads(dataset.attrs.get("learned_program_paths_json", "[]"))
            if actual != expected:
                raise PipelineError(f"Dataset learned-program provenance does not match collection config: {path}")
            identities = dataset.attrs.get("learned_program_identities_json")
            if identities and json.loads(identities) != [learned_program_identity(p) for p in expected]:
                raise PipelineError(f"Learned-program contents changed after collection: {path}; collect into a new run")

    def _stage_files(self, stage: str, artifacts: dict[str, Any]) -> dict[str, str]:
        required: list[Path] = []
        if stage == "learn-programs":
            for value in artifacts.get("learned_program_paths", []):
                path = Path(value)
                if not self._learned_program_is_valid(
                    path,
                    allow_legacy=path.resolve() in {
                        self.config.resolve_path(item)
                        for item in self.config.string_list("collect", "program_paths") if item.strip()
                    },
                ):
                    raise PipelineError(f"Missing successful learned program: {path}")
                required.extend(p for p in (path / "program").rglob("*") if p.is_file())
        if stage in {"collect", "train-prior", "eval-prior", "train-residual"}:
            dataset = Path(artifacts.get("dataset") or self._dataset_path())
            self._validate_dataset(dataset)
            required.append(dataset)
        if stage in {"train-prior", "eval-prior", "train-residual"}:
            csv_path = Path(artifacts.get("prior_csv") or self._prior_csv())
            rows = self._csv_rows(csv_path)
            for row in rows:
                for key in ("best_checkpoint", "config"):
                    value = row.get(key, "").strip()
                    if not value:
                        raise PipelineError(f"Prior {row.get('name')!r} has no {key}: {csv_path}")
                    required.append(Path(value))
                if row.get("published_prior_dir"):
                    required.append(Path(row["published_prior_dir"]) / "train_config.yaml")
                if stage in {"eval-prior", "train-residual"}:
                    stats_path = Path(row.get("bc_eval_stats_path") or "")
                    if not stats_path.is_file():
                        raise PipelineError(f"Missing evaluation statistics for {row.get('name')}: {stats_path}")
                    stats = json.loads(stats_path.read_text())
                    validate_program_metadata(stats, learned_program_identity(self._execution_program_path()))
                    eval_cfg = self.config.section("eval")
                    identity = evaluation_identity(
                        checkpoint=row["best_checkpoint"], task=self.config.task_name,
                        program_path=str(self._execution_program_path()), params_path=stats.get("params_path"),
                        duration_sec=float(eval_cfg.get("duration_sec", 180.0)), num_envs=int(eval_cfg.get("num_envs", 8192)),
                        chunk_exec=int(eval_cfg.get("chunk_exec", 1)),
                    )
                    if stats.get("evaluation_identity") != identity or row.get("bc_eval_identity") != identity:
                        raise PipelineError(f"Evaluation inputs changed for {row.get('name')}; rerun eval-prior")
                    rate_column = str(eval_cfg.get("success_rate_col") or
                                      ("success_rate" if self.config.robot_id == "fr3_finray" else f"success_rate_{self.config.robot_id}"))
                    recorded_rate = float(row.get(rate_column) or "nan")
                    measured_rate = float(stats.get("success_rate_pct", "nan"))
                    if not (math.isfinite(recorded_rate) and 0 <= recorded_rate <= 100 and
                            math.isclose(recorded_rate, round(measured_rate, 1), abs_tol=0.051)):
                        raise PipelineError(f"CSV success rate does not match evaluation statistics for {row.get('name')}")
                    required.append(stats_path)
        for key in ("residual_params", "residual_weights", "residual_launcher", "residual_checkpoint", "residual_completion"):
            if artifacts.get(key):
                required.append(Path(artifacts[key]))
        if stage == "train-residual" and artifacts.get("residual_executed") and not artifacts.get("residual_checkpoint"):
            raise PipelineError("Residual training returned without a verified checkpoint")
        if not required:
            raise PipelineError(f"{stage} did not produce any verifiable outputs")
        return {str(path.resolve()): _sha256(path) for path in required}

    def doctor(self) -> tuple[list[str], list[str]]:
        root = project_root()
        required = (
            root / "src" / "kuafu" / "mpc" / "controller" / "mpc.py",
            root / "src" / "kuafu" / "programs",
            root / "src" / "kuafu" / "data" / "collection" / "launch_collect_from_mpc.py",
            root / "src" / "kuafu" / "bc" / "training" / "run_bc_prior_pipeline.py",
            root / "src" / "kuafu" / "bc" / "training" / "evaluate_sweep.py",
            root / "scripts" / "residual_rl" / "train_residual_hparams.py",
            self.isaaclab_sh,
            Path(self.config.python),
        )
        errors = [f"missing: {path}" for path in required if not path.exists()]
        warnings: list[str] = []
        for key, stages in (
            ("OPENAI_API_KEY", "learn-programs/residual"),
            ("GEMINI_API_KEY", "optional residual video feedback"),
        ):
            if not os.environ.get(key):
                warnings.append(f"{key} is unset ({stages})")
        if self.config.artifact_root.exists() and not os.access(self.config.artifact_root, os.W_OK):
            errors.append(f"artifact root is not writable: {self.config.artifact_root}")
        return errors, warnings

    def _isaac_command(self, script: Path, args: Iterable[str]) -> list[str]:
        return [str(self.isaaclab_sh), "-p", str(script), *[str(value) for value in args]]

    def _learned_program_is_valid(self, path: Path, *, allow_legacy: bool = False) -> bool:
        if not path.is_dir() or not (path / "program").is_dir():
            return False
        success_file = path / "success.txt"
        if success_file.is_file() and success_file.read_text(encoding="utf-8").strip().lower() != "true":
            return False
        stats_file = path / "stats.txt"
        if stats_file.is_file() and "status: succeeded" not in stats_file.read_text(encoding="utf-8"):
            return False
        if success_file.is_file() or stats_file.is_file():
            return True
        # Explicitly selected archived programs predate run-level success markers.
        # Accept a complete stage program, but never discover an unfinished new
        # formulation run merely because it has a program directory.
        if not allow_legacy:
            return False
        stages = list((path / "program").glob("subtask_*"))
        if not stages or any(not stage.is_dir() or not stage.name[8:].isdigit() for stage in stages):
            return False
        if sorted(int(stage.name[8:]) for stage in stages) != list(range(len(stages))):
            return False
        try:
            for stage in stages:
                for name in ("task.txt", "atomic.txt"):
                    if not (stage / name).read_text(encoding="utf-8").strip():
                        return False
                status = stage / "status.txt"
                if status.is_file() and "status: succeeded" not in status.read_text(encoding="utf-8"):
                    return False
                constraints = json.loads((stage / "constraints.json").read_text(encoding="utf-8"))
                if not isinstance(constraints, dict) or not isinstance(constraints.get("Constraints"), list):
                    return False
                if not constraints["Constraints"] or any(not isinstance(item, dict) for item in constraints["Constraints"]):
                    return False
        except (OSError, ValueError):
            return False
        return True

    def _configured_program_paths(self) -> list[Path]:
        raw = self.config.string_list("collect", "program_paths")
        paths = [self.config.resolve_path(item) for item in raw if item.strip()]
        explicitly_selected = bool(paths)
        if not paths:
            state_paths = self.state.artifacts.get("learned_program_paths", [])
            if isinstance(state_paths, list):
                paths = [Path(item).expanduser().resolve() for item in state_paths]
        if not paths and self.dry_run:
            env_name = str(self.config.section("learn_programs").get("output_env", "learned-task"))
            paths = [self.config.task_artifact_root / "programs" / env_name / "<successful-run>"]
        if not paths:
            raise PipelineError(
                "No learned programs are available. Set [collect].program_paths or run learn-programs first."
            )
        if not self.dry_run:
            invalid = [path for path in paths if not self._learned_program_is_valid(path, allow_legacy=explicitly_selected)]
            if invalid:
                raise PipelineError(f"Invalid or unsuccessful learned-program paths: {invalid}")
        return paths

    def learn_programs(self) -> dict[str, Any]:
        section = self.config.section("learn_programs")
        if not _as_bool(section, "enabled", True):
            print("[learn-programs] disabled by configuration", flush=True)
            existing = self._configured_program_paths()
            return {"learned_program_paths": [str(path) for path in existing]}

        script = self.config.resolve_path(str(section["script"]))
        args = self.config.string_list("learn_programs", "args")
        output_env = str(section.get("output_env", "")).strip()
        if not output_env:
            raise ConfigError("[learn_programs].output_env is required when learn-programs is enabled")
        output_root = self.config.task_artifact_root / "programs" / output_env
        command = self._isaac_command(script, args)
        self.runner.run("learn-programs", command)
        if self.dry_run:
            predicted = output_root / "<timestamp>"
            return {"learned_program_paths": [str(predicted)]}

        candidates = sorted(
            (path for path in output_root.glob("*") if self._learned_program_is_valid(path)),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise PipelineError(f"Learn-programs command completed without a successful run under {output_root}")
        selected = candidates[0].resolve()
        accumulated = self.state.artifacts.get("learned_program_paths", [])
        paths = [str(item) for item in accumulated] if isinstance(accumulated, list) else []
        if str(selected) not in paths:
            paths.append(str(selected))
        print(f"[learn-programs] selected {selected}", flush=True)
        return {"learned_program_paths": paths}

    def _collection_paths(self, program_paths: list[Path]) -> tuple[list[Path], list[Path], Path]:
        section = self.config.section("collect")
        target = int(section["successful_demos_per_program"])
        collection_root = self.config.task_artifact_root / "collection"
        program_roots: list[Path] = []
        program_outputs: list[Path] = []
        for index, program in enumerate(program_paths, start=1):
            name = _safe_token(program.name)
            program_roots.append(collection_root / f"program_{index:02d}_{name}")
            program_outputs.append(collection_root / f"program_{index:02d}_{name}_mpc{target}.hdf5")
        total = target * len(program_paths)
        configured = str(section.get("dataset", "")).strip()
        final_output = (
            self.config.resolve_artifact_path(configured, field="[collect].dataset")
            if configured
            else self.config.task_artifact_root / "datasets" / f"{self.config.task_slug}_mpc{total}.hdf5"
        )
        return program_roots, program_outputs, final_output

    def _collection_command(
        self,
        *,
        program_path: Path,
        program_root: Path,
        program_output: Path,
        program_index: int,
    ) -> list[str]:
        section = self.config.section("collect")
        gpus = self.config.gpus
        workers_per_gpu = int(section["workers_per_gpu_per_program"])
        target = int(section["successful_demos_per_program"])
        worker_count = len(gpus) * workers_per_gpu
        if target % worker_count:
            raise ConfigError(
                f"successful_demos_per_program={target} must be divisible by {worker_count} workers"
            )
        base_seeds = section.get("base_seeds", [])
        if base_seeds:
            if not isinstance(base_seeds, list) or program_index >= len(base_seeds):
                raise ConfigError("[collect].base_seeds must provide one seed per learned program")
            base_seed = int(base_seeds[program_index])
        else:
            base_seed = int(section.get("base_seed", 100000)) + program_index * 10000

        launcher = project_root() / "src" / "kuafu" / "data" / "collection" / "launch_collect_from_mpc.py"
        args = [
            "--gpus",
            *[str(gpu) for gpu in gpus],
            "--workers_per_gpu",
            str(workers_per_gpu),
            "--num_demos_per_worker",
            str(target // worker_count),
            "--cpu_threads_per_worker",
            str(int(section.get("cpu_threads_per_worker", 8))),
            "--output_root",
            str(program_root),
            "--task_name",
            self.config.task_name,
            "--robot_name",
            self.config.robot_name,
            "--base_seed",
            str(base_seed),
            "--stagger_sec",
            str(float(section.get("worker_stagger_sec", 8.0))),
            "--compact_output",
            str(program_output),
            "--no_bc_prep",
            "--program_path",
            str(program_path),
            "--control_output_mode",
            str(section.get("control_output_mode", "absolute_ee")),
            "--action_noise_std",
            str(float(section.get("action_noise_std", 0.003))),
            *self.config.string_list("collect", "extra_args"),
        ]
        return self._isaac_command(launcher, args)

    def _run_collection_launchers(self, commands: list[tuple[list[str], Path]], parallel: bool) -> None:
        if self.dry_run:
            for index, (command, log_path) in enumerate(commands, start=1):
                print(f"[collect-program-{index}] log={log_path}")
                self.runner.run(f"collect-program-{index}", command)
            return

        processes: list[tuple[subprocess.Popen[str], Any, Path]] = []
        offset = float(self.config.section("collect").get("program_start_offset_sec", 4.0))
        try:
            for index, (command, log_path) in enumerate(commands, start=1):
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = log_path.open("w", encoding="utf-8")
                print(f"[collect-program-{index}] {command_text(command)}", flush=True)
                process = subprocess.Popen(
                    command,
                    cwd=project_root(),
                    env=pipeline_environment(self.config),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                processes.append((process, handle, log_path))
                if not parallel:
                    code = process.wait()
                    handle.close()
                    if code:
                        raise PipelineError(f"Collection launcher failed ({code}); see {log_path}")
                elif index < len(commands):
                    time.sleep(offset)
            if parallel:
                for process, handle, log_path in processes:
                    code = process.wait()
                    handle.close()
                    if code:
                        raise PipelineError(f"Collection launcher failed ({code}); see {log_path}")
        except BaseException:
            for process, _handle, _log_path in processes:
                if process.poll() is None:
                    process.terminate()
            for process, handle, _log_path in processes:
                if process.poll() is None:
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                if not handle.closed:
                    handle.close()
            raise

    def collect(self) -> dict[str, Any]:
        section = self.config.section("collect")
        program_paths = self._configured_program_paths()
        program_roots, program_outputs, final_output = self._collection_paths(program_paths)
        per_program_target = int(section["successful_demos_per_program"])
        final_target = per_program_target * len(program_paths)

        if not self.dry_run:
            existing_final = _count_demos(final_output)
            if existing_final == final_target and not self.force:
                self._validate_dataset(final_output)
                checksum = _sha256(final_output)
                print(f"[collect] reuse {final_output} ({final_target} demos)", flush=True)
                return {"dataset": str(final_output), "dataset_sha256": checksum}
            if existing_final not in (0, final_target):
                raise PipelineError(
                    f"Refusing to overwrite incomplete final dataset: {final_output} ({existing_final} demos)"
                )

        commands: list[tuple[list[str], Path]] = []
        for index, (program, program_root, program_output) in enumerate(
            zip(program_paths, program_roots, program_outputs, strict=True)
        ):
            if not self.dry_run and _count_demos(program_output) == per_program_target and not self.force:
                print(f"[collect] reuse {program_output} ({per_program_target} demos)", flush=True)
                continue
            command = self._collection_command(
                program_path=program,
                program_root=program_root,
                program_output=program_output,
                program_index=index,
            )
            commands.append((command, program_root.parent / f"launcher_program{index + 1:02d}.log"))

        self._run_collection_launchers(
            commands,
            parallel=_as_bool(section, "parallel_programs", True),
        )

        combiner = project_root() / "src" / "kuafu" / "data" / "datasets" / "combine_hdf5.py"
        partial = final_output.with_name(f".{final_output.stem}.partial{final_output.suffix}")
        combine_command = [
            self.config.python,
            str(combiner),
            *[str(path) for path in program_outputs],
            "-o",
            str(partial),
            "--robot-name",
            self.config.robot_name,
        ]
        if self.dry_run:
            self.runner.run("combine", combine_command)
            print(f"[combine] promote validated dataset to {final_output}", flush=True)
            return {
                "dataset": str(final_output),
                "learned_program_paths": [str(path) for path in program_paths],
            }

        counts = [_count_demos(path) for path in program_outputs]
        if counts != [per_program_target] * len(program_outputs):
            raise PipelineError(f"Expected {per_program_target} demos per learned program, got {counts}")
        final_output.parent.mkdir(parents=True, exist_ok=True)
        if partial.exists():
            partial.unlink()
        self.runner.run("combine", combine_command)
        if _count_demos(partial) != final_target:
            raise PipelineError(f"Combined dataset is not exactly {final_target} demos: {partial}")

        import h5py

        with h5py.File(partial, "a") as dataset:
            dataset.attrs["kuafu_pipeline_schema"] = 1
            dataset.attrs["task_slug"] = self.config.task_slug
            dataset.attrs["task_name"] = self.config.task_name
            dataset.attrs["learned_program_paths_json"] = json.dumps([str(path) for path in program_paths])
            dataset.attrs["learned_program_identities_json"] = json.dumps([learned_program_identity(path) for path in program_paths])
            dataset.attrs["demos_per_learned_program_json"] = json.dumps(counts)
        os.replace(partial, final_output)
        checksum = _sha256(final_output)
        checksum_path = final_output.with_suffix(final_output.suffix + ".sha256")
        checksum_path.write_text(f"{checksum}  {final_output.name}\n", encoding="utf-8")
        completion = {
            "completed_at_utc": utc_now(),
            "dataset": str(final_output),
            "demo_count": final_target,
            "sha256": checksum,
            "learned_program_paths": [str(path) for path in program_paths],
            "demos_per_learned_program": counts,
        }
        (final_output.parent / "collection_completion.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if not _as_bool(section, "keep_intermediates", False):
            for program_root in program_roots:
                for worker_file in sorted((program_root / "datasets").glob("mpc_demos_worker*.hdf5")):
                    worker_file.unlink()
            for program_output in program_outputs:
                if program_output.exists():
                    program_output.unlink()

        return {
            "dataset": str(final_output),
            "dataset_sha256": checksum,
            "learned_program_paths": [str(path) for path in program_paths],
        }

    def _dataset_path(self) -> Path:
        configured = str(self.config.section("prior").get("dataset", "")).strip()
        if configured:
            return self.config.resolve_artifact_path(configured, field="[prior].dataset")
        state_value = str(self.state.artifacts.get("dataset", "")).strip()
        if state_value:
            return self.config.require_external_artifact_path(
                Path(state_value), field="pipeline state dataset"
            )
        program = self._configured_program_paths()
        return self._collection_paths(program)[2]

    def _prior_identity(self) -> tuple[str, str, Path, Path]:
        section = self.config.section("prior")
        program_count = max(1, len(self._configured_program_paths()))
        total = int(self.config.section("collect")["successful_demos_per_program"]) * program_count
        run_name = _safe_token(str(section.get("run_name", f"mpc{total}")))
        name_prefix = _safe_token(str(section.get("name_prefix", f"{self.config.prior_task_slug}_{run_name}")))
        artifact_root_value = str(section.get("artifact_root", "")).strip()
        artifact_root = (
            self.config.resolve_artifact_path(artifact_root_value, field="[prior].artifact_root")
            if artifact_root_value
            else self.config.task_artifact_root / "prior_artifacts"
        )
        csv_path = artifact_root / "prior_sweep_csv" / "generated" / f"{name_prefix}.csv"
        return run_name, name_prefix, artifact_root, csv_path

    def _prior_base_command(self, mode: str, *, csv_path: Path | None = None) -> list[str]:
        section = self.config.section("prior")
        run_name, name_prefix, artifact_root, _expected_csv = self._prior_identity()
        command = [
            self.config.python,
            str(project_root() / "src" / "kuafu" / "bc" / "training" / "run_bc_prior_pipeline.py"),
            "--dataset",
            str(self._dataset_path()),
            "--task_name",
            self.config.task_name,
            "--robot_id",
            self.config.robot_id,
            "--mode",
            mode,
            "--task_slug",
            self.config.prior_task_slug,
            "--gpus",
            *[str(gpu) for gpu in self.config.gpus],
            "--workers_per_gpu",
            str(int(section.get("workers_per_gpu", 2))),
            "--stagger_sec",
            str(float(section.get("stagger_sec", 3.0))),
            "--python",
            self.config.python,
            "--artifact_root",
            str(artifact_root),
            "--priors_root",
            str(artifact_root / "published_priors"),
        ]
        if mode != "residual":
            command.append("--no_residual_launcher")
        if mode in {"test", "weights", "residual"}:
            command.extend(["--program_path", str(self._execution_program_path())])
        if csv_path is None:
            command.extend(
                [
                    "--run_name",
                    run_name,
                    "--name_prefix",
                    name_prefix,
                    "--obs_key",
                    str(section.get("obs_key", "state")),
                    "--action_mode",
                    str(section.get("action_mode", "absolute_ee")),
                    "--time_shift",
                    str(int(section.get("time_shift", 0))),
                    "--num_epochs",
                    str(int(section.get("num_epochs", 150))),
                    "--bc_num_workers",
                    str(int(section.get("bc_num_workers", 0))),
                    "--chunk_sizes",
                    *self.config.string_list("prior", "chunk_sizes", ("10", "15", "30")),
                    "--chunk_strides",
                    *self.config.string_list("prior", "chunk_strides", ("1", "3")),
                    "--variants",
                    *self.config.string_list("prior", "variants", ("f", "ff", "fff", "f3", "ff3", "fff3")),
                ]
            )
            if _as_bool(section, "pin_memory", False):
                command.append("--bc_pin_memory")
            if not _as_bool(section, "sample_weighting", True):
                command.append("--no_sample_weighting")
            if not _as_bool(section, "sample_weighting_validation", True):
                command.append("--sample_weighting_no_validation")
            if self.force:
                command.extend(["--overwrite", "--retrain_existing", "--publish_overwrite"])
        else:
            command.extend(["--csv", str(csv_path)])
        return command

    def train_prior(self) -> dict[str, Any]:
        dataset = self._dataset_path()
        _run_name, _name_prefix, _artifact_root, csv_path = self._prior_identity()
        if not self.dry_run and not dataset.is_file():
            raise PipelineError(f"Collected dataset does not exist: {dataset}")
        if not self.dry_run:
            self._validate_dataset(dataset)
        command = self._prior_base_command("train")
        self.runner.run("train-prior", command)
        if not self.dry_run and not csv_path.is_file():
            raise PipelineError(f"BC training completed without the expected sweep CSV: {csv_path}")
        return {"prior_csv": str(csv_path), "dataset": str(dataset)}

    def _prior_csv(self) -> Path:
        copied = self.config.data.get("handoff", {}).get("prior_csv")
        if copied:
            path = Path(copied).resolve()
            if not path.is_relative_to(self.config.task_artifact_root):
                raise ConfigError("The handoff prior CSV must belong to this learning attempt")
            return path
        configured = str(self.config.section("eval").get("csv", "")).strip()
        if configured:
            return self.config.resolve_artifact_path(configured, field="[eval].csv")
        state_value = str(self.state.artifacts.get("prior_csv", "")).strip()
        if state_value:
            return self.config.require_external_artifact_path(
                Path(state_value), field="pipeline state prior CSV"
            )
        return self._prior_identity()[3]

    def _residual_artifact_dir(self) -> Path:
        slug = self.config.prior_task_slug
        suffix = RESIDUAL_TASK_SUFFIXES.get(slug)
        prefix = ROBOT_PREFIXES.get(self.config.robot_id)
        if suffix is None or prefix is None:
            raise ConfigError(
                f"No residual-RL artifact mapping for robot={self.config.robot_id!r}, task_slug={slug!r}"
            )
        return (
            self._prior_identity()[2]
            / "residual_rl"
            / "parameter_search"
            / f"{prefix}_{suffix}"
        )

    def _eval_command(self, mode: str, csv_path: Path, params_path: Path | None = None) -> list[str]:
        section = self.config.section("eval")
        command = self._prior_base_command(mode, csv_path=csv_path)
        command.extend(
            [
                "--duration_sec",
                str(float(section.get("duration_sec", 180.0))),
                "--num_envs",
                str(int(section.get("num_envs", 8192))),
                "--chunk_exec",
                str(int(section.get("chunk_exec", 1))),
            ]
        )
        success_col = str(section.get("success_rate_col", "")).strip()
        if success_col:
            command.extend(["--success_rate_col", success_col])
        conda_env = str(section.get("conda_env", "")).strip()
        if conda_env:
            command.extend(["--conda_env", conda_env])
        if params_path is not None:
            command.extend(["--params_path", str(params_path)])
        if mode == "weights":
            residual = self.config.section("residual")
            command.extend([
                "--weight_selection_policy", str(residual.get("selection_policy", "smallest_class_over_threshold")),
                "--residual_success_threshold", str(float(residual.get("success_threshold", 20.0))),
                "--residual_model_class_order", *self.config.string_list(
                    "residual", "model_class_order", ("bc", "class", "ff3", "ff", "fff3", "fff", "f3", "f")),
            ])
        if _as_bool(section, "skip_parameter_search", False):
            command.append("--no_residual_params")
        if self.force:
            command.extend(["--retest_existing", "--force_residual_params", "--force_residual_weights"])
        return command

    def eval_prior(self) -> dict[str, Any]:
        csv_path = self._prior_csv()
        if not self.dry_run and not csv_path.is_file():
            raise PipelineError(f"Prior sweep CSV does not exist: {csv_path}")
        if not self.dry_run:
            self._validate_dataset(self._dataset_path())
        artifact_dir = self._residual_artifact_dir()
        params_path = artifact_dir / "params_best.json"
        weights_path = artifact_dir / "weights_best.json"
        configured_params = str(self.config.section("eval").get("params_path", "")).strip()
        if configured_params:
            params_path = self.config.resolve_artifact_path(configured_params, field="[eval].params_path")
        self.runner.run("eval-prior", self._eval_command("test", csv_path, params_path if configured_params else None))
        artifacts: dict[str, Any] = {"prior_csv": str(csv_path)}
        if params_path.exists() or self.dry_run:
            artifacts["residual_params"] = str(params_path)

        if _as_bool(self.config.section("eval"), "acquire_weights", True):
            params_arg = params_path if params_path.exists() or self.dry_run else None
            self.runner.run("acquire-weights", self._eval_command("weights", csv_path, params_arg))
            if not self.dry_run and not weights_path.is_file():
                raise PipelineError(f"Weight acquisition completed without {weights_path}")
            artifacts["residual_weights"] = str(weights_path)
        return artifacts

    def _weights_path(self) -> Path:
        configured = str(self.config.section("residual").get("weights_params", "")).strip()
        if configured:
            return self.config.resolve_artifact_path(configured, field="[residual].weights_params")
        state_value = str(self.state.artifacts.get("residual_weights", "")).strip()
        if state_value:
            return self.config.require_external_artifact_path(
                Path(state_value), field="pipeline state residual weights"
            )
        return self._residual_artifact_dir() / "weights_best.json"

    def train_residual(self) -> dict[str, Any]:
        section = self.config.section("residual")
        hparam_iterations = section.get("hparam_iterations", 5)
        if type(hparam_iterations) is not int or hparam_iterations < 0:
            raise ConfigError("[residual].hparam_iterations must be a non-negative integer")
        csv_path = self._prior_csv()
        weights_path = self._weights_path()
        configured_launcher = str(section.get("launcher", "")).strip()
        launcher = (
            self.config.resolve_artifact_path(configured_launcher, field="[residual].launcher")
            if configured_launcher
            else self.config.task_artifact_root / "residual" / "train_residual_hparams.sh"
        )
        if not self.dry_run:
            if not csv_path.is_file():
                raise PipelineError(f"Prior sweep CSV does not exist: {csv_path}")
            if not weights_path.is_file():
                raise PipelineError(f"Residual-RL weights/parameters do not exist: {weights_path}")
            validate_program_metadata(json.loads(weights_path.read_text()), learned_program_identity(self._execution_program_path()))
            self._stage_files("eval-prior", {"prior_csv": str(csv_path)})

        command = self._prior_base_command("residual", csv_path=csv_path)
        command.extend(
            [
                "--residual_hparam_iters",
                str(hparam_iterations),
                "--params_path",
                str(weights_path),
                "--residual_output_dir",
                str(launcher.parent),
                "--residual_output_name",
                launcher.name,
                "--residual_selection_policy",
                str(section.get("selection_policy", "smallest_class_over_threshold")),
                "--residual_success_threshold",
                str(float(section.get("success_threshold", 20.0))),
                "--residual_model_class_order",
                *self.config.string_list(
                    "residual",
                    "model_class_order",
                    ("bc", "class", "ff3", "ff", "fff3", "fff", "f3", "f"),
                ),
            ]
        )
        if self.force or (launcher.is_file() and self.state.artifacts.get("residual_launcher") == str(launcher)):
            command.append("--residual_overwrite")
        self.runner.run("prepare-residual", command)
        if not self.dry_run and not launcher.is_file():
            raise PipelineError(f"Residual launcher was not generated: {launcher}")

        execute = _as_bool(section, "execute", True) and not self.prepare_only
        artifacts = {"residual_launcher": str(launcher), "residual_weights": str(weights_path), "residual_executed": execute}
        if execute:
            raw_env = section.get("env", {})
            if not isinstance(raw_env, dict):
                raise ConfigError("[residual.env] must be a TOML table")
            extra_env = {str(key): str(value) for key, value in raw_env.items()}
            selected_program = str(self._execution_program_path())
            if extra_env.get("KUAFU_PROGRAM_PATH", selected_program) != selected_program:
                raise PipelineError("Residual env cannot override the learned program selected for evaluation")
            completion = launcher.parent / f"training_completion_{time.time_ns()}.json"
            extra_env["KUAFU_PROGRAM_PATH"] = selected_program
            extra_env["KUAFU_RESIDUAL_COMPLETION_PATH"] = str(completion)
            extra_env["KUAFU_RESIDUAL_INPUT_IDENTITY"] = self._stage_inputs("train-residual") if not self.dry_run else "dry-run"
            if "RESIDUAL_CUDA_VISIBLE_DEVICES" in extra_env:
                extra_env["KUAFU_CUDA_VISIBLE_DEVICES"] = extra_env.pop("RESIDUAL_CUDA_VISIBLE_DEVICES")
            if "RESIDUAL_DEVICE" in extra_env:
                extra_env["KUAFU_DEVICE"] = extra_env.pop("RESIDUAL_DEVICE")
            configured_python = Path(self.config.python).expanduser()
            old_path = os.environ.get("PATH", "")
            extra_env["PATH"] = f"{configured_python.parent}{os.pathsep}{old_path}"
            self.runner.run("train-residual", ["bash", str(launcher)], extra_env=extra_env)
            if not self.dry_run:
                if not completion.is_file():
                    raise PipelineError("Residual command exited without a training completion receipt (a dry-run is not training)")
                receipt = json.loads(completion.read_text())
                if receipt.get("input_identity") != extra_env["KUAFU_RESIDUAL_INPUT_IDENTITY"]:
                    raise PipelineError("Residual completion receipt does not match this run")
                checkpoint = Path(receipt["checkpoint"])
                if _sha256(checkpoint) != receipt.get("checkpoint_sha256"):
                    raise PipelineError("Residual completion checkpoint is missing or changed")
                artifacts.update(residual_checkpoint=str(checkpoint), residual_completion=str(completion))
        else:
            print(f"[train-residual] launcher prepared but not executed: {launcher}", flush=True)
        return artifacts


def stage_slice(start: str | None, through: str | None) -> tuple[str, ...]:
    start_index = 0 if start is None else STAGES.index(start)
    end_index = len(STAGES) if through is None else STAGES.index(through) + 1
    if start_index >= end_index:
        raise PipelineError("--from-stage must not come after --through-stage")
    return STAGES[start_index:end_index]
