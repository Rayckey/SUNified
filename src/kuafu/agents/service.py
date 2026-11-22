"""Kuafu operations that an external agent can discover, call, inspect and revise.

This module owns no model client. The caller supplies scene and program data;
Kuafu supplies geometry, receiver-owned tolerances, execution and measured feedback.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import sys
import threading
import traceback
import uuid
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from kuafu.paths import ASSET_ROOT, PROMPT_ROOT, RESOURCE_ROOT, external_artifact_root, require_external_artifact_path
from kuafu.pipeline.handoff import HandoffRun, TOOL_STEPS

from . import jobs
from .schemas import PROGRAM, definitions


class ToolService:
    """A local tool dispatcher; reuse it directly or through the CLI/MCP adapters.

    ``isaac_python`` and ``timeout_s`` are host configuration, not agent-supplied
    tool arguments. Long jobs persist independently of this service instance.
    """

    def __init__(self, artifact_root=None, *, isaac_python=None, device=None, timeout_s=None, run_id=None):
        self.artifact_root = require_external_artifact_path(artifact_root or external_artifact_root())
        self.root = self.artifact_root / "agents"
        self.isaac_python = (
            isaac_python or os.getenv("KUAFU_ISAAC_PYTHON") or os.getenv("KUAFU_PYTHON") or sys.executable
        )
        self.device = device or os.getenv("KUAFU_AGENT_DEVICE", "cuda:0")
        self.timeout_s = int(os.getenv("KUAFU_AGENT_JOB_TIMEOUT", "900") if timeout_s is None else timeout_s)
        if not 1 <= self.timeout_s <= 86400:
            raise ValueError("Job timeout must be 1-86400 seconds")
        if not re.fullmatch(r"cpu|cuda:\d+", self.device):
            raise ValueError("Device must be cpu or cuda:N")
        self.run_id = run_id or os.getenv("KUAFU_RUN_ID") or None
        self._run = None
        self._attempt = None
        self._lock = threading.RLock()

    def list_tools(self):
        return definitions()

    def call(self, name: str, arguments: dict | None = None) -> dict:
        """Return one JSON envelope; invalid proposals become actionable feedback."""
        envelope = {"protocol_version": 1, "tool": name}
        try:
            definition = next((tool for tool in definitions() if tool["name"] == name), None)
            if definition is None:
                raise ValueError(f"Unknown tool: {name}")
            arguments = {} if arguments is None else deepcopy(arguments)
            # JSON rejects NaN/Infinity even though Python's JSON parser permits them.
            json.dumps(arguments, allow_nan=False)
            error = best_match(Draft202012Validator(definition["inputSchema"]).iter_errors(arguments))
            if error is not None:
                raise ValueError(f"{'/'.join(map(str, error.absolute_path)) or '/'}: {error.message}")
            with self._lock, ExitStack() as stack:
                previous = self._run, self._attempt
                try:
                    run_id = arguments.pop("run_id", self.run_id) if name != "kuafu_create_run" else None
                    self._run = HandoffRun(self.artifact_root, run_id) if run_id else None
                    self._attempt = None
                    if self._run:
                        envelope["run_id"] = self._run.run_id
                        if name in TOOL_STEPS or name == "kuafu_start_stage":
                            self._attempt = self._run.begin(name, arguments)
                            envelope.update(attempt_id=self._attempt.name, directory=str(self._attempt))
                            log = stack.enter_context((self._attempt / "tool.log").open("a"))
                            stack.enter_context(redirect_stdout(log))
                            stack.enter_context(redirect_stderr(log))
                    if not self._attempt:
                        stack.enter_context(redirect_stdout(sys.stderr))
                    try:
                        result = getattr(self, "_" + name)(**arguments)
                        response = {**envelope, "ok": True, "result": result}
                    except Exception as exc:
                        if self._attempt:
                            traceback.print_exc()
                        response = {**envelope, "ok": False,
                                    "error": {"type": type(exc).__name__, "message": str(exc)[:8000]}}
                    if self._attempt:
                        self._run.finish(self._attempt, response)
                    return response
                finally:
                    self._run, self._attempt = previous
        except (ValueError, TypeError, OSError, RuntimeError, ImportError, KeyError) as exc:
            return {**envelope, "ok": False, "error": {"type": type(exc).__name__, "message": str(exc)[:8000]}}

    def _new(self, category):
        if self._run and self._attempt:
            return self._run.allocate(self._attempt, category)
        path = self.root / category / uuid.uuid4().hex
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _lookup(self, category, identifier):
        if not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise ValueError("Use the ID returned by the tool, not a filesystem path")
        if self._run:
            return self._run.lookup(category, identifier)
        path = (self.root / category / identifier).resolve()
        if not path.is_relative_to(self.root.resolve()) or not path.is_dir():
            raise ValueError(f"Unknown {category} ID: {identifier}")
        return path

    def _kuafu_create_run(self, task, description="", videos=None, run_id=None):
        run = HandoffRun.create(self.artifact_root, task=task, description=description,
                                videos=videos or [], run_id=run_id)
        return run.status()

    def _kuafu_list_tasks(self, robot_id=None):
        from kuafu.pipeline.task_catalog import list_tasks

        return {"tasks": list_tasks(robot_id)}

    def _kuafu_get_task(self, task_id, robot_id=None):
        from kuafu.pipeline.task_catalog import get_task

        return get_task(task_id, robot_id)

    def _kuafu_run_status(self, verify_outputs=False):
        if self._run is None:
            raise ValueError("Supply run_id or set KUAFU_RUN_ID")
        return self._run.status(verify_outputs=verify_outputs)

    def _kuafu_retry_step(self, attempt_id):
        if self._run is None:
            raise ValueError("Supply run_id or set KUAFU_RUN_ID")
        request = self._run.retry_request(attempt_id)
        response = self.call(request["tool"], {**request["arguments"], "run_id": self._run.run_id})
        if not response["ok"]:
            raise ValueError(f"Retry failed; inspect attempt {response.get('attempt_id')}: {response['error']['message']}")
        return {"attempt_id": response["attempt_id"], "directory": response["directory"], **response["result"]}

    def _python(self):
        path = shutil.which(self.isaac_python)
        if path is None:
            raise ValueError("Set KUAFU_ISAAC_PYTHON to an executable from your Isaac environment")
        return path

    def _start(self, directory, request):
        if self._run and self._attempt:
            self._run.attach_job(self._attempt, directory)
        jobs.launch(
            directory, {"python": self._python(), "device": self.device, "timeout_s": self.timeout_s, **request}
        )
        return {
            "job_id": directory.name,
            "status": "queued",
            "directory": str(directory),
            "next_tool": "kuafu_job_status",
        }

    def _kuafu_contract(self, kind):
        if kind == "asset":
            from kuafu.asset_generation.schema import SceneSpec

            return {
                "schema": SceneSpec.model_json_schema(),
                "instructions": "Inspect the sampled frames. Propose data in meters with joint topology, "
                "limits, interaction frames and language semantics. Build, inspect previews and revise. "
                "Dimensions inferred from images are estimates; a mesh build is not physical verification.",
            }
        return {
            "schema": deepcopy(PROGRAM),
            "instructions": (PROMPT_ROOT / "Manipulation/constraint_prompt.txt").read_text(),
            "loop": [
                "kuafu_inspect_drawer",
                "kuafu_job_status",
                "kuafu_submit_program",
                "kuafu_verify_program",
                "kuafu_job_status",
                "revise proposal if any stage, solver or physical check failed",
            ],
            "scope": "Program schema is shared; the provided scene-inspection/verification runner "
            "checks opening the top drawer of a three-drawer cabinet while the lower drawers stay still.",
        }

    def _kuafu_prepare_video(self, video, description, reference_videos=None, frames=12, motion="auto"):
        from kuafu.asset_generation.video import prepare

        output = self._new("inputs")
        manifest = prepare(
            Path(video),
            description,
            output,
            frame_limit=frames,
            reference_videos=[Path(p) for p in reference_videos or []],
            motion=motion,
        )
        return {
            "directory": str(output),
            "input": manifest.model_dump(mode="json"),
            "contact_sheet": str(output / "frames.jpg"),
            "model_requests": 0,
        }

    def _kuafu_build_asset(self, scene, mass_kg=0.2, estimated_mass=True):
        from kuafu.asset_generation.export import export_asset
        from kuafu.asset_generation.geometry import build_asset
        from kuafu.asset_generation.schema import PhysicsConfig, SceneSpec
        from kuafu.asset_generation.semantics import require_agent_semantics

        spec = SceneSpec.model_validate(scene)
        require_agent_semantics(spec)
        asset = build_asset(spec)
        output = self._new("assets")
        export_asset(asset, spec, output, PhysicsConfig(mass_kg=mass_kg, mass_is_estimate=estimated_mass))
        return {
            "asset": str(output),
            "semantics": jobs.read_json(output / "semantics.json"),
            "files": [str(p) for p in sorted(output.iterdir()) if p.is_file()],
            "physics_verified": False,
            "model_requests": 0,
        }

    def _kuafu_inspect_drawer(self, asset, task, target_travel_m, position=None, yaw_deg=180.0):
        asset = Path(asset).expanduser().resolve()
        context = {
            "asset": str(asset),
            "asset_hashes": jobs.asset_hashes(asset),
            "task": task,
            "target_travel_m": target_travel_m,
            "position": position or [0.6, -0.5, 0.2],
            "yaw_deg": yaw_deg,
        }
        job = self._start(self._new("jobs"), {"kind": "inspect", "context": context})
        return {
            **job,
            "physical_objective": {
                "joint": "upper_drawer_joint",
                "opening_travel_m": target_travel_m,
                "other_drawers": "must stay within 5 mm of their initial positions",
            },
        }

    def _kuafu_submit_program(self, context_job, program):
        from kuafu.skills.constraints.tolerance_policy import receive_constraint_geometry
        from kuafu.programs.compiler.constraint_validation import (
            find_directional_rotate_axis_align_conflicts,
            find_manipulator_effort_condition_conflicts,
        )

        directory = self._lookup("jobs", context_job)
        request = jobs.read_json(directory / "request.json")
        state = jobs.status(directory)
        if request["kind"] != "inspect" or state["status"] != "succeeded":
            raise ValueError("Wait for a successful kuafu_inspect_drawer job before submitting a program")
        context = request["context"]
        if jobs.asset_hashes(context["asset"]) != context["asset_hashes"]:
            raise ValueError("Asset changed since inspection; inspect this new revision first")
        scene = jobs.read_json(directory / "output/scene.json")
        names = set(scene["initial"]["frames"])
        robot_names = {name for name in names if name.startswith("panda_")}
        normalized = []
        for index, stage in enumerate(program["stages"]):
            terms = stage["constraints"]["Constraints"]
            conflicts = find_directional_rotate_axis_align_conflicts(
                terms
            ) + find_manipulator_effort_condition_conflicts(terms)
            ops = [term["op"] for term in terms]
            if sum(op in {"reach_point", "directional_move", "freeze_position"} for op in ops) > 1:
                conflicts.append("Use at most one position operator per stage")
            if len(set(ops) & {"open_gripper", "close_gripper", "object_grasped"}) > 1:
                conflicts.append("Use exactly one intended gripper state per stage")
            if conflicts:
                raise ValueError(f"Stage {index}: {'; '.join(conflicts)}")
            for term in terms:
                args = term["args"]
                for key in ("controlled_target", "reference_origin"):
                    if key in args and args[key] not in names | ({"world"} if key == "reference_origin" else set()):
                        raise ValueError(f"Stage {index}: unknown {key} {args[key]!r}; use an observed frame")
                if "controlled_target" in args:
                    is_robot = args["controlled_target"] in robot_names
                    if is_robot != (args["controlled_entity"] == "robot_frame"):
                        raise ValueError(f"Stage {index}: controlled_entity does not match the target frame")
                if "reference_origin" in args:
                    origin = args["reference_origin"]
                    expected = "world" if origin == "world" else "robot_frame" if origin in robot_names else "object"
                    if args["reference_basis"] != expected:
                        raise ValueError(f"Stage {index}: reference_basis does not match reference_origin")
            constraints = receive_constraint_geometry(
                stage["constraints"],
                atomic_action=" ".join(stage["atomic_actions"]),
                detectable_objects=scene["descriptions"],
                profile="mpc",
            )
            normalized.append({**stage, "constraints": constraints})
        output = self._new("programs")
        for index, stage in enumerate(normalized):
            folder = output / "program" / f"subtask_{index}"
            folder.mkdir(parents=True)
            (folder / "task.txt").write_text(stage["task"] + "\n")
            (folder / "atomic.txt").write_text(repr(stage["atomic_actions"]) + "\n")
            (folder / "status.txt").write_text("status: unverified\n")
            jobs.write_json(folder / "constraints.json", stage["constraints"])
        (output / "success.txt").write_text("False\n")
        jobs.write_json(output / "proposal.json", program)
        jobs.write_json(output / "context.json", context)
        return {
            "program_id": output.name,
            "directory": str(output),
            "status": "unverified",
            "stages": len(normalized),
            "next_tool": "kuafu_verify_program",
        }

    def _kuafu_verify_program(self, program_id, max_steps=5000, stage_steps=700):
        program = self._lookup("programs", program_id)
        context = jobs.read_json(program / "context.json")
        if jobs.asset_hashes(context["asset"]) != context["asset_hashes"]:
            raise ValueError("Asset changed since the program was grounded; inspect and submit a new revision")
        directory = self._new("jobs")
        shutil.copytree(program / "program", directory / "input/program")
        return self._start(
            directory,
            {
                "kind": "verify",
                "context": context,
                "program_hashes": jobs.tree_hashes(directory / "input/program"),
                "max_steps": max_steps,
                "stage_steps": stage_steps,
            },
        )

    def _drawer_learning_source(self, job_id, *, kind, step):
        source = self._lookup("jobs", job_id)
        status = jobs.status(source)
        request = jobs.read_json(source / "request.json")
        if status["status"] != "succeeded" or request["kind"] != kind:
            raise ValueError(f"Use a completed {kind} job")
        if self._run:
            latest = self._run.status(verify_outputs=True)["steps"].get(step, {})
            if latest.get("job_id") != job_id or latest.get("stale"):
                raise ValueError(f"Use the current, unchanged {step} result in this run")
        if kind == "verify":
            from kuafu.artifact_contracts import learned_program_identity

            accepted = source / "accepted"
            provenance = jobs.read_json(accepted / "provenance.json")
            if (learned_program_identity(accepted) != status["accepted_identity"]
                    or provenance["context"] != request["context"]
                    or jobs.digest(source / "output/verification.json") != provenance["verification_sha256"]):
                raise ValueError("Accepted program or verification changed")
            program = accepted / "program"
        else:
            from kuafu.pipeline.drawer_policy import finish

            status.update(finish(source, request))
            program = source / "input/program"
        context = deepcopy(request["context"])
        if jobs.asset_hashes(context["asset"]) != context["asset_hashes"]:
            raise ValueError("Asset changed since verification")
        directory = self._new("jobs")
        shutil.copytree(program, directory / "input/program")
        shutil.copytree(context["asset"], directory / "input/asset")
        context["asset"] = str(directory / "input/asset")
        return directory, context, status

    def _start_drawer_learning(self, directory, context, kind, **parameters):
        return self._start(directory, {"kind": kind, "context": context, **parameters,
                                       "input_hashes": jobs.tree_hashes(directory / "input")})

    def _kuafu_collect_drawer(self, verification_job, episodes=4, seed=42):
        directory, context, _ = self._drawer_learning_source(
            verification_job, kind="verify", step="verify-program")
        return self._start_drawer_learning(directory, context, "drawer-collect", episodes=episodes, seed=seed)

    def _kuafu_train_drawer_policy(self, collection_job, epochs=300, seed=42, method="nearest_neighbor"):
        directory, context, source = self._drawer_learning_source(
            collection_job, kind="drawer-collect", step="collect")
        shutil.copyfile(source["artifacts"]["dataset"], directory / "input/demonstrations.hdf5")
        return self._start_drawer_learning(directory, context, "drawer-train", epochs=epochs, seed=seed, method=method)

    def _kuafu_evaluate_drawer_policy(self, training_job, seed=46):
        directory, context, source = self._drawer_learning_source(
            training_job, kind="drawer-train", step="train-prior")
        shutil.copyfile(source["artifacts"]["checkpoint"], directory / "input/policy.pth")
        return self._start_drawer_learning(directory, context, "drawer-evaluate", seed=seed)

    def _kuafu_job_status(self, job_id):
        directory = self._lookup("jobs", job_id)
        return {
            "job_id": job_id,
            **jobs.status(directory),
            "directory": str(directory),
            "log": str(directory / "run.log"),
            "worker_log": str(directory / "worker.log"),
        }

    def _kuafu_cancel_job(self, job_id):
        directory = self._lookup("jobs", job_id)
        if jobs.status(directory)["status"] in {"queued", "running"}:
            (directory / "cancel.request").touch()
            return {"job_id": job_id, "status": "cancellation_requested"}
        return self._kuafu_job_status(job_id)

    def _kuafu_read_artifact(self, path):
        path = Path(path).expanduser().resolve()
        if not any(path.is_relative_to(root.resolve()) for root in (self.artifact_root, RESOURCE_ROOT, ASSET_ROOT)):
            raise ValueError("Read only files within the configured artifact root or repository resources")
        if path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("Artifact exceeds 4 MiB; use the job's summary or inspect the file locally")
        result = {"path": str(path), "sha256": jobs.digest(path)}
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            return {
                **result,
                "mime_type": "image/png" if path.suffix.lower() == ".png" else "image/jpeg",
                "image_base64": base64.b64encode(path.read_bytes()).decode(),
            }
        if path.suffix.lower() not in {".json", ".txt", ".md", ".log", ".toml", ".csv"}:
            raise ValueError("Supported artifact types: JSON, text, logs, TOML, CSV, PNG and JPEG")
        return {**result, "data": jobs.read_json(path) if path.suffix == ".json" else path.read_text()}

    def _kuafu_pipeline_status(self, config):
        from kuafu.pipeline.config import PipelineConfig
        from kuafu.pipeline.pipeline import Pipeline

        pipeline = Pipeline(PipelineConfig.load(config), dry_run=True)
        errors, warnings = pipeline.doctor()
        return {
            "state": pipeline.state.data,
            "state_path": str(pipeline.state.path),
            "errors": errors,
            "warnings": warnings,
        }

    def _kuafu_start_stage(self, config, stage, prepare_only=False):
        from kuafu.pipeline.config import PipelineConfig

        cfg = PipelineConfig.load(config)
        if cfg.section("learn_programs").get("enabled") is not False:
            raise ValueError("External agents supply verified program: set [learn_programs].enabled = false")
        if stage == "eval-prior" and cfg.section("eval").get("acquire_weights", True) is not False:
            raise ValueError("Set [eval].acquire_weights = false and supply your agent's weights separately")
        if stage == "eval-prior" and cfg.section("eval").get("skip_parameter_search", False) is not True:
            raise ValueError("Set [eval].skip_parameter_search = true to keep model calls in your agent")
        if stage == "train-residual" and cfg.section("residual").get("hparam_iterations", 5) != 0:
            raise ValueError("Set [residual].hparam_iterations = 0 to train with supplied hyperparameters")
        directory = self._new("jobs")
        content = cfg.path.read_text()
        if self._run:
            import tomllib
            for section, fields in {"collect": ("dataset",), "prior": ("artifact_root",),
                                    "residual": ("launcher",)}.items():
                for field in fields:
                    if cfg.section(section).get(field):
                        raise ValueError(f"For a handoff run, leave [{section}].{field} empty to use run-local outputs")
            previous = cfg.data.get("handoff", {})
            if previous:
                if Path(previous.get("run_root", "")).resolve() != self._run.root.resolve():
                    raise ValueError("The saved manifest belongs to another handoff run")
                content = re.sub(r"(?ms)^\[handoff\][ \t]*(?:#[^\n]*)?\n.*?(?=^\[|\Z)", "", content)
                if "handoff" in tomllib.loads(content):
                    raise ValueError("Use a standalone [handoff] table for the saved run settings")
            handoff = self._run.prepare_learning(self._attempt, cfg)
            content += "\n[handoff]\n" + "".join(f"{key} = {json.dumps(value)}\n" for key, value in handoff.items())
            tomllib.loads(content)
        (directory / "pipeline.toml").write_text(content)
        return self._start(directory, {"kind": "stage", "stage": stage, "prepare_only": prepare_only})
