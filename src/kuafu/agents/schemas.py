"""Provider-neutral tools built on the shared SUN skill grammar."""

from copy import deepcopy
from kuafu.skills.schema import PROGRAM, TEXT, NUMBER, VECTOR, obj


def definitions():
    """Return standard tool descriptions and JSON Schemas, independent of MCP."""
    tools = [
        (
            "kuafu_list_tasks",
            "List published task examples, available robots and interface status.",
            obj({"robot_id": TEXT}, []),
            True,
        ),
        (
            "kuafu_get_task",
            "Read a task's instructions, environment IDs, object assets, success checks and saved SUN Programs.",
            obj({"task_id": TEXT, "robot_id": TEXT}, ["task_id"]),
            True,
        ),
        (
            "kuafu_create_run",
            "Start a task run with immutable video inputs and a directory for every module handoff.",
            obj({"task": TEXT, "description": TEXT,
                 "videos": {"type": "array", "items": TEXT, "maxItems": 4},
                 "run_id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"}}, ["task"]),
            False,
        ),
        (
            "kuafu_run_status",
            "Inspect all step attempts, input/output files, jobs and recovery information for a run.",
            obj({"verify_outputs": {"type": "boolean", "description": "Check saved artifact hashes before resuming."}}, []),
            True,
        ),
        (
            "kuafu_retry_step",
            "Repeat a finished or interrupted step from its saved inputs, keeping every earlier attempt.",
            obj({"attempt_id": TEXT}),
            False,
        ),
        (
            "kuafu_contract",
            "Get asset or program proposal schemas and instructions before authoring a candidate.",
            obj({"kind": {"enum": ["asset", "program"]}}),
            True,
        ),
        (
            "kuafu_prepare_video",
            "Sample local videos and an object description for YOUR agent; no model call.",
            obj(
                {
                    "video": TEXT,
                    "description": TEXT,
                    "reference_videos": {"type": "array", "items": TEXT, "maxItems": 3},
                    "frames": {"type": "integer", "minimum": 1, "maximum": 32},
                    "motion": {"enum": ["auto", "rigid", "articulated"]},
                },
                ["video", "description"],
            ),
            False,
        ),
        (
            "kuafu_build_asset",
            "Build YOUR scene proposal into USD, previews and interaction semantics. This is not a physics pass.",
            obj(
                {
                    "scene": {"type": "object"},
                    "mass_kg": {"type": "number", "exclusiveMinimum": 0, "maximum": 1000},
                    "estimated_mass": {"type": "boolean"},
                },
                ["scene"],
            ),
            False,
        ),
        (
            "kuafu_inspect_drawer",
            "Observe a three-drawer cabinet in Isaac for TOP-DRAWER OPENING. "
            "The physical target is upper_drawer_joint travel; lower drawers must stay still.",
            obj(
                {
                    "asset": TEXT,
                    "task": {**TEXT, "description": "Language phrasing of the top-drawer opening objective."},
                    "target_travel_m": {"type": "number", "minimum": 0.02, "maximum": 0.5},
                    "position": VECTOR,
                    "yaw_deg": NUMBER,
                },
                ["asset", "task", "target_travel_m"],
            ),
            False,
        ),
        (
            "kuafu_submit_program",
            "Validate YOUR stage proposal against an inspected scene; create a new unverified revision.",
            obj({"context_job": TEXT, "program": PROGRAM}),
            False,
        ),
        (
            "kuafu_verify_program",
            "Run YOUR candidate with MPC. Independent physical success is required before promotion.",
            obj(
                {
                    "program_id": TEXT,
                    "max_steps": {"type": "integer", "minimum": 1, "maximum": 5000},
                    "stage_steps": {"type": "integer", "minimum": 1, "maximum": 1000},
                },
                ["program_id"],
            ),
            False,
        ),
        (
            "kuafu_collect_drawer",
            "Collect physically verified MPC episodes from an accepted generated-cabinet program.",
            obj({"verification_job": TEXT, "episodes": {"type": "integer", "minimum": 2, "maximum": 32},
                 "seed": {"type": "integer", "minimum": 0}}, ["verification_job"]),
            False,
        ),
        (
            "kuafu_train_drawer_policy",
            "Train a BC joint policy on collected cabinet episodes; retain the asset and SUN Program identity.",
            obj({"collection_job": TEXT, "method": {"enum": ["nearest_neighbor", "mlp"]},
                 "epochs": {"type": "integer", "minimum": 1, "maximum": 10000, "description": "Epochs per stage for method=mlp."},
                 "seed": {"type": "integer", "minimum": 0}}, ["collection_job"]),
            False,
        ),
        (
            "kuafu_evaluate_drawer_policy",
            "Measure the trained cabinet policy in Isaac with SUN predicates and independent physical checks.",
            obj({"training_job": TEXT, "seed": {"type": "integer", "minimum": 0}}, ["training_job"]),
            False,
        ),
        (
            "kuafu_job_status",
            "Poll an Isaac/training job; get observations, failed predicates, physical results and log paths.",
            obj({"job_id": TEXT}),
            True,
        ),
        (
            "kuafu_cancel_job",
            "Request cancellation of this job's owned processes; retain diagnostic artifacts.",
            obj({"job_id": TEXT}),
            False,
        ),
        (
            "kuafu_read_artifact",
            "Read an output JSON/text file or view an image within the artifact/resource directories.",
            obj({"path": TEXT}),
            True,
        ),
        (
            "kuafu_pipeline_status",
            "Inspect an existing task manifest, stage state and runtime diagnostics without running it.",
            obj({"config": TEXT}),
            True,
        ),
        (
            "kuafu_start_stage",
            "Run one configured collection/training/evaluation stage as a bounded job. No built-in model generation.",
            obj(
                {
                    "config": TEXT,
                    "stage": {"enum": ["collect", "train-prior", "eval-prior", "train-residual"]},
                    "prepare_only": {"type": "boolean"},
                },
                ["config", "stage"],
            ),
            False,
        ),
    ]
    definitions = []
    for name, description, schema, read_only in tools:
        schema = deepcopy(schema)
        if name != "kuafu_create_run":
            schema["properties"]["run_id"] = {
                "type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$",
                "description": "Run returned by kuafu_create_run; share it across tools and reconnects.",
            }
        definitions.append({"name": name, "description": description,
                            "inputSchema": schema, "readOnly": read_only})
    return definitions
