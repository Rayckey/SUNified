#!/usr/bin/env python3
# Create a prior-conditioned residual launcher for residual-RL hyperparameter search.

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

from kuafu.paths import resolve_artifact_path


REPO_ROOT = Path(__file__).resolve().parents[5]
DUAL_ROOT = Path(__file__).resolve().parent

ROBOT_SUCCESS_COLS = {
    "fr3_finray": "success_rate",
    "franka_panda": "success_rate_franka_panda",
    "kinova_gen3": "success_rate_kinova_gen3",
}

TASK_SLUG_ALIASES = {
    "right_can": "can",
    "can": "can",
    "drawer": "drawer",
    "drawer_stack": "drawer_stack",
    "estop_open": "estop_open",
    "estop_close": "estop_close",
    "hang_cup": "hang_cup",
    "pen_cup": "pen_cup",
    "stack2": "stack2",
    "stack_2": "stack2",
    "stack3": "stack3",
    "stack_3": "stack3",
}

DUAL_TASK_SUFFIXES = {
    "can": "learned_can",
    "drawer": "learned_drawer",
    "drawer_stack": "learned_drawer_stack",
    "estop_open": "learned_estop_open_state",
    "estop_close": "learned_estop_close_state",
    "hang_cup": "learned_hang_cup",
    "pen_cup": "learned_pen_cup_state",
    "stack2": "learned_stack2_state",
    "stack3": "learned_stack3_state",
}

ROBOT_TASK_PREFIXES = {
    "fr3_finray": "fr3_finray",
    "franka_panda": "panda",
    "kinova_gen3": "kinova",
}


@dataclass(frozen=True)
class BestPrior:
    row: dict[str, str]
    success_rate: float
    prior_checkpoint: str
    prior_config: str
    name: str
    variant: str
    model_class: str


def _recipe_or_artifact_path(value: str, label: str) -> Path:
    """Keep immutable source recipes local and resolve generated inputs externally."""

    candidate = Path(value).expanduser()
    allowed_source_roots = {"assets", "resources", "src"}
    if not candidate.is_absolute():
        if candidate.parts and candidate.parts[0] in allowed_source_roots:
            candidate = (REPO_ROOT / candidate).resolve()
        else:
            return resolve_artifact_path(candidate, label)
    else:
        candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(REPO_ROOT)
    except ValueError:
        return resolve_artifact_path(candidate, label)
    if relative.parts and relative.parts[0] in allowed_source_roots:
        return candidate
    return resolve_artifact_path(candidate, label)


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-").lower() or "run"


def _shell_default(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")


def _shell_dq(value: str) -> str:
    return '"' + _shell_default(value) + '"'


def _infer_task_slug(task_name: str) -> str:
    text = task_name.strip()
    for prefix in (
        "FR3-FinRay-Table-",
        "Panda-Table-",
        "Kinova-Table-Gen3-",
        "Kinova-Table-",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    text = re.sub(r"-(f|ff|fff|f3|ff3|fff3)$", "", text, flags=re.IGNORECASE)
    slug = _safe_token(text.replace("-", "_"))
    return TASK_SLUG_ALIASES.get(slug, slug)


BC_TASK_VARIANTS = {"f", "ff", "fff", "f3", "ff3", "fff3"}
BC_TASK_VARIANT_PATTERN = r"f{1,3}3?"
DEFAULT_MODEL_CLASS_ORDER = ("bc", "class", "ff3", "ff", "fff3", "fff", "f3", "f")
SELECTION_POLICY_ALIASES = {
    "best": "best",
    "smallest_class_over_threshold": "smallest_class_over_threshold",
    "smallest_eligible_model_class": "smallest_class_over_threshold",
}
SELECTION_POLICIES = tuple(SELECTION_POLICY_ALIASES)
VARIANT_RE = re.compile(r"(?:^|_)(fff3|ff3|f3|fff|ff|f|class|bc)(?:_stride\d+)?$", re.IGNORECASE)


def _task_with_variant(task_name: str, variant: str) -> str:
    base = re.sub(rf"-({BC_TASK_VARIANT_PATTERN})$", "", task_name.strip(), flags=re.IGNORECASE)
    variant = variant.strip().lower()
    if variant in BC_TASK_VARIANTS:
        return f"{base}-{variant}"
    return task_name.strip()


def _float_or_none(value: str | None) -> float | None:
    text = (value or "").strip().rstrip("%")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _int_or_zero(value: str | None) -> int:
    try:
        return int((value or "").strip())
    except ValueError:
        return 0


def _variant_from_row(row: dict[str, str]) -> str:
    explicit = (row.get("variant") or "").strip().lower()
    if explicit:
        return explicit
    for key in ("name", "config", "prior_path", "best_checkpoint"):
        value = (row.get(key) or "").strip()
        if not value:
            continue
        token = Path(value).stem if key != "name" else value
        match = VARIANT_RE.search(token)
        if match:
            return match.group(1).lower()
    return "unknown"


def _model_class_from_row(row: dict[str, str], model_class_col: str) -> str:
    explicit = (row.get(model_class_col) or "").strip().lower()
    if explicit:
        return explicit
    return _variant_from_row(row)


def _model_class_order(raw_order: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if raw_order is None:
        order_items: list[str] | tuple[str, ...] = DEFAULT_MODEL_CLASS_ORDER
    elif isinstance(raw_order, str):
        order_items = [raw_order]
    else:
        order_items = raw_order

    normalized: list[str] = []
    for item in order_items:
        for token in item.split(","):
            token = token.strip().lower()
            if token and token not in normalized:
                normalized.append(token)
    return tuple(normalized)


def _normalize_selection_policy(value: str) -> str:
    policy = SELECTION_POLICY_ALIASES.get(value.strip())
    if policy is None:
        raise ValueError(f"Unknown selection_policy={value!r}")
    return policy


def _success_col(robot_id: str, explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    return ROBOT_SUCCESS_COLS.get(robot_id, "success_rate")


def _candidate_sort_key(item: tuple[float, dict[str, str], str, str]) -> tuple[float, int, int, str]:
    rate, row, _variant, _model_class = item
    return (
        rate,
        -_int_or_zero(row.get("chunk_size")),
        -_int_or_zero(row.get("action_chunk_stride")),
        row.get("name") or "",
    )


def _select_best_prior(
    csv_path: Path,
    success_col: str,
    *,
    selection_policy: str = "smallest_class_over_threshold",
    success_threshold: float = 20.0,
    model_class_order: str | list[str] | tuple[str, ...] | None = None,
    model_class_col: str = "variant",
) -> BestPrior:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV has no rows: {csv_path}")

    candidates: list[tuple[float, dict[str, str], str, str]] = []
    for row in rows:
        rate = _float_or_none(row.get(success_col))
        if rate is None:
            continue
        variant = _variant_from_row(row)
        model_class = _model_class_from_row(row, model_class_col)
        if model_class == "unknown" and variant != "unknown":
            model_class = variant
        candidates.append((float(rate), row, variant, model_class))
    if not candidates:
        raise ValueError(f"No numeric {success_col!r} values found in {csv_path}")

    policy = _normalize_selection_policy(selection_policy)
    if policy == "best":
        best_rate, best_row, best_variant, best_model_class = max(candidates, key=_candidate_sort_key)
    elif policy == "smallest_class_over_threshold":
        ordered_classes = _model_class_order(model_class_order)
        if not ordered_classes:
            raise ValueError("--model_class_order must contain at least one class")

        eligible_classes = {
            model_class for rate, _row, _variant, model_class in candidates if rate >= success_threshold
        }
        if not eligible_classes:
            raise ValueError(
                f"No model class has {success_col} >= {success_threshold:g} in {csv_path}"
            )

        selected_class = next((model_class for model_class in ordered_classes if model_class in eligible_classes), None)
        if selected_class is None:
            selected_class = sorted(eligible_classes)[0]

        class_candidates = [
            item for item in candidates if item[3] == selected_class and item[0] >= success_threshold
        ]
        best_rate, best_row, best_variant, best_model_class = max(class_candidates, key=_candidate_sort_key)
    else:
        raise ValueError(f"Unknown selection_policy={selection_policy!r}")

    prior_checkpoint = (best_row.get("prior_path") or "").strip()
    if not prior_checkpoint:
        prior_checkpoint = (best_row.get("best_checkpoint") or "").strip()
    if not prior_checkpoint:
        raise ValueError(f"Selected row has no prior_path or best_checkpoint: {best_row}")

    if best_variant == "unknown" and best_model_class in BC_TASK_VARIANTS:
        best_variant = best_model_class

    name = (best_row.get("name") or "best_prior").strip() or "best_prior"
    return BestPrior(
        row=best_row,
        success_rate=float(best_rate),
        prior_checkpoint=prior_checkpoint,
        prior_config=(best_row.get("config") or "").strip(),
        name=name,
        variant=best_variant,
        model_class=best_model_class,
    )


def _select_prior(args: argparse.Namespace, csv_path: Path, success_col: str) -> BestPrior:
    return _select_best_prior(
        csv_path,
        success_col,
        selection_policy=args.selection_policy,
        success_threshold=float(args.success_threshold),
        model_class_order=args.model_class_order,
        model_class_col=args.model_class_col,
    )


def _robot_task_prefix(robot_id: str) -> str:
    prefix = ROBOT_TASK_PREFIXES.get(robot_id)
    if prefix is None:
        known = ", ".join(sorted(ROBOT_TASK_PREFIXES))
        raise ValueError(f"No Kuafu dual robot prefix mapping for robot_id={robot_id!r}; known: {known}")
    return prefix


def _robot_dual_root(robot_id: str) -> Path:
    explicit = os.environ.get("KUAFU_TASK_ARTIFACT_ROOT", "").strip()
    if explicit:
        base = Path(explicit).expanduser().resolve()
    else:
        configured = os.environ.get("KUAFU_ARTIFACT_ROOT", "").strip()
        root = Path(configured).expanduser().resolve() if configured else REPO_ROOT.parent / "KuafuArtifacts"
        base = root / "standalone" / robot_id
    return base / "residual" / "launchers"


def _resolve_output_dir(task_slug: str, override: str | None, robot_id: str) -> Path:
    if override:
        return resolve_artifact_path(override, "Residual launcher output")
    suffix = DUAL_TASK_SUFFIXES.get(task_slug)
    if suffix is None:
        raise ValueError(f"No prior-conditioned residual output directory mapping for task_slug={task_slug!r}")
    return _robot_dual_root(robot_id) / f"{_robot_task_prefix(robot_id)}_{suffix}"



def _launcher_text(args: argparse.Namespace, *, task: str, best: BestPrior) -> str:
    initial_weights_params = str(args.initial_weights_params).strip()
    if not initial_weights_params:
        raise ValueError("--initial_weights_params is required for the active residual-hparam pipeline")

    return f'''#!/usr/bin/env bash
# =============================================================================
# Kuafu residual-hparam run - generated from BC prior sweep CSV
# Selected BC prior row: {best.name}
# Selected model class: {best.model_class}
# Prior config: {best.prior_config}
# Prior checkpoint: {best.prior_checkpoint}
# =============================================================================
set -euo pipefail

KUAFU_EXECUTION_ROBOT_ID="${{KUAFU_EXECUTION_ROBOT_ID:-{args.robot_id}}}"
KUAFU_ARTIFACT_ROBOT_ID="${{KUAFU_ARTIFACT_ROBOT_ID:-{args.robot_id}}}"

export GEMINI_API_KEY="${{GEMINI_API_KEY:-}}"
export CUDA_VISIBLE_DEVICES="${{KUAFU_CUDA_VISIBLE_DEVICES:-{args.cuda_visible_devices}}}"

TASK="${{KUAFU_TASK:-{task}}}"
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
GENERATED_SOURCE_ROOT={_shell_dq(str(REPO_ROOT))}
KUAFU_REPO_ROOT="${{KUAFU_PROJECT_ROOT:-${{KUAFU_REPO_ROOT:-${{GENERATED_SOURCE_ROOT}}}}}}"
if [[ ! -f "${{KUAFU_REPO_ROOT}}/src/kuafu/rl/launchers/prior_conditioned/dual_launcher_common.sh" ]]; then
    echo "Cannot locate Kuafu source. Set KUAFU_PROJECT_ROOT to the checkout root." >&2
    exit 1
fi
export KUAFU_PROJECT_ROOT="${{KUAFU_REPO_ROOT}}"
export PYTHONPATH="${{KUAFU_PROJECT_ROOT}}/src${{PYTHONPATH:+:${{PYTHONPATH}}}}"
{('export KUAFU_PROGRAM_PATH=' + _shell_dq(args.program_path)) if args.program_path else ''}
source "${{KUAFU_REPO_ROOT}}/src/kuafu/rl/launchers/prior_conditioned/dual_launcher_common.sh"
kuafu_dual_init_runtime_options
cd "${{KUAFU_PROJECT_ROOT}}"

DEVICE="${{KUAFU_DEVICE:-{args.device}}}"
NUM_ENVS={int(args.num_envs)}
NUM_WORKERS={int(args.num_workers)}

INIT_TRAIN_EPOCH={int(args.init_train_epoch)}
MAX_TRAIN_EPOCH={int(args.max_train_epoch)}
EPOCH_TRAIN_INCREMENT={int(args.epoch_train_increment)}

RESIDUAL_HPARAM_ITERS={int(args.residual_hparam_iters)}

GPT_MODEL="{args.gpt_model}"
TEMPERATURE={float(args.temperature)}
GEMINI_MODEL="{args.gemini_model}"
VIDEO_FEEDBACK="{args.video_feedback_flag}"
VIDEO_LENGTH={int(args.video_length)}
VIDEO_NUM_ENVS_TO_RECORD={int(args.video_num_envs_to_record)}
FEEDBACK_SUBSAMPLING={int(args.feedback_subsampling)}

INITIAL_EVAL_CHECKPOINT=""
INITIAL_CHECKPOINT=""
INITIAL_PRIOR_CHECKPOINT={_shell_dq(best.prior_checkpoint)}
INITIAL_PRIOR_MODE="residual_rl"
INITIAL_PRIOR_OBS_STATS_COUNT="{args.initial_prior_obs_stats_count}"
INITIAL_PRIOR_VERBOSE=""
INITIAL_PRIOR_CRITIC_INIT_MODE="{args.initial_prior_critic_init_mode}"
INITIAL_WEIGHTS_PARAMS="${{INITIAL_WEIGHTS_PARAMS:-{_shell_default(initial_weights_params)}}}"
INITIAL_AGENT_HYPERPARAMETERS="${{INITIAL_AGENT_HYPERPARAMETERS:-}}"
kuafu_dual_require_precomputed_weights_params

RESTART_EACH_ITER_FROM_INITIAL_WARM_START="--restart_each_iter_from_initial_warm_start"
STOP_ON_ERROR="{args.stop_on_error_flag}"
ENV_SEED={int(args.env_seed)}
RENDER=""
PRINT_PROGRAM_FUNCTIONS=""

LAUNCH_SCRIPT="${{SCRIPT_DIR}}/$(basename "${{BASH_SOURCE[0]}}")"

if [[ -n "${{INITIAL_CHECKPOINT}}" && -n "${{INITIAL_PRIOR_CHECKPOINT}}" && "${{INITIAL_PRIOR_MODE}}" != "residual_rl" ]]; then
    echo "ERROR: Choose only one warm-start source unless INITIAL_PRIOR_MODE=residual_rl." >&2
    exit 1
fi

CMD=(
    "${{PYTHON_BIN:-python}}" "${{KUAFU_PROJECT_ROOT}}/scripts/residual_rl/train_residual_hparams.py"
    --launch_script "${{LAUNCH_SCRIPT}}"
    --task "${{TASK}}"
    --device "${{DEVICE}}"
    --num_workers "${{NUM_WORKERS}}"
    --num_envs "${{NUM_ENVS}}"
    --env_seed "${{ENV_SEED}}"
    --init_train_epoch "${{INIT_TRAIN_EPOCH}}"
    --max_train_epoch "${{MAX_TRAIN_EPOCH}}"
    --epoch_train_increment "${{EPOCH_TRAIN_INCREMENT}}"
    --feedback_subsampling "${{FEEDBACK_SUBSAMPLING}}"
    --residual_hparam_iters "${{RESIDUAL_HPARAM_ITERS}}"
    --gpt_model "${{GPT_MODEL}}"
    --temperature "${{TEMPERATURE}}"
    --gemini_model "${{GEMINI_MODEL}}"
    "${{KUAFU_ROBOT_ARGS[@]}}"
    ${{VIDEO_FEEDBACK}}
    --video_length "${{VIDEO_LENGTH}}"
    --video_num_envs_to_record "${{VIDEO_NUM_ENVS_TO_RECORD}}"
    ${{RENDER}}
)

[[ -n "${{INITIAL_EVAL_CHECKPOINT}}" ]] && CMD+=(--initial_eval_checkpoint "${{INITIAL_EVAL_CHECKPOINT}}")
[[ -n "${{INITIAL_CHECKPOINT}}" ]] && CMD+=(--initial_checkpoint "${{INITIAL_CHECKPOINT}}")
[[ -n "${{INITIAL_PRIOR_CHECKPOINT}}" ]] && CMD+=(--initial_prior_checkpoint "${{INITIAL_PRIOR_CHECKPOINT}}")
[[ -n "${{INITIAL_PRIOR_MODE}}" ]] && CMD+=(--initial_prior_mode "${{INITIAL_PRIOR_MODE}}")
[[ -n "${{INITIAL_PRIOR_OBS_STATS_COUNT}}" ]] && CMD+=(--initial_prior_obs_stats_count "${{INITIAL_PRIOR_OBS_STATS_COUNT}}")
[[ -n "${{INITIAL_PRIOR_CRITIC_INIT_MODE}}" ]] && CMD+=(--initial_prior_critic_init_mode "${{INITIAL_PRIOR_CRITIC_INIT_MODE}}")
[[ -n "${{INITIAL_PRIOR_VERBOSE}}" ]] && CMD+=(${{INITIAL_PRIOR_VERBOSE}})
[[ -n "${{INITIAL_WEIGHTS_PARAMS}}" ]] && CMD+=(--initial_weights_params "${{INITIAL_WEIGHTS_PARAMS}}")
[[ -n "${{INITIAL_AGENT_HYPERPARAMETERS}}" ]] && CMD+=(--initial_agent_hyperparameters "${{INITIAL_AGENT_HYPERPARAMETERS}}")
[[ -n "${{RESTART_EACH_ITER_FROM_INITIAL_WARM_START}}" ]] && CMD+=(${{RESTART_EACH_ITER_FROM_INITIAL_WARM_START}})
[[ -n "${{STOP_ON_ERROR}}" ]] && CMD+=(${{STOP_ON_ERROR}})
[[ -n "${{PRINT_PROGRAM_FUNCTIONS}}" ]] && CMD+=(${{PRINT_PROGRAM_FUNCTIONS}})

echo "========================================"
echo " Kuafu: Residual Hyperparameter Run - ${{TASK}}"
echo "========================================"
echo "Workers:           ${{NUM_WORKERS}}"
echo "Envs:              ${{NUM_ENVS}}"
echo "Epochs:            ${{INIT_TRAIN_EPOCH}} -> ${{MAX_TRAIN_EPOCH}} (+${{EPOCH_TRAIN_INCREMENT}})"
echo "Residual hparams:  ${{RESIDUAL_HPARAM_ITERS}} iters"
echo "Stage tuning:      off"
echo "Global finetune:   off"
echo "LLM:               ${{GPT_MODEL}} (temp=${{TEMPERATURE}})"
echo "Initial prior:     ${{INITIAL_PRIOR_CHECKPOINT}}"
echo "Weights/params:    ${{INITIAL_WEIGHTS_PARAMS}}"
echo "Agent hparams:     ${{INITIAL_AGENT_HYPERPARAMETERS:-<agent yaml defaults>}}"
echo "Execution robot:   ${{EXECUTION_ROBOT_ID:-<inferred from task>}}"
echo "Artifact robot:    ${{ARTIFACT_ROBOT_ID:-<default>}}"
echo "Restart from warm: ${{RESTART_EACH_ITER_FROM_INITIAL_WARM_START:-<off>}}"
echo "Stop on error:     ${{STOP_ON_ERROR:-<off>}}"
echo "========================================"
echo ""
echo "Running: ${{CMD[*]}}"
echo ""

if kuafu_dual_bool_enabled "${{KUAFU_LAUNCH_DRY_RUN:-}}"; then
    kuafu_dual_print_command "train" "${{CMD[@]}}"
    exit 0
fi

: "${{OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script.}}"
export OPENAI_API_KEY
exec "${{CMD[@]}}"
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a prior-conditioned residual launcher for residual-RL hyperparameter search.")
    parser.add_argument("--csv", required=True, help="BC prior sweep CSV.")
    parser.add_argument("--task_name", required=True, help="Base Isaac task name.")
    parser.add_argument("--program_path", default=os.environ.get("KUAFU_PROGRAM_PATH"), help="Learned-program directory shared with collection/evaluation.")
    parser.add_argument("--robot_id", default="fr3_finray", help="Robot id used for success column and launcher defaults.")
    parser.add_argument("--task_slug", default=None, help="Override inferred task slug.")
    parser.add_argument("--success_rate_col", default=None, help="Override success-rate CSV column.")
    parser.add_argument(
        "--selection_policy",
        choices=SELECTION_POLICIES,
        default="smallest_class_over_threshold",
        help=(
            "Prior selection policy. smallest_class_over_threshold and "
            "smallest_eligible_model_class are equivalent."
        ),
    )
    parser.add_argument(
        "--success_threshold",
        "--success_threshold_pct",
        dest="success_threshold",
        type=float,
        default=20.0,
        help="Eligibility threshold for threshold-based model-class selection, expressed as percent.",
    )
    parser.add_argument(
        "--model_class_col",
        default="variant",
        help="CSV column that defines model class before falling back to inferred variant.",
    )
    parser.add_argument(
        "--model_class_order",
        nargs="+",
        default=list(DEFAULT_MODEL_CLASS_ORDER),
        help="Preferred model class order; accepts space-separated tokens or comma-separated groups.",
    )
    parser.add_argument(
        "--initial_weights_params",
        required=True,
        help="Precomputed best.json from automatic tolerance assignment and the post-BC weight query.",
    )
    parser.add_argument("--output_dir", default=None, help="Override output directory.")
    parser.add_argument("--output_name", default=None, help="Override generated launcher filename.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing launcher.")
    parser.add_argument("--dry_run", action="store_true", help="Print what would be generated.")

    parser.add_argument("--cuda_visible_devices", default="0,1", help="Default CUDA_VISIBLE_DEVICES for the launcher.")
    parser.add_argument("--device", default="cuda:0,1", help="Default residual-training device string for the launcher.")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_envs", type=int, default=8192)
    parser.add_argument("--init_train_epoch", type=int, default=2000)
    parser.add_argument("--max_train_epoch", type=int, default=8000)
    parser.add_argument("--epoch_train_increment", type=int, default=500)
    parser.add_argument(
        "--residual_hparam_iters", type=int, default=5,
        help="Model tuning iterations after initial training; 0 runs the initial training trial only.",
    )
    parser.add_argument("--gpt_model", default="gpt-5.5")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gemini_model", default="gemini-3.1-pro-preview")
    parser.add_argument("--video_feedback_flag", default="--no-video_feedback")
    parser.add_argument("--video_length", type=int, default=400)
    parser.add_argument("--video_num_envs_to_record", type=int, default=9)
    parser.add_argument("--feedback_subsampling", type=int, default=10)
    parser.add_argument("--initial_prior_obs_stats_count", default="1000000")
    parser.add_argument(
        "--initial_prior_critic_init_mode",
        choices=["actor_only_random_critic", "actor_and_critic_prior", "actor_only_zero_critic"],
        default="actor_only_random_critic",
    )
    parser.add_argument("--stop_on_error_flag", default="")
    parser.add_argument("--env_seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = _recipe_or_artifact_path(args.csv, "BC prior sweep CSV")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    initial_weights_params = _recipe_or_artifact_path(
        args.initial_weights_params, "Residual weights/parameters"
    )
    if not args.dry_run:
        if not initial_weights_params.exists():
            raise FileNotFoundError(f"initial_weights_params not found: {args.initial_weights_params}")
        if not initial_weights_params.is_file():
            raise ValueError(f"initial_weights_params must be a file, got: {args.initial_weights_params}")
    args.initial_weights_params = str(initial_weights_params)

    task_slug = (
        TASK_SLUG_ALIASES.get(_safe_token(args.task_slug), _safe_token(args.task_slug))
        if args.task_slug
        else _infer_task_slug(args.task_name)
    )
    success_col = _success_col(args.robot_id, args.success_rate_col)
    selection_policy = _normalize_selection_policy(args.selection_policy)
    best = _select_prior(args, csv_path, success_col)
    task = _task_with_variant(args.task_name, best.variant)
    output_dir = _resolve_output_dir(task_slug, args.output_dir, args.robot_id)
    if output_dir == REPO_ROOT or REPO_ROOT in output_dir.parents:
        raise ValueError(
            f"Residual launcher output must be outside the source project ({REPO_ROOT}); got {output_dir}"
        )
    output_name = args.output_name or f"{_safe_token(best.name)}_residual_hparams.sh"
    output_path = output_dir / output_name

    print("=== Residual Hyperparameter Launcher ===")
    print(f"  CSV:          {csv_path}")
    print(f"  Success col:  {success_col}")
    if selection_policy == "best":
        print(f"  Selection:    {args.selection_policy}")
    else:
        order_text = ",".join(_model_class_order(args.model_class_order))
        print(
            f"  Selection:    {args.selection_policy} "
            f"(threshold={args.success_threshold:g}, order={order_text}, class_col={args.model_class_col})"
        )
    print(f"  Best row:     {best.name} ({best.success_rate:.3f})")
    print(f"  Model class:  {best.model_class}")
    print(f"  Prior:        {best.prior_checkpoint}")
    print(f"  Task:         {task}")
    print(f"  Weights:      {args.initial_weights_params}")
    print(f"  Output:       {output_path}")
    if args.dry_run:
        return
    if output_path.exists() and not args.overwrite:
        existing_text = output_path.read_text(encoding="utf-8")
        stale_tokens = (
            "--stage_success_fraction",
            "--maximun_proposals",
            "--global_finetune_iters",
            "--global_finetune_max_false_to_true_ratio",
            "--stage_advance_completion_rate_cap",
        )
        if not any(token in existing_text for token in stale_tokens):
            print(f"[resume] launcher already exists: {output_path}")
            return
        print(f"[resume] replacing stale stage/global launcher: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_launcher_text(args, task=task, best=best), encoding="utf-8")
    os.chmod(output_path, 0o755)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
