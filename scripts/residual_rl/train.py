"""CLI entry point for residual-RL training."""

import argparse
import os
import sys
from pathlib import Path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train with residual reward and policy tuning.")
    parser.add_argument("--task", type=str, default="FR3-FinRay-Table-Drawer", help="Robot-aware Gym task name (e.g. FR3-FinRay-Table-Drawer).")
    parser.add_argument(
        "--execution_robot_id",
        type=str,
        default=None,
        help="Robot env package to execute. Defaults to inference from --task; task prefixes select the robot package.",
    )
    parser.add_argument(
        "--artifact_robot_id",
        type=str,
        default=None,
        help="Robot id for artifact lineage metadata. Defaults to fr3_finray for existing priors/CSVs.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Simulation/RL device (e.g. cuda, cuda:0).")
    parser.add_argument("--num_workers", type=int, default=3, help="Number of parallel worker processes.")
    parser.add_argument("--num_envs", type=int, default=8192, help="Override number of envs (RL-Games constraint applies).")
    parser.add_argument("--env_seed", type=int, default=42, help="Environment seed.")
    parser.add_argument("--max_train_epoch", type=int, default=300, help="RL train iterations in the last epoch.")
    parser.add_argument("--init_train_epoch", type=int, default=200, help="RL train iterations in the first epoch.")
    parser.add_argument("--epoch_train_increment", type=int, default=5, help="RL train iterations increment per epoch.")
    parser.add_argument("--feedback_subsampling", type=int, default=10, help="Subsample stride for metrics feedback.")

    parser.add_argument(
        "--stage_success_fraction",
        type=float,
        default=0.7,
        help="Fraction (0-1) of envs that must advance to pass a stage (e.g. 0.75).",
    )
    parser.add_argument(
        "--maximun_proposals",
        type=int,
        default=25,
        help="Total proposals allowed per stage (budget). Internally converted to stage iterations via ceil(maximun_proposals/num_workers).",
    )
    
    parser.add_argument("--global_finetune_iters", type=int, default=0, help="Global finetune iterations.")
    parser.add_argument(
        "--global_finetune_max_false_to_true_ratio",
        type=float,
        default=0.5,
        help=(
            "Early-stop global finetune when: "
            "true_success_rate >= stage_success_fraction AND "
            "false_success_rate / true_success_rate <= this ratio."
        ),
    )
    
    parser.add_argument(
        "--stage_advance_completion_rate_cap",
        type=float,
        default=0.5,
        help=(
            "When advancing stages: stop advancing if (true_success + false_complete)/total_episodes "
            ">= this value (e.g. 0.99 = 99 pct of episodes terminated)."
        ),
    )

    parser.add_argument("--gpt_model", type=str, default="gpt-5.5", help="OpenAI model name.")
    parser.add_argument("--temperature", type=float, default=1.0, help="LLM sampling temperature.")
    parser.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API key (optional).")

    parser.add_argument("--gemini_model", type=str, default="gemini-2.0-flash", help="Gemini model for video summarisation (e.g. gemini-2.0-flash, gemini-2.5-pro).")
    parser.add_argument("--render", action="store_true", default=False, help="Render (headless=False) in workers.")

    parser.add_argument("--video_feedback", action=argparse.BooleanOptionalAction, default=True, help="Generate playback videos for LLM.")
    parser.add_argument("--video_length", type=int, default=200, help="Playback video length in steps.")
    parser.add_argument("--video_num_envs_to_record", type=int, default=4, help="How many env videos to record.")

    # -- Checkpoint resumption options --
    parser.add_argument(
        "--initial_eval_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to an RL-Games checkpoint (.pth) to evaluate once at the start "
            "to seed rollout stats without further training."
        ),
    )
    warm_start_group = parser.add_mutually_exclusive_group()
    warm_start_group.add_argument(
        "--initial_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to an RL-Games checkpoint (.pth) to warm-start training from during "
            "the initial dry run. In residual_rl mode this may be used together with "
            "--initial_prior_checkpoint: the RL checkpoint resumes the residual policy, "
            "while the prior checkpoint reconstructs the frozen BC base policy."
        ),
    )
    parser.add_argument(
        "--initial_weights_params",
        type=str,
        default=None,
        help=(
            "Path to a JSON file with initial per-stage weights and parameters. "
            "These are applied during the initial bootstrap rollout and shown in the first LLM prompt. "
            "Expected format (same as best.json 'stages' dict):\n"
            '  {"0": {"weights": {"0": 1.0, ...}, "parameters": {"0": {"tolerance_m": 0.05}, ...}}, ...}'
        ),
    )
    parser.add_argument(
        "--initial_agent_hyperparameters",
        type=str,
        default=None,
        help=(
            "Path to residual-agent hyperparameters to use instead of the agent YAML defaults. "
            "Accepts a direct hyperparameter object, an external.json with Hyperparameters, "
            "or a residual-RL best.json with global.agent_hyperparameters."
        ),
    )
    parser.add_argument(
        "--initial_prior_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a BC prior checkpoint or BC run directory to seed the initial dry-run policy "
            "before RL training starts."
        ),
    )
    parser.add_argument(
        "--initial_prior_mode",
        type=str,
        choices=["full_actor", "residual_rl"],
        default="full_actor",
        help=(
            "How to use --initial_prior_checkpoint. "
            "`full_actor` keeps the existing BC->PPO actor warm-start flow. "
            "`residual_rl` keeps BC frozen forever and trains PPO only as a bounded residual."
        ),
    )
    parser.add_argument(
        "--initial_prior_obs_stats_count",
        type=float,
        default=1e6,
        help="Synthetic sample count used when copying BC observation stats into RL-Games normalization.",
    )
    parser.add_argument(
        "--initial_prior_verbose",
        action="store_true",
        default=False,
        help="Print verbose BC prior weight-copy logging during the initial dry run.",
    )
    parser.add_argument(
        "--initial_prior_critic_init_mode",
        type=str,
        choices=[
            "actor_only_random_critic",
            "actor_and_critic_prior",
            "actor_only_zero_critic",
        ],
        default="actor_only_random_critic",
        help=(
            "How to initialize the PPO critic when warm-starting from a BC actor prior: "
            "`actor_only_random_critic` leaves the critic at RL-Games random init, "
            "`actor_and_critic_prior` requires and loads critic weights from the BC checkpoint, "
            "`actor_only_zero_critic` zeros critic weights and value normalization stats after actor load."
        ),
    )
    parser.add_argument(
        "--initial_prior_freeze_actor_epochs",
        type=int,
        default=0,
        help=(
            "Number of epochs to freeze the BC actor (policy trunk, mean head, sigma) after loading "
            "a prior checkpoint, so early noisy RL gradients cannot overwrite the BC initialization. "
            "0 disables freezing (default). Only takes effect when --initial_prior_checkpoint is set."
        ),
    )
    parser.add_argument(
        "--initial_prior_sigma_ramp_epochs",
        type=int,
        default=0,
        help=(
            "While the actor is frozen, linearly ramp sigma (log-std) from its silenced "
            "value back to the saved PPO sigma over the LAST N frozen epochs. This way "
            "sigma is already restored by the time the actor is unfrozen, so PPO never "
            "sees an unfrozen actor with a degenerate near-zero sigma (which produces "
            "exploding importance ratios / log-probs and destroys the policy). The critic "
            "also gets the ramp window to re-calibrate V against noisier rollouts. 0 "
            "disables the ramp (sigma snaps from silenced to saved at unfreeze)."
        ),
    )
    parser.add_argument(
        "--initial_prior_sigma_final_val",
        type=float,
        default=None,
        help=(
            "Override the sigma ramp/restore target (log-std) used when the BC prior is "
            "unfrozen. Defaults to None = use the PPO config's saved sigma_init (for "
            "example -3 ⇒ std ≈ 0.05, or ~5 cm EE noise at clip_actions=1.0). On tight "
            "manipulation tasks the BC prior was trained from effectively deterministic "
            "demos, and restoring sigma to this level injects enough action noise to "
            "break stage predicates (for example the ~2 cm freeze_position check), so "
            "the policy appears to collapse at unfreeze even though its weights are "
            "unchanged by LR-warmup. Passing a tighter target like -5.5 (std ≈ 0.004, "
            "~4 mm) keeps post-unfreeze noise below BC precision and lets PPO expand "
            "sigma back upward only if exploration actually helps. Only takes effect "
            "when --initial_prior_checkpoint is set and --initial_prior_freeze_actor_epochs > 0."
        ),
    )
    parser.add_argument(
        "--initial_prior_actor_lr_warmup_epochs",
        type=int,
        default=0,
        help=(
            "After the actor is unfrozen (at --initial_prior_freeze_actor_epochs), linearly ramp "
            "the actor's optimizer LR from 0 up to the base adaptive LR over N epochs. The "
            "BC-initialized actor loads without any Adam state; the very first post-unfreeze "
            "gradient step therefore produces step_size = base_lr * sign(g) per parameter "
            "(Adam's cold-start behavior). With a default base_lr of 1e-2 and ~5 mini-epochs "
            "per PPO update, every actor weight moves by ~5e-2 in L∞ in a single epoch, which "
            "pushes the policy's mean output far outside [-1,1] and triggers the bounds loss. "
            "The adaptive LR scheduler then crushes last_lr to ~1e-6 in response to the KL "
            "explosion, but only after the BC prior is already corrupted. Ramping the actor's "
            "LR up from zero lets Adam accumulate a meaningful v_t (gradient-magnitude running "
            "average) before it is allowed to take full-sized steps. Critic LR is unaffected. "
            "0 disables the warmup (default). Only takes effect when "
            "--initial_prior_freeze_actor_epochs > 0."
        ),
    )
    
    parser.add_argument("--initial_prior_actor_lr_cap", type=float, default=0.0)
    
    parser.add_argument(
        "--initial_prior_freeze_input_running_mean_std",
        action="store_true",
        default=False,
        help=(
            "Pin rl_games' observation running_mean_std to the BC expert stats that "
            "apply_bc_prior_to_rl_games_agent seeds, so PPO's per-rollout updates do NOT drift "
            "the normalizer toward the rollout distribution. At num_envs=8192 this drift was "
            "shown to silently push the frozen BC weights out-of-distribution during the freeze "
            "window. Equivalent in intent to loading a raw-obs-baked prior with normalize_input=False, "
            "but keeps normalize_input=True so that post-freeze PPO still gets normalized inputs. "
            "Only takes effect when --initial_prior_checkpoint is set."
        ),
    )
    parser.add_argument(
        "--initial_prior_unfreeze_input_running_mean_std_after_freeze",
        action="store_true",
        default=False,
        help=(
            "If --initial_prior_freeze_input_running_mean_std is set, also unfreeze the "
            "observation running_mean_std at the same epoch the actor is unfrozen "
            "(--initial_prior_freeze_actor_epochs). This lets PPO's fine-tuning phase adapt the "
            "normalizer to its own rollout distribution after the BC prior has had its freeze window."
        ),
    )
    parser.add_argument(
        "--initial_prior_rms_unfreeze_after_actor_lr_warmup",
        action="store_true",
        default=False,
        help=(
            "Only meaningful when both --initial_prior_unfreeze_input_running_mean_std_after_freeze "
            "and --initial_prior_actor_lr_warmup_epochs > 0 are set. Defers the observation "
            "running_mean_std unfreeze from the actor-unfreeze boundary to the end of the actor "
            "LR warmup window (unfreeze_epoch + actor_lr_warmup_epochs). Without this, RMS drifts "
            "during PPO minibatch updates in the same epoch the actor is unfrozen: because the BC "
            "mean head was trained against the expert-seeded RMS stats, any shift of the normalizer "
            "changes network outputs with weights unchanged, producing a fake KL spike that triggers "
            "the adaptive LR scheduler to crush the base LR even though actor_lr was correctly held "
            "at 0 by warmup. With the flag enabled, the BC-seeded normalizer stays pinned until the "
            "actor has had its full warmup window of real gradient updates."
        ),
    )
    parser.add_argument(
        "--initial_prior_lr_schedule_freeze_during_warmup",
        action="store_true",
        default=False,
        help=(
            "Only meaningful when --initial_prior_actor_lr_warmup_epochs > 0. Pin rl_games' "
            "agent.scheduler.update(...) to return the pre-unfreeze last_lr for the duration "
            "of the actor LR warmup window. This prevents the default AdaptiveScheduler from "
            "reacting to the spurious one-shot KL that PPO reports at the actor-unfreeze "
            "boundary (empirically ~130 for a BC-initialized actor). Without this flag, the "
            "scheduler cuts last_lr by 100x+ in a single epoch and the actor LR warmup — "
            "which multiplies warmup_factor * last_lr — delivers effectively zero LR for the "
            "entire warmup window, stalling learning. The scheduler is restored unchanged at "
            "the end of warmup, at which point any subsequent KL is of natural PPO scale."
        ),
    )
    parser.add_argument(
        "--initial_prior_fixed_lr",
        type=str,
        default="",
        help=(
            "Optional fixed learning rate (as a float like 1e-4). When set, rl_games' "
            "adaptive LR scheduler is neutralized for the entire run: agent.scheduler.update(...) "
            "is patched to return the fixed LR unconditionally, and all optimizer param-group "
            "LRs are pinned to it. This disables the whole scheduler-avoidance coordination set "
            "(actor_lr_cap, actor_lr_warmup_epochs, lr_schedule_freeze_during_warmup, "
            "rms_unfreeze_after_actor_lr_warmup) as no-ops: the fixed LR IS the LR. Recommended "
            "for BC-warm-start regimes where the policy starts at a good operating point and "
            "the scheduler's from-scratch-PPO assumptions (aggressive early, tapering later) are "
            "inverted. Leave empty / 0 to use rl_games' default adaptive scheduler."
        ),
    )
    parser.add_argument(
        "--initial_prior_bc_anchor_epochs",
        type=int,
        default=0,
        help=(
            "For the first N epochs after actor unfreeze, add a temporary MSE anchor on the "
            "policy mean against the frozen BC prior model. This helps PPO fine-tune the BC "
            "policy without immediately drifting away from successful demonstrated behavior."
        ),
    )
    parser.add_argument(
        "--initial_prior_bc_anchor_coef",
        type=float,
        default=0.0,
        help=(
            "Coefficient for the temporary post-unfreeze BC anchor on the policy mean. "
            "0 disables the anchor."
        ),
    )
    parser.add_argument(
        "--initial_prior_bc_dataset_coef",
        type=float,
        default=0.0,
        help=(
            "Coefficient for the *supervised* BC-on-dataset loss. Unlike --initial_prior_bc_anchor_coef "
            "(which compares the policy mu on rollout obs to a frozen-copy-of-initial-policy's mu), this "
            "loss samples (obs_expert, action_expert) pairs from the BC HDF5 demonstration dataset and "
            "adds coef(t) * MSE(policy.mu(obs_expert_normalized), action_expert) to every PPO minibatch. "
            "Obs normalization for the dataset pathway is pinned to the BC checkpoint's saved stats, "
            "making the regularizer correct even after running_mean_std unfreezes. The coefficient "
            "decays linearly from coef to coef*0.1 over --initial_prior_bc_dataset_decay_epochs, then "
            "holds at the 0.1*coef floor for the rest of training. Recommended starting value: 10.0 "
            "(dataset-BC dominates, policy fine-tunes within the BC manifold). 0 disables."
        ),
    )
    parser.add_argument(
        "--initial_prior_bc_dataset_decay_epochs",
        type=int,
        default=0,
        help=(
            "Number of epochs after actor unfreeze over which the BC-dataset loss coefficient "
            "linearly decays from coef to coef*0.1. After this window, the coefficient holds at "
            "the 0.1*coef floor indefinitely (the loss is always-on, just weaker). 0 means "
            "constant coefficient (no decay)."
        ),
    )
    parser.add_argument(
        "--initial_prior_bc_dataset_batch_size",
        type=int,
        default=1024,
        help=(
            "Minibatch size sampled from the BC demonstration dataset per PPO step. "
            "1024 is usually sufficient; the dataset is pre-loaded to GPU so larger values are cheap."
        ),
    )
    parser.add_argument(
        "--initial_prior_bc_dataset_path",
        type=str,
        default="",
        help=(
            "Optional explicit path to the BC HDF5 dataset. When omitted (default), the dataset "
            "path is auto-resolved from the BC run's train_config.yaml adjacent to the prior "
            "checkpoint. Use this to override the dataset for experiments where the BC prior was "
            "trained on one dataset but you want to regularize against another."
        ),
    )
    parser.add_argument(
        "--initial_prior_bc_action_reward_coef",
        type=float,
        default=0.0,
        help=(
            "When > 0 and --initial_prior_mode=full_actor, adds a per-step reward penalty "
            "of coef * ||action - bc_prior(obs)||^2 to the environment reward. Unlike the "
            "gradient-level BC anchor loss, this penalty flows through GAE into the value "
            "function so both the actor and critic internalize the cost of diverging from BC. "
            "Recommended starting value: 0.1 to 1.0. Requires --initial_prior_checkpoint. "
            "0.0 disables."
        ),
    )
    parser.add_argument(
        "--restart_each_iter_from_initial_warm_start",
        action="store_true",
        default=False,
        help=(
            "After each LLM iteration, restart RL training from the initial warm-start "
            "source (--initial_checkpoint or --initial_prior_checkpoint) instead of "
            "starting those iterations from scratch."
        ),
    )
    parser.add_argument("--dry_run", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument(
        "--resume_from_prev_iter_checkpoint",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--restart_each_iter_from_initial_prior_checkpoint",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--print_program_functions",
        action="store_true",
        default=False,
        help="When the env's program_runtime (SUNProgramEvaluator) compiles residual/predicate scopes, print the generated function source.",
    )
    parser.add_argument(
        "--launch_script",
        type=str,
        default=None,
        help="Path to the bash script that launched this run. A copy is saved in the run directory with run_params.json.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        default=False,
        help="Abort the run immediately when any worker reports an error instead of continuing to the next LLM step.",
    )

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    os.environ["KUAFU_PRINT_PROGRAM_FUNCTIONS"] = "1" if args.print_program_functions else "0"

    # Make the unified package importable when this file is run directly.
    source_root = Path(__file__).resolve().parents[2] / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    from kuafu.paths import PROJECT_ROOT, resolve_artifact_path
    from kuafu.rl.cli.cil_utilities import run_from_cli

    def _source_recipe_or_artifact(
        value: str,
        label: str,
        source_roots: set[str],
    ) -> str:
        """Resolve immutable source inputs locally and generated inputs externally."""

        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            if candidate.parts and candidate.parts[0] in source_roots:
                return str((PROJECT_ROOT / candidate).resolve())
            return str(resolve_artifact_path(candidate, label))

        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(PROJECT_ROOT)
        except ValueError:
            return str(resolve_artifact_path(resolved, label))
        if relative.parts and relative.parts[0] in source_roots:
            return str(resolved)
        return str(resolve_artifact_path(resolved, label))

    external_inputs = {
        "initial_eval_checkpoint": "Initial evaluation checkpoint",
        "initial_checkpoint": "Initial residual checkpoint",
        "initial_prior_checkpoint": "Initial BC prior checkpoint",
        "initial_prior_bc_dataset_path": "Initial BC trajectory dataset",
    }
    for field, label in external_inputs.items():
        value = getattr(args, field, None)
        if value:
            setattr(args, field, str(resolve_artifact_path(value, label)))

    recipe_inputs = {
        "initial_weights_params": (
            "Initial residual weights/parameters",
            {"configs", "resources", "src"},
        ),
        "initial_agent_hyperparameters": (
            "Initial residual-agent hyperparameters",
            {"configs", "resources", "src"},
        ),
        "launch_script": (
            "Residual launch script",
            {"resources", "scripts"},
        ),
    }
    for field, (label, source_roots) in recipe_inputs.items():
        value = getattr(args, field, None)
        if value:
            setattr(args, field, _source_recipe_or_artifact(value, label, source_roots))

    run_from_cli(args)


if __name__ == "__main__":
    main()
