"""CLI script for the separate residual-RL hyperparameter search pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from train import _build_arg_parser


def _remove_parser_options(parser, *option_strings: str) -> None:
    for option_string in option_strings:
        action = parser._option_string_actions.get(option_string)
        if action is None:
            continue
        for option in list(action.option_strings):
            parser._option_string_actions.pop(option, None)
        if action in parser._actions:
            parser._actions.remove(action)
        for group in parser._action_groups:
            if action in group._group_actions:
                group._group_actions.remove(action)


def main() -> None:
    parser = _build_arg_parser()
    parser.description = "Train residual RL while tuning only residual agent hyperparameters."
    _remove_parser_options(
        parser,
        "--stage_success_fraction",
        "--maximun_proposals",
        "--global_finetune_iters",
        "--global_finetune_max_false_to_true_ratio",
        "--stage_advance_completion_rate_cap",
    )
    if "--residual_hparam_iters" not in parser._option_string_actions:
        parser.add_argument(
            "--residual_hparam_iters",
            type=int,
            default=5,
            help="Model tuning iterations after initial training; 0 trains with supplied hyperparameters only.",
        )
    args = parser.parse_args()

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    os.environ["KUAFU_PRINT_PROGRAM_FUNCTIONS"] = "1" if args.print_program_functions else "0"

    source_root = Path(__file__).resolve().parents[2] / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    from kuafu.rl.cli.residual_hparams import run_from_cli

    run_from_cli(args)


if __name__ == "__main__":
    main()
