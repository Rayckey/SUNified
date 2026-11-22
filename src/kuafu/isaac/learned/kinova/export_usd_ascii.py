#!/usr/bin/env python3
"""Export a composed USD stage to USDA for text inspection."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_usd")
    parser.add_argument("output_usda")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = False
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd  # noqa: E402


def main() -> None:
    stage = Usd.Stage.Open(args_cli.input_usd)
    if not stage.Export(args_cli.output_usda):
        raise RuntimeError(f"Failed to export {args_cli.input_usd} to {args_cli.output_usda}")
    print(args_cli.output_usda, flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
