"""Compatibility shim for the deprecated stage/global Kuafu runner.

New residual-RL hyperparameter search should import ResidualBaseRunner from
kuafu.rl.core.runner_base. The old ResidualRunner remains available here so
legacy train.py launchers keep working.
"""

from __future__ import annotations

from kuafu.rl.core.deprecated.runner_stage_global import ResidualRunner

__all__ = ["ResidualRunner"]
