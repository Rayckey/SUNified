"""Program-free Panda scenes for LLM task formulation.

The production Panda task classes contain the scene geometry and observation
plumbing needed by formulation, but their normal configurations also point at a
previously learned task program and implement hand-written success predicates.
The adapters in this module deliberately retain only the simulator and robot
plumbing:

* no learned stage program is loaded;
* no task-success predicate is evaluated;
* every environment reward is the zero-valued dummy reward; and
* no environment-level termination can be mistaken for formulation success.

The separate :class:`kuafu.mpc.simulation.sun_program_executor.SUNProgramExecutor` instance owned by a
formulation runner is therefore the only component that proposes and evaluates
the candidate predicates and residuals.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import torch

from kuafu.isaac.learned.panda.panda_cabinet_door_env_learned import (
    PandaCabinetDoorEnvLearned,
)
from kuafu.isaac.learned.panda.panda_can_env_learned import (
    PandaCanEnvLearned,
)
from kuafu.isaac.learned.panda.panda_drawer_env_learned import (
    PandaDrawerEnvLearned,
)
from kuafu.isaac.learned.panda.panda_drawer_stack_env_learned import (
    PandaDrawerStackEnv,
)
from kuafu.isaac.learned.panda.panda_estop_closed_env_learned import (
    PandaEStopClosedEnvLearned,
)
from kuafu.isaac.learned.panda.panda_estop_env_learned import (
    PandaEStopEnvLearned,
)
from kuafu.isaac.learned.panda.panda_hang_cup_env_learned import (
    PandaHangCupEnvLearned,
)
from kuafu.isaac.learned.panda.panda_pen_cup_env_learned import (
    PandaPenCupEnvLearned,
)
from kuafu.isaac.learned.panda.panda_stack2_env_learned import (
    PandaStack2EnvLearned,
)
from kuafu.isaac.learned.panda.panda_stack3_env_learned import (
    PandaStack3EnvLearned,
)


class _ProgramFreePandaMixin:
    """Enforce the simulator contract required by formulation runners."""

    _kuafu_program_free_formulation_env = True

    def __init__(self, cfg: Any, render_mode: str | None = None, **kwargs: Any):
        # The original learn-programs runners disabled reset randomization. Keep
        # one reproducible Panda scene across restart/replay corrections.
        cfg.seed = 0 if cfg.seed is None else cfg.seed
        self._formulation_reset_seed = cfg.seed

        # Config-class dictionaries may otherwise be shared between instances.
        # Copy before clearing the learned-program path so production configs are
        # never modified as a side effect of a formulation run.
        cfg.mpc_config = deepcopy(dict(cfg.mpc_config))
        cfg.mpc_config["load_program"] = None
        cfg.mpc_config["device"] = str(cfg.sim.device)

        # SUNProgramExecutor emits absolute Panda joint targets during formulation.
        cfg.action_mode = "absolute_joint"
        cfg.action_space = 9

        # These are defensive as well as documentary: stage rewards and their
        # diagnostics stay off even if the parent defaults change later.
        cfg.skip_dense_reward_computation = True
        cfg.use_external_rewards = False
        cfg.reward_print_every = 0
        cfg.episode_outcome_print_every = 0
        cfg.collect_terminal_stuck_constraint_stats = False

        super().__init__(cfg, render_mode, **kwargs)

        if self._has_program_stages():
            raise RuntimeError(
                "A formulation Panda environment loaded stage program; "
                "program-free execution is required."
            )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return no terminations without invoking a task-success predicate."""

        inactive = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cached_success = inactive
        self._cached_failure = inactive
        return inactive, inactive.clone()

    def _get_rewards(self) -> torch.Tensor:
        """Return the formulation environment's zero-valued dummy reward."""

        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def step(self, actions: torch.Tensor):
        """Accept the single joint-target vector emitted by learn-programs SUNProgramExecutor."""
        # The original manipulation scenes broadcast one trajectory row to the
        # environment batch. Panda indexes actions by environment explicitly.
        # Keep that adaptation here for both formulation and program replay.
        if actions.ndim == 1:
            actions = actions.unsqueeze(0).expand(self.num_envs, -1)
        return super().step(actions)

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Replay corrections from the same initial scene and robot state."""
        if seed is not None:
            self._formulation_reset_seed = seed
        return super().reset(seed=self._formulation_reset_seed, options=options)


class PandaDrawerFormulationEnv(
    _ProgramFreePandaMixin, PandaDrawerEnvLearned
):
    pass


class PandaDrawerStackFormulationEnv(
    _ProgramFreePandaMixin, PandaDrawerStackEnv
):
    pass


class PandaCabinetDoorFormulationEnv(
    _ProgramFreePandaMixin, PandaCabinetDoorEnvLearned
):
    pass


class PandaCanFormulationEnv(_ProgramFreePandaMixin, PandaCanEnvLearned):
    pass


class PandaEStopCloseFormulationEnv(
    _ProgramFreePandaMixin, PandaEStopEnvLearned
):
    pass


class PandaEStopOpenFormulationEnv(
    _ProgramFreePandaMixin, PandaEStopClosedEnvLearned
):
    pass


class PandaHangCupFormulationEnv(
    _ProgramFreePandaMixin, PandaHangCupEnvLearned
):
    pass


class PandaPenCupFormulationEnv(
    _ProgramFreePandaMixin, PandaPenCupEnvLearned
):
    pass


class PandaStack2FormulationEnv(
    _ProgramFreePandaMixin, PandaStack2EnvLearned
):
    pass


class PandaStack3FormulationEnv(
    _ProgramFreePandaMixin, PandaStack3EnvLearned
):
    pass


_ENV_SPECS = {
    "Formulate-Panda-Drawer-v0": (
        PandaDrawerFormulationEnv,
        "kuafu.isaac.learned.panda.panda_drawer_env_learned:PandaDrawerEnvLearnedCfg",
    ),
    "Formulate-Panda-Drawer-Stack-v0": (
        PandaDrawerStackFormulationEnv,
        "kuafu.isaac.learned.panda.panda_drawer_stack_env_learned:PandaDrawerStackEnvCfg",
    ),
    "Formulate-Panda-Cabinet-Door-Either-v0": (
        PandaCabinetDoorFormulationEnv,
        "kuafu.isaac.learned.panda.panda_cabinet_door_env_learned:PandaCabinetDoorEnvLearnedCfg",
    ),
    "Formulate-Panda-Cabinet-Doors-Both-v0": (
        PandaCabinetDoorFormulationEnv,
        "kuafu.isaac.learned.panda.panda_cabinet_door_env_learned:PandaCabinetDoorEnvLearnedCfg",
    ),
    "Formulate-Panda-Can-v0": (
        PandaCanFormulationEnv,
        "kuafu.isaac.learned.panda.panda_can_env_learned:PandaCanEnvLearnedCfg",
    ),
    "Formulate-Panda-EStop-Close-v0": (
        PandaEStopCloseFormulationEnv,
        "kuafu.isaac.learned.panda.panda_estop_env_learned:PandaEStopEnvLearnedCfg",
    ),
    "Formulate-Panda-EStop-Open-v0": (
        PandaEStopOpenFormulationEnv,
        "kuafu.isaac.learned.panda.panda_estop_closed_env_learned:PandaEStopClosedEnvLearnedCfg",
    ),
    "Formulate-Panda-Hang-Cup-v0": (
        PandaHangCupFormulationEnv,
        "kuafu.isaac.learned.panda.panda_hang_cup_env_learned:PandaHangCupEnvLearnedCfg",
    ),
    "Formulate-Panda-Pen-Cup-v0": (
        PandaPenCupFormulationEnv,
        "kuafu.isaac.learned.panda.panda_pen_cup_env_learned:PandaPenCupEnvLearnedCfg",
    ),
    "Formulate-Panda-Stack-2-v0": (
        PandaStack2FormulationEnv,
        "kuafu.isaac.learned.panda.panda_stack2_env_learned:PandaStack2EnvLearnedCfg",
    ),
    "Formulate-Panda-Stack-3-v0": (
        PandaStack3FormulationEnv,
        "kuafu.isaac.learned.panda.panda_stack3_env_learned:PandaStack3EnvLearnedCfg",
    ),
}


for _env_id, (_entry_point, _cfg_entry_point) in _ENV_SPECS.items():
    if _env_id not in gym.registry:
        gym.register(
            id=_env_id,
            entry_point=_entry_point,
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": _cfg_entry_point},
        )


def assert_program_free_panda_env(env: Any) -> None:
    """Fail fast if a formulation runner is accidentally wired to another env."""

    unwrapped = env.unwrapped
    if not getattr(unwrapped, "_kuafu_program_free_formulation_env", False):
        raise RuntimeError("Formulation requires a program-free Panda environment.")
    if unwrapped.cfg.mpc_config.get("load_program"):
        raise RuntimeError("Formulation environment must not load a learned program.")
    if unwrapped._has_program_stages():
        raise RuntimeError("Formulation environment must not contain stage predicates.")


__all__ = [
    "assert_program_free_panda_env",
    "PandaCabinetDoorFormulationEnv",
    "PandaCanFormulationEnv",
    "PandaDrawerFormulationEnv",
    "PandaDrawerStackFormulationEnv",
    "PandaEStopCloseFormulationEnv",
    "PandaEStopOpenFormulationEnv",
    "PandaHangCupFormulationEnv",
    "PandaPenCupFormulationEnv",
    "PandaStack2FormulationEnv",
    "PandaStack3FormulationEnv",
]
