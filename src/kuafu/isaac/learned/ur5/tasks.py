# SPDX-License-Identifier: Apache-2.0
"""UR5 embodiments of Kuafu's existing tabletop tasks and physical success checks."""

from isaaclab.utils import configclass
from kuafu.paths import LEARNED_PROGRAM_ROOT

from ..kinova.kinova_drawer_env_learned import KinovaDrawerEnvLearned, KinovaDrawerEnvLearnedCfg
from ..kinova.kinova_drawer_stack_env_learned import KinovaDrawerStackEnv, KinovaDrawerStackEnvCfg
from ..kinova.kinova_stack2_env_learned import KinovaStack2EnvLearned, KinovaStack2EnvLearnedCfg
from ..kinova.kinova_stack3_env_learned import KinovaStack3EnvLearned, KinovaStack3EnvLearnedCfg
from ..kinova.kinova_hang_cup_env_learned import KinovaHangCupEnvLearned, KinovaHangCupEnvLearnedCfg
from ..kinova.kinova_can_env_learned import KinovaCanEnvLearned, KinovaCanEnvLearnedCfg
from ..kinova.kinova_pen_cup_env_learned import KinovaPenCupEnvLearned, KinovaPenCupEnvLearnedCfg
from ..kinova.kinova_estop_env_learned import KinovaEStopEnvLearned, KinovaEStopEnvLearnedCfg
from ..kinova.kinova_estop_closed_env_learned import KinovaEStopClosedEnvLearned, KinovaEStopClosedEnvLearnedCfg
from .robot import UR5ConfigMixin, UR5EnvMixin


@configclass
class UR5DrawerEnvCfg(UR5ConfigMixin, KinovaDrawerEnvLearnedCfg):
    pass


class UR5DrawerEnv(UR5EnvMixin, KinovaDrawerEnvLearned):
    pass


@configclass
class UR5DrawerStackEnvCfg(UR5ConfigMixin, KinovaDrawerStackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.mpc_config["load_program"] = str(LEARNED_PROGRAM_ROOT / "ur5-drawer-stack")


class UR5DrawerStackEnv(UR5EnvMixin, KinovaDrawerStackEnv):
    pass


@configclass
class UR5Stack2EnvCfg(UR5ConfigMixin, KinovaStack2EnvLearnedCfg):
    pass


class UR5Stack2Env(UR5EnvMixin, KinovaStack2EnvLearned):
    pass


@configclass
class UR5Stack3EnvCfg(UR5ConfigMixin, KinovaStack3EnvLearnedCfg):
    pass


class UR5Stack3Env(UR5EnvMixin, KinovaStack3EnvLearned):
    pass


@configclass
class UR5HangCupEnvCfg(UR5ConfigMixin, KinovaHangCupEnvLearnedCfg):
    pass


class UR5HangCupEnv(UR5EnvMixin, KinovaHangCupEnvLearned):
    pass


@configclass
class UR5CanEnvCfg(UR5ConfigMixin, KinovaCanEnvLearnedCfg):
    pass


class UR5CanEnv(UR5EnvMixin, KinovaCanEnvLearned):
    pass


@configclass
class UR5PenCupEnvCfg(UR5ConfigMixin, KinovaPenCupEnvLearnedCfg):
    pass


class UR5PenCupEnv(UR5EnvMixin, KinovaPenCupEnvLearned):
    pass


@configclass
class UR5EStopEnvCfg(UR5ConfigMixin, KinovaEStopEnvLearnedCfg):
    pass


class UR5EStopEnv(UR5EnvMixin, KinovaEStopEnvLearned):
    pass


@configclass
class UR5EStopClosedEnvCfg(UR5ConfigMixin, KinovaEStopClosedEnvLearnedCfg):
    pass


class UR5EStopClosedEnv(UR5EnvMixin, KinovaEStopClosedEnvLearned):
    pass
