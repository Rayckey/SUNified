# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cabinet environment.
"""

import gymnasium as gym
import os
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="fr3-finray-drawer-stack",
    entry_point=f"{__name__}.fr3_finray_drawer_stack_env_learned:DrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_drawer_stack_env_learned:DrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "bc_mlp.json"),
    },
)
