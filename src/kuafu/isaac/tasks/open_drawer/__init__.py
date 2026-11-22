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
    id="open-drawer",
    entry_point=f"{__name__}.open_drawer_env_learned:OpenDrawerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.open_drawer_env_learned:OpenDrawerEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "bc_mlp.json"),
    },
)

gym.register(
    id="open-fr3-finray-drawer",
    entry_point=f"{__name__}.open_fr3_finray_drawer_env_learned:OpenDrawerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.open_fr3_finray_drawer_env_learned:OpenDrawerEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_v2.yaml",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "bc_mlp.json"),
    },
)
