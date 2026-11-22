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
    id="FR3-FinRay-Table-Drawer",
    entry_point=f"{__name__}.fr3_finray_drawer_env_learned:FR3FinRayDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_drawer_env_learned:FR3FinRayDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Cabinet-Door",
    entry_point=f"{__name__}.fr3_finray_cabinet_door_env_learned:FR3FinRayCabinetDoorEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_cabinet_door_env_learned:FR3FinRayCabinetDoorEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Cabinet-Door-f",
    entry_point=f"{__name__}.fr3_finray_cabinet_door_env_learned:FR3FinRayCabinetDoorEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_cabinet_door_env_learned:FR3FinRayCabinetDoorEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Cabinet-Door-f.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Drawer-Stack",
    entry_point=f"{__name__}.fr3_finray_drawer_stack_env_learned:FR3FinRayTableDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_drawer_stack_env_learned:FR3FinRayTableDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Drawer-Stack-ff",
    entry_point=f"{__name__}.fr3_finray_drawer_stack_env_learned:FR3FinRayTableDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_drawer_stack_env_learned:FR3FinRayTableDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Drawer-Stack-ff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Drawer-Stack-fff",
    entry_point=f"{__name__}.fr3_finray_drawer_stack_env_learned:FR3FinRayTableDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_drawer_stack_env_learned:FR3FinRayTableDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Drawer-Stack-fff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Drawer-Stack-ffff",
    entry_point=f"{__name__}.fr3_finray_drawer_stack_env_learned:FR3FinRayTableDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_drawer_stack_env_learned:FR3FinRayTableDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Drawer-Stack-ffff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Drawer-Stack-ffff3",
    entry_point=f"{__name__}.fr3_finray_drawer_stack_env_learned:FR3FinRayTableDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_drawer_stack_env_learned:FR3FinRayTableDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Drawer-Stack-ffff3.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Drawer-Stack-fffff",
    entry_point=f"{__name__}.fr3_finray_drawer_stack_env_learned:FR3FinRayTableDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_drawer_stack_env_learned:FR3FinRayTableDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Drawer-Stack-fffff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Drawer-Stack-ffffff",
    entry_point=f"{__name__}.fr3_finray_drawer_stack_env_learned:FR3FinRayTableDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_drawer_stack_env_learned:FR3FinRayTableDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Drawer-Stack-ffffff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Pen-Cup",
    entry_point=f"{__name__}.fr3_finray_pen_cup_env_learned:FR3FinRayPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_pen_cup_env_learned:FR3FinRayPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_fr3_finray_drawer.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Hang-Cup",
    entry_point=f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Can",
    entry_point=f"{__name__}.fr3_finray_can_env_learned:FR3FinRayCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_can_env_learned:FR3FinRayCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)


gym.register(
    id="FR3-FinRay-Table-EStop-Close",
    entry_point=f"{__name__}.fr3_finray_estop_env_learned:FR3FinRayEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_estop_env_learned:FR3FinRayEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-EStop-Close-Long-Long-U",
    entry_point=f"{__name__}.fr3_finray_estop_env_learned:FR3FinRayEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_estop_env_learned:FR3FinRayEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-EStop-Close-Long-Long-U.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-EStop-Close-f",
    entry_point=f"{__name__}.fr3_finray_estop_env_learned:FR3FinRayEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_estop_env_learned:FR3FinRayEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-EStop-Close-f.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-EStop-Close-l",
    entry_point=f"{__name__}.fr3_finray_estop_env_learned:FR3FinRayEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_estop_env_learned:FR3FinRayEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-EStop-Close-l.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-EStop-Close-c",
    entry_point=f"{__name__}.fr3_finray_estop_env_learned:FR3FinRayEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_estop_env_learned:FR3FinRayEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-EStop-Close-c.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-EStop-Close-lf",
    entry_point=f"{__name__}.fr3_finray_estop_env_learned:FR3FinRayEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_estop_env_learned:FR3FinRayEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-EStop-Close-lf.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-EStop-Close-ff",
    entry_point=f"{__name__}.fr3_finray_estop_env_learned:FR3FinRayEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_estop_env_learned:FR3FinRayEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-EStop-Close-ff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-EStop-Open",
    entry_point=f"{__name__}.fr3_finray_estop_closed_env_learned:FR3FinRayEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_estop_closed_env_learned:FR3FinRayEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-EStop-Open-Long-Long-U",
    entry_point=f"{__name__}.fr3_finray_estop_closed_env_learned:FR3FinRayEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_estop_closed_env_learned:FR3FinRayEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-EStop-Open-Long-Long-U.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-EStop-Open-f",
    entry_point=f"{__name__}.fr3_finray_estop_closed_env_learned:FR3FinRayEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_estop_closed_env_learned:FR3FinRayEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-EStop-Open-f.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-EStop-Open-l",
    entry_point=f"{__name__}.fr3_finray_estop_closed_env_learned:FR3FinRayEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_estop_closed_env_learned:FR3FinRayEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-EStop-Open-l.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-EStop-Open-c",
    entry_point=f"{__name__}.fr3_finray_estop_closed_env_learned:FR3FinRayEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_estop_closed_env_learned:FR3FinRayEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-EStop-Open-c.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-EStop-Open-lf",
    entry_point=f"{__name__}.fr3_finray_estop_closed_env_learned:FR3FinRayEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_estop_closed_env_learned:FR3FinRayEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-EStop-Open-lf.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-EStop-Open-ff",
    entry_point=f"{__name__}.fr3_finray_estop_closed_env_learned:FR3FinRayEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_estop_closed_env_learned:FR3FinRayEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-EStop-Open-ff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-EStop-Open-fff",
    entry_point=f"{__name__}.fr3_finray_estop_closed_env_learned:FR3FinRayEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_estop_closed_env_learned:FR3FinRayEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-EStop-Open-fff.yaml",
    },
)


gym.register(
    id="FR3-FinRay-Table-Stack-2",
    entry_point=f"{__name__}.fr3_finray_stack2_env_learned:FR3FinRayStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack2_env_learned:FR3FinRayStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-3",
    entry_point=f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "bc_mlp.json"),
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-2-Long-Long-U",
    entry_point=f"{__name__}.fr3_finray_stack2_env_learned:FR3FinRayStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack2_env_learned:FR3FinRayStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-2-Long-Long-U.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-2-f",
    entry_point=f"{__name__}.fr3_finray_stack2_env_learned:FR3FinRayStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack2_env_learned:FR3FinRayStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-2-f.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-2-l",
    entry_point=f"{__name__}.fr3_finray_stack2_env_learned:FR3FinRayStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack2_env_learned:FR3FinRayStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-2-l.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-2-c",
    entry_point=f"{__name__}.fr3_finray_stack2_env_learned:FR3FinRayStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack2_env_learned:FR3FinRayStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-2-c.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-2-lf",
    entry_point=f"{__name__}.fr3_finray_stack2_env_learned:FR3FinRayStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack2_env_learned:FR3FinRayStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-2-lf.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-2-ff",
    entry_point=f"{__name__}.fr3_finray_stack2_env_learned:FR3FinRayStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack2_env_learned:FR3FinRayStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-2-ff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-3-Long-Long-U",
    entry_point=f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-3-Long-Long-U.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-3-f",
    entry_point=f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-3-f.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-3-l",
    entry_point=f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-3-l.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-3-c",
    entry_point=f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-3-c.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-3-lf",
    entry_point=f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-3-lf.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-3-ff",
    entry_point=f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-3-ff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-3-fff3",
    entry_point=f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-3-fff3.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-3-fff",
    entry_point=f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-3-fff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-3-ffff3",
    entry_point=f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-3-ffff3.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-3-ffff",
    entry_point=f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-3-ffff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-3-fffff3",
    entry_point=f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-3-fffff3.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-3-fffff",
    entry_point=f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-3-fffff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-3-ffffff3",
    entry_point=f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-3-ffffff3.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Stack-3-ffffff",
    entry_point=f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_stack3_env_learned:FR3FinRayStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Stack-3-ffffff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Drawer-Long-Long-U",
    entry_point=f"{__name__}.fr3_finray_drawer_env_learned:FR3FinRayDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_drawer_env_learned:FR3FinRayDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Drawer-Long-Long-U.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Drawer-f",
    entry_point=f"{__name__}.fr3_finray_drawer_env_learned:FR3FinRayDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_drawer_env_learned:FR3FinRayDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Drawer-f.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Drawer-l",
    entry_point=f"{__name__}.fr3_finray_drawer_env_learned:FR3FinRayDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_drawer_env_learned:FR3FinRayDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Drawer-l.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Drawer-c",
    entry_point=f"{__name__}.fr3_finray_drawer_env_learned:FR3FinRayDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_drawer_env_learned:FR3FinRayDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Drawer-c.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Drawer-lf",
    entry_point=f"{__name__}.fr3_finray_drawer_env_learned:FR3FinRayDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_drawer_env_learned:FR3FinRayDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Drawer-lf.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Drawer-ff",
    entry_point=f"{__name__}.fr3_finray_drawer_env_learned:FR3FinRayDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_drawer_env_learned:FR3FinRayDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Drawer-ff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Hang-Cup-Long-Long-U",
    entry_point=f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Hang-Cup-Long-Long-U.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Hang-Cup-f",
    entry_point=f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Hang-Cup-f.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Hang-Cup-l",
    entry_point=f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Hang-Cup-l.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Hang-Cup-c",
    entry_point=f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Hang-Cup-c.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Hang-Cup-lf",
    entry_point=f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Hang-Cup-lf.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Hang-Cup-ff",
    entry_point=f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Hang-Cup-ff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Hang-Cup-fff",
    entry_point=f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Hang-Cup-fff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Hang-Cup-fff3",
    entry_point=f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Hang-Cup-fff3.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Hang-Cup-ffff",
    entry_point=f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Hang-Cup-ffff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Hang-Cup-ffff3",
    entry_point=f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Hang-Cup-ffff3.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Hang-Cup-fffff",
    entry_point=f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Hang-Cup-fffff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Hang-Cup-ffffff",
    entry_point=f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_hang_cup_env_learned:FR3FinRayHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Hang-Cup-ffffff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Can-Long-Long-U",
    entry_point=f"{__name__}.fr3_finray_can_env_learned:FR3FinRayCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_can_env_learned:FR3FinRayCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Can-Long-Long-U.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Can-f",
    entry_point=f"{__name__}.fr3_finray_can_env_learned:FR3FinRayCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_can_env_learned:FR3FinRayCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Can-f.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Can-l",
    entry_point=f"{__name__}.fr3_finray_can_env_learned:FR3FinRayCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_can_env_learned:FR3FinRayCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Can-l.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Can-c",
    entry_point=f"{__name__}.fr3_finray_can_env_learned:FR3FinRayCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_can_env_learned:FR3FinRayCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Can-c.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Can-lf",
    entry_point=f"{__name__}.fr3_finray_can_env_learned:FR3FinRayCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_can_env_learned:FR3FinRayCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Can-lf.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Can-ff",
    entry_point=f"{__name__}.fr3_finray_can_env_learned:FR3FinRayCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_can_env_learned:FR3FinRayCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Can-ff.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Pen-Cup-Long-Long-U",
    entry_point=f"{__name__}.fr3_finray_pen_cup_env_learned:FR3FinRayPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_pen_cup_env_learned:FR3FinRayPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Pen-Cup-Long-Long-U.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_fr3_finray_drawer.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Pen-Cup-f",
    entry_point=f"{__name__}.fr3_finray_pen_cup_env_learned:FR3FinRayPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_pen_cup_env_learned:FR3FinRayPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Pen-Cup-f.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_fr3_finray_drawer.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Pen-Cup-l",
    entry_point=f"{__name__}.fr3_finray_pen_cup_env_learned:FR3FinRayPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_pen_cup_env_learned:FR3FinRayPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Pen-Cup-l.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_fr3_finray_drawer.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Pen-Cup-c",
    entry_point=f"{__name__}.fr3_finray_pen_cup_env_learned:FR3FinRayPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_pen_cup_env_learned:FR3FinRayPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Pen-Cup-c.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_fr3_finray_drawer.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Pen-Cup-lf",
    entry_point=f"{__name__}.fr3_finray_pen_cup_env_learned:FR3FinRayPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_pen_cup_env_learned:FR3FinRayPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Pen-Cup-lf.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_fr3_finray_drawer.yaml",
    },
)

gym.register(
    id="FR3-FinRay-Table-Pen-Cup-ff",
    entry_point=f"{__name__}.fr3_finray_pen_cup_env_learned:FR3FinRayPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_finray_pen_cup_env_learned:FR3FinRayPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:FR3-FinRay-Table-Pen-Cup-ff.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_fr3_finray_drawer.yaml",
    },
)
