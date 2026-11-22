# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Kinova-Cabinet environment.
"""

import gymnasium as gym
import os
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Kinova-Table-Gen3-Drawer",
    entry_point=f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Cabinet-Door",
    entry_point=f"{__name__}.kinova_cabinet_door_env_learned:KinovaCabinetDoorEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_cabinet_door_env_learned:KinovaCabinetDoorEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Cabinet-Door-f",
    entry_point=f"{__name__}.kinova_cabinet_door_env_learned:KinovaCabinetDoorEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_cabinet_door_env_learned:KinovaCabinetDoorEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Cabinet-Door-f.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-Stack",
    entry_point=f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-Stack-f",
    entry_point=f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Drawer-Stack-f.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-Stack-f3",
    entry_point=f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Drawer-Stack-f3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-Stack-ff",
    entry_point=f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Drawer-Stack-ff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-Stack-ff3",
    entry_point=f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Drawer-Stack-ff3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-Stack-fff",
    entry_point=f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Drawer-Stack-fff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-Stack-fff3",
    entry_point=f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Drawer-Stack-fff3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-Stack-ffff",
    entry_point=f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Drawer-Stack-ffff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-Stack-ffff3",
    entry_point=f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Drawer-Stack-ffff3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-Stack-fffff",
    entry_point=f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Drawer-Stack-fffff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-Stack-ffffff",
    entry_point=f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_stack_env_learned:KinovaDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Drawer-Stack-ffffff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Pen-Cup",
    entry_point=f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_drawer.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Hang-Cup",
    entry_point=f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Can",
    entry_point=f"{__name__}.kinova_can_env_learned:KinovaCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_can_env_learned:KinovaCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)


gym.register(
    id="Kinova-Table-Gen3-EStop-Close",
    entry_point=f"{__name__}.kinova_estop_env_learned:KinovaEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_estop_env_learned:KinovaEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-EStop-Close-f",
    entry_point=f"{__name__}.kinova_estop_env_learned:KinovaEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_estop_env_learned:KinovaEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-EStop-Close-f.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-EStop-Close-ff",
    entry_point=f"{__name__}.kinova_estop_env_learned:KinovaEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_estop_env_learned:KinovaEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-EStop-Close-ff.yaml",
    },
)


gym.register(
    id="Kinova-Table-Gen3-EStop-Close-fff",
    entry_point=f"{__name__}.kinova_estop_env_learned:KinovaEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_estop_env_learned:KinovaEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-EStop-Close-fff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-EStop-Close-f3",
    entry_point=f"{__name__}.kinova_estop_env_learned:KinovaEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_estop_env_learned:KinovaEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-EStop-Close-f3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-EStop-Close-ff3",
    entry_point=f"{__name__}.kinova_estop_env_learned:KinovaEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_estop_env_learned:KinovaEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-EStop-Close-ff3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-EStop-Close-fff3",
    entry_point=f"{__name__}.kinova_estop_env_learned:KinovaEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_estop_env_learned:KinovaEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-EStop-Close-fff3.yaml",
    },
)


gym.register(
    id="Kinova-Table-Gen3-EStop-Open",
    entry_point=f"{__name__}.kinova_estop_closed_env_learned:KinovaEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_estop_closed_env_learned:KinovaEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-EStop-Open-f",
    entry_point=f"{__name__}.kinova_estop_closed_env_learned:KinovaEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_estop_closed_env_learned:KinovaEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-EStop-Open-f.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-EStop-Open-ff",
    entry_point=f"{__name__}.kinova_estop_closed_env_learned:KinovaEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_estop_closed_env_learned:KinovaEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-EStop-Open-ff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-EStop-Open-fff",
    entry_point=f"{__name__}.kinova_estop_closed_env_learned:KinovaEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_estop_closed_env_learned:KinovaEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-EStop-Open-fff.yaml",
    },
)


gym.register(
    id="Kinova-Table-Gen3-EStop-Open-f3",
    entry_point=f"{__name__}.kinova_estop_closed_env_learned:KinovaEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_estop_closed_env_learned:KinovaEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-EStop-Open-f3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-EStop-Open-ff3",
    entry_point=f"{__name__}.kinova_estop_closed_env_learned:KinovaEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_estop_closed_env_learned:KinovaEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-EStop-Open-ff3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-EStop-Open-fff3",
    entry_point=f"{__name__}.kinova_estop_closed_env_learned:KinovaEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_estop_closed_env_learned:KinovaEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-EStop-Open-fff3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-2",
    entry_point=f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-3",
    entry_point=f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "bc_mlp.json"),
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-2-Long-Long-U",
    entry_point=f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-2-Long-Long-U.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-2-f",
    entry_point=f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-2-f.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-2-l",
    entry_point=f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-2-l.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-2-c",
    entry_point=f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-2-c.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-2-lf",
    entry_point=f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-2-lf.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-2-ff",
    entry_point=f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-2-ff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-2-fff",
    entry_point=f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-2-fff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-2-f3",
    entry_point=f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-2-f3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-2-ff3",
    entry_point=f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-2-ff3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-2-fff3",
    entry_point=f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack2_env_learned:KinovaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-2-fff3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-3-Long-Long-U",
    entry_point=f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-3-Long-Long-U.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-3-f",
    entry_point=f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-3-f.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-3-f3",
    entry_point=f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-3-f3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-3-l",
    entry_point=f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-3-l.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-3-c",
    entry_point=f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-3-c.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-3-lf",
    entry_point=f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-3-lf.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-3-ff",
    entry_point=f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-3-ff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-3-ff3",
    entry_point=f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-3-ff3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-3-fff3",
    entry_point=f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-3-fff3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-3-fff",
    entry_point=f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-3-fff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-3-ffff3",
    entry_point=f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-3-ffff3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-3-ffff",
    entry_point=f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-3-ffff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-3-fffff3",
    entry_point=f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-3-fffff3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-3-fffff",
    entry_point=f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-3-fffff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-3-ffffff3",
    entry_point=f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-3-ffffff3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Stack-3-ffffff",
    entry_point=f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_stack3_env_learned:KinovaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Stack-3-ffffff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-Long-Long-U",
    entry_point=f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Drawer-Long-Long-U.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-f",
    entry_point=f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Drawer-f.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-l",
    entry_point=f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Drawer-l.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-c",
    entry_point=f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Drawer-c.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-lf",
    entry_point=f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Drawer-lf.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-ff",
    entry_point=f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Drawer-ff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-ff3",
    entry_point=f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Drawer-ff3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-fff",
    entry_point=f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Drawer-fff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Drawer-fff3",
    entry_point=f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_drawer_env_learned:KinovaDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Drawer-fff3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Hang-Cup-Long-Long-U",
    entry_point=f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Hang-Cup-Long-Long-U.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Hang-Cup-f",
    entry_point=f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Hang-Cup-f.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Hang-Cup-f3",
    entry_point=f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Hang-Cup-f3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Hang-Cup-l",
    entry_point=f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Hang-Cup-l.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Hang-Cup-c",
    entry_point=f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Hang-Cup-c.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Hang-Cup-lf",
    entry_point=f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Hang-Cup-lf.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Hang-Cup-ff",
    entry_point=f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Hang-Cup-ff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Hang-Cup-ff3",
    entry_point=f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Hang-Cup-ff3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Hang-Cup-fff",
    entry_point=f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Hang-Cup-fff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Hang-Cup-fff3",
    entry_point=f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Hang-Cup-fff3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Hang-Cup-ffff",
    entry_point=f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Hang-Cup-ffff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Hang-Cup-ffff3",
    entry_point=f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Hang-Cup-ffff3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Hang-Cup-fffff",
    entry_point=f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Hang-Cup-fffff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Hang-Cup-ffffff",
    entry_point=f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_hang_cup_env_learned:KinovaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Hang-Cup-ffffff.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Can-Long-Long-U",
    entry_point=f"{__name__}.kinova_can_env_learned:KinovaCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_can_env_learned:KinovaCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Can-Long-Long-U.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Can-f",
    entry_point=f"{__name__}.kinova_can_env_learned:KinovaCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_can_env_learned:KinovaCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Can-f.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Can-l",
    entry_point=f"{__name__}.kinova_can_env_learned:KinovaCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_can_env_learned:KinovaCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Can-l.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Can-c",
    entry_point=f"{__name__}.kinova_can_env_learned:KinovaCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_can_env_learned:KinovaCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Can-c.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Can-lf",
    entry_point=f"{__name__}.kinova_can_env_learned:KinovaCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_can_env_learned:KinovaCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Can-lf.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Can-ff",
    entry_point=f"{__name__}.kinova_can_env_learned:KinovaCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_can_env_learned:KinovaCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Can-ff.yaml",
    },
)


gym.register(
    id="Kinova-Table-Gen3-Can-fff3",
    entry_point=f"{__name__}.kinova_can_env_learned:KinovaCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_can_env_learned:KinovaCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Can-fff3.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Pen-Cup-Long-Long-U",
    entry_point=f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Pen-Cup-Long-Long-U.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_drawer.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Pen-Cup-f",
    entry_point=f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Pen-Cup-f.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_drawer.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Pen-Cup-l",
    entry_point=f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Pen-Cup-l.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_drawer.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Pen-Cup-c",
    entry_point=f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Pen-Cup-c.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_drawer.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Pen-Cup-lf",
    entry_point=f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Pen-Cup-lf.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_drawer.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Pen-Cup-ff",
    entry_point=f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Pen-Cup-ff.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_drawer.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Pen-Cup-fff",
    entry_point=f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Pen-Cup-fff.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_drawer.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Pen-Cup-f3",
    entry_point=f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Pen-Cup-f3.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_drawer.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Pen-Cup-ff3",
    entry_point=f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Pen-Cup-ff3.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_drawer.yaml",
    },
)

gym.register(
    id="Kinova-Table-Gen3-Pen-Cup-fff3",
    entry_point=f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_pen_cup_env_learned:KinovaPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Kinova-Table-Gen3-Pen-Cup-fff3.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_drawer.yaml",
    },
)
