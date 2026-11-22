# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka Panda environments.
"""

import gymnasium as gym
import os
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Panda-Table-Drawer",
    entry_point=f"{__name__}.panda_drawer_env_learned:PandaDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_drawer_env_learned:PandaDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Panda-Table-Cabinet-Door",
    entry_point=f"{__name__}.panda_cabinet_door_env_learned:PandaCabinetDoorEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_cabinet_door_env_learned:PandaCabinetDoorEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Panda-Table-Cabinet-Door-f",
    entry_point=f"{__name__}.panda_cabinet_door_env_learned:PandaCabinetDoorEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_cabinet_door_env_learned:PandaCabinetDoorEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Cabinet-Door-f.yaml",
    },
)

gym.register(
    id="Panda-Table-Drawer-Stack",
    entry_point=f"{__name__}.panda_drawer_stack_env_learned:PandaDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_drawer_stack_env_learned:PandaDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Panda-Table-Drawer-Stack-ff",
    entry_point=f"{__name__}.panda_drawer_stack_env_learned:PandaDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_drawer_stack_env_learned:PandaDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Drawer-Stack-ff.yaml",
    },
)

gym.register(
    id="Panda-Table-Drawer-Stack-fff",
    entry_point=f"{__name__}.panda_drawer_stack_env_learned:PandaDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_drawer_stack_env_learned:PandaDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Drawer-Stack-fff.yaml",
    },
)

gym.register(
    id="Panda-Table-Drawer-Stack-ffff",
    entry_point=f"{__name__}.panda_drawer_stack_env_learned:PandaDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_drawer_stack_env_learned:PandaDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Drawer-Stack-ffff.yaml",
    },
)

gym.register(
    id="Panda-Table-Drawer-Stack-ffff3",
    entry_point=f"{__name__}.panda_drawer_stack_env_learned:PandaDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_drawer_stack_env_learned:PandaDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Drawer-Stack-ffff3.yaml",
    },
)

gym.register(
    id="Panda-Table-Drawer-Stack-fffff",
    entry_point=f"{__name__}.panda_drawer_stack_env_learned:PandaDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_drawer_stack_env_learned:PandaDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Drawer-Stack-fffff.yaml",
    },
)

gym.register(
    id="Panda-Table-Drawer-Stack-ffffff",
    entry_point=f"{__name__}.panda_drawer_stack_env_learned:PandaDrawerStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_drawer_stack_env_learned:PandaDrawerStackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Drawer-Stack-ffffff.yaml",
    },
)

gym.register(
    id="Panda-Table-Pen-Cup",
    entry_point=f"{__name__}.panda_pen_cup_env_learned:PandaPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_pen_cup_env_learned:PandaPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_drawer.yaml",
    },
)

gym.register(
    id="Panda-Table-Hang-Cup",
    entry_point=f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Panda-Table-Can",
    entry_point=f"{__name__}.panda_can_env_learned:PandaCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_can_env_learned:PandaCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)


gym.register(
    id="Panda-Table-EStop-Close",
    entry_point=f"{__name__}.panda_estop_env_learned:PandaEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_estop_env_learned:PandaEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Panda-Table-EStop-Close-Long-Long-U",
    entry_point=f"{__name__}.panda_estop_env_learned:PandaEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_estop_env_learned:PandaEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-EStop-Close-Long-Long-U.yaml",
    },
)

gym.register(
    id="Panda-Table-EStop-Close-f",
    entry_point=f"{__name__}.panda_estop_env_learned:PandaEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_estop_env_learned:PandaEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-EStop-Close-f.yaml",
    },
)

gym.register(
    id="Panda-Table-EStop-Close-l",
    entry_point=f"{__name__}.panda_estop_env_learned:PandaEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_estop_env_learned:PandaEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-EStop-Close-l.yaml",
    },
)

gym.register(
    id="Panda-Table-EStop-Close-c",
    entry_point=f"{__name__}.panda_estop_env_learned:PandaEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_estop_env_learned:PandaEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-EStop-Close-c.yaml",
    },
)

gym.register(
    id="Panda-Table-EStop-Close-lf",
    entry_point=f"{__name__}.panda_estop_env_learned:PandaEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_estop_env_learned:PandaEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-EStop-Close-lf.yaml",
    },
)

gym.register(
    id="Panda-Table-EStop-Close-ff",
    entry_point=f"{__name__}.panda_estop_env_learned:PandaEStopEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_estop_env_learned:PandaEStopEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-EStop-Close-ff.yaml",
    },
)

gym.register(
    id="Panda-Table-EStop-Open",
    entry_point=f"{__name__}.panda_estop_closed_env_learned:PandaEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_estop_closed_env_learned:PandaEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Panda-Table-EStop-Open-Long-Long-U",
    entry_point=f"{__name__}.panda_estop_closed_env_learned:PandaEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_estop_closed_env_learned:PandaEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-EStop-Open-Long-Long-U.yaml",
    },
)

gym.register(
    id="Panda-Table-EStop-Open-f",
    entry_point=f"{__name__}.panda_estop_closed_env_learned:PandaEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_estop_closed_env_learned:PandaEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-EStop-Open-f.yaml",
    },
)

gym.register(
    id="Panda-Table-EStop-Open-l",
    entry_point=f"{__name__}.panda_estop_closed_env_learned:PandaEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_estop_closed_env_learned:PandaEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-EStop-Open-l.yaml",
    },
)

gym.register(
    id="Panda-Table-EStop-Open-c",
    entry_point=f"{__name__}.panda_estop_closed_env_learned:PandaEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_estop_closed_env_learned:PandaEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-EStop-Open-c.yaml",
    },
)

gym.register(
    id="Panda-Table-EStop-Open-lf",
    entry_point=f"{__name__}.panda_estop_closed_env_learned:PandaEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_estop_closed_env_learned:PandaEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-EStop-Open-lf.yaml",
    },
)

gym.register(
    id="Panda-Table-EStop-Open-ff",
    entry_point=f"{__name__}.panda_estop_closed_env_learned:PandaEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_estop_closed_env_learned:PandaEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-EStop-Open-ff.yaml",
    },
)

gym.register(
    id="Panda-Table-EStop-Open-fff",
    entry_point=f"{__name__}.panda_estop_closed_env_learned:PandaEStopClosedEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_estop_closed_env_learned:PandaEStopClosedEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-EStop-Open-fff.yaml",
    },
)


gym.register(
    id="Panda-Table-Stack-2",
    entry_point=f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-3",
    entry_point=f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack3_env_learned:PandaStack3MPCEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "bc_mlp.json"),
    },
)

gym.register(
    id="Panda-Table-Stack-2-Long-Long-U",
    entry_point=f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-2-Long-Long-U.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-2-f",
    entry_point=f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-2-f.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-2-l",
    entry_point=f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-2-l.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-2-c",
    entry_point=f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-2-c.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-2-lf",
    entry_point=f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-2-lf.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-2-ff",
    entry_point=f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-2-ff.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-2-fff",
    entry_point=f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-2-fff.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-2-f3",
    entry_point=f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-2-f3.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-2-ff3",
    entry_point=f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-2-ff3.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-2-fff3",
    entry_point=f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack2_env_learned:PandaStack2EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-2-fff3.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-3-Long-Long-U",
    entry_point=f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-3-Long-Long-U.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-3-f",
    entry_point=f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-3-f.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-3-l",
    entry_point=f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-3-l.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-3-c",
    entry_point=f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-3-c.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-3-lf",
    entry_point=f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-3-lf.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-3-ff",
    entry_point=f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-3-ff.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-3-fff3",
    entry_point=f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-3-fff3.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-3-fff",
    entry_point=f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-3-fff.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-3-ffff3",
    entry_point=f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-3-ffff3.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-3-ffff",
    entry_point=f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-3-ffff.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-3-fffff3",
    entry_point=f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-3-fffff3.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-3-fffff",
    entry_point=f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-3-fffff.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-3-ffffff3",
    entry_point=f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-3-ffffff3.yaml",
    },
)

gym.register(
    id="Panda-Table-Stack-3-ffffff",
    entry_point=f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_stack3_env_learned:PandaStack3EnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Stack-3-ffffff.yaml",
    },
)

gym.register(
    id="Panda-Table-Drawer-Long-Long-U",
    entry_point=f"{__name__}.panda_drawer_env_learned:PandaDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_drawer_env_learned:PandaDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Drawer-Long-Long-U.yaml",
    },
)

gym.register(
    id="Panda-Table-Drawer-f",
    entry_point=f"{__name__}.panda_drawer_env_learned:PandaDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_drawer_env_learned:PandaDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Drawer-f.yaml",
    },
)

gym.register(
    id="Panda-Table-Drawer-l",
    entry_point=f"{__name__}.panda_drawer_env_learned:PandaDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_drawer_env_learned:PandaDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Drawer-l.yaml",
    },
)

gym.register(
    id="Panda-Table-Drawer-c",
    entry_point=f"{__name__}.panda_drawer_env_learned:PandaDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_drawer_env_learned:PandaDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Drawer-c.yaml",
    },
)

gym.register(
    id="Panda-Table-Drawer-lf",
    entry_point=f"{__name__}.panda_drawer_env_learned:PandaDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_drawer_env_learned:PandaDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Drawer-lf.yaml",
    },
)

gym.register(
    id="Panda-Table-Drawer-ff",
    entry_point=f"{__name__}.panda_drawer_env_learned:PandaDrawerEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_drawer_env_learned:PandaDrawerEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Drawer-ff.yaml",
    },
)

gym.register(
    id="Panda-Table-Hang-Cup-Long-Long-U",
    entry_point=f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Hang-Cup-Long-Long-U.yaml",
    },
)

gym.register(
    id="Panda-Table-Hang-Cup-f",
    entry_point=f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Hang-Cup-f.yaml",
    },
)

gym.register(
    id="Panda-Table-Hang-Cup-l",
    entry_point=f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Hang-Cup-l.yaml",
    },
)

gym.register(
    id="Panda-Table-Hang-Cup-c",
    entry_point=f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Hang-Cup-c.yaml",
    },
)

gym.register(
    id="Panda-Table-Hang-Cup-lf",
    entry_point=f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Hang-Cup-lf.yaml",
    },
)

gym.register(
    id="Panda-Table-Hang-Cup-ff",
    entry_point=f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Hang-Cup-ff.yaml",
    },
)

gym.register(
    id="Panda-Table-Hang-Cup-fff",
    entry_point=f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Hang-Cup-fff.yaml",
    },
)

gym.register(
    id="Panda-Table-Hang-Cup-fff3",
    entry_point=f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Hang-Cup-fff3.yaml",
    },
)

gym.register(
    id="Panda-Table-Hang-Cup-ffff",
    entry_point=f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Hang-Cup-ffff.yaml",
    },
)

gym.register(
    id="Panda-Table-Hang-Cup-ffff3",
    entry_point=f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Hang-Cup-ffff3.yaml",
    },
)

gym.register(
    id="Panda-Table-Hang-Cup-fffff",
    entry_point=f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Hang-Cup-fffff.yaml",
    },
)

gym.register(
    id="Panda-Table-Hang-Cup-ffffff",
    entry_point=f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_hang_cup_env_learned:PandaHangCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Hang-Cup-ffffff.yaml",
    },
)

gym.register(
    id="Panda-Table-Can-Long-Long-U",
    entry_point=f"{__name__}.panda_can_env_learned:PandaCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_can_env_learned:PandaCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Can-Long-Long-U.yaml",
    },
)

gym.register(
    id="Panda-Table-Can-f",
    entry_point=f"{__name__}.panda_can_env_learned:PandaCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_can_env_learned:PandaCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Can-f.yaml",
    },
)

gym.register(
    id="Panda-Table-Can-l",
    entry_point=f"{__name__}.panda_can_env_learned:PandaCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_can_env_learned:PandaCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Can-l.yaml",
    },
)

gym.register(
    id="Panda-Table-Can-c",
    entry_point=f"{__name__}.panda_can_env_learned:PandaCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_can_env_learned:PandaCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Can-c.yaml",
    },
)

gym.register(
    id="Panda-Table-Can-lf",
    entry_point=f"{__name__}.panda_can_env_learned:PandaCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_can_env_learned:PandaCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Can-lf.yaml",
    },
)

gym.register(
    id="Panda-Table-Can-ff",
    entry_point=f"{__name__}.panda_can_env_learned:PandaCanEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_can_env_learned:PandaCanEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Can-ff.yaml",
    },
)

gym.register(
    id="Panda-Table-Pen-Cup-Long-Long-U",
    entry_point=f"{__name__}.panda_pen_cup_env_learned:PandaPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_pen_cup_env_learned:PandaPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Pen-Cup-Long-Long-U.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_drawer.yaml",
    },
)

gym.register(
    id="Panda-Table-Pen-Cup-f",
    entry_point=f"{__name__}.panda_pen_cup_env_learned:PandaPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_pen_cup_env_learned:PandaPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Pen-Cup-f.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_drawer.yaml",
    },
)

gym.register(
    id="Panda-Table-Pen-Cup-l",
    entry_point=f"{__name__}.panda_pen_cup_env_learned:PandaPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_pen_cup_env_learned:PandaPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Pen-Cup-l.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_drawer.yaml",
    },
)

gym.register(
    id="Panda-Table-Pen-Cup-c",
    entry_point=f"{__name__}.panda_pen_cup_env_learned:PandaPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_pen_cup_env_learned:PandaPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Pen-Cup-c.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_drawer.yaml",
    },
)

gym.register(
    id="Panda-Table-Pen-Cup-lf",
    entry_point=f"{__name__}.panda_pen_cup_env_learned:PandaPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_pen_cup_env_learned:PandaPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Pen-Cup-lf.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_drawer.yaml",
    },
)

gym.register(
    id="Panda-Table-Pen-Cup-ff",
    entry_point=f"{__name__}.panda_pen_cup_env_learned:PandaPenCupEnvLearned",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.panda_pen_cup_env_learned:PandaPenCupEnvLearnedCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:Panda-Table-Pen-Cup-ff.yaml",
        "sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_drawer.yaml",
    },
)


_PANDA_VARIANT_TASKS = {
    "Panda-Table-Cabinet-Door": (
        "panda_cabinet_door_env_learned",
        "PandaCabinetDoorEnvLearned",
        "PandaCabinetDoorEnvLearnedCfg",
        {},
    ),
    "Panda-Table-Drawer-Stack": (
        "panda_drawer_stack_env_learned",
        "PandaDrawerStackEnv",
        "PandaDrawerStackEnvCfg",
        {},
    ),
    "Panda-Table-Drawer": (
        "panda_drawer_env_learned",
        "PandaDrawerEnvLearned",
        "PandaDrawerEnvLearnedCfg",
        {},
    ),
    "Panda-Table-EStop-Close": (
        "panda_estop_env_learned",
        "PandaEStopEnvLearned",
        "PandaEStopEnvLearnedCfg",
        {},
    ),
    "Panda-Table-EStop-Open": (
        "panda_estop_closed_env_learned",
        "PandaEStopClosedEnvLearned",
        "PandaEStopClosedEnvLearnedCfg",
        {},
    ),
    "Panda-Table-Hang-Cup": (
        "panda_hang_cup_env_learned",
        "PandaHangCupEnvLearned",
        "PandaHangCupEnvLearnedCfg",
        {},
    ),
    "Panda-Table-Can": (
        "panda_can_env_learned",
        "PandaCanEnvLearned",
        "PandaCanEnvLearnedCfg",
        {},
    ),
    "Panda-Table-Pen-Cup": (
        "panda_pen_cup_env_learned",
        "PandaPenCupEnvLearned",
        "PandaPenCupEnvLearnedCfg",
        {"sac_cfg_entry_point": f"{agents.__name__}:sac_rlpd_cfg_drawer.yaml"},
    ),
    "Panda-Table-Stack-2": (
        "panda_stack2_env_learned",
        "PandaStack2EnvLearned",
        "PandaStack2EnvLearnedCfg",
        {},
    ),
    "Panda-Table-Stack-3": (
        "panda_stack3_env_learned",
        "PandaStack3EnvLearned",
        "PandaStack3EnvLearnedCfg",
        {},
    ),
}


def _register_missing_panda_variant(env_id, module_name, env_cls, cfg_cls, extra_kwargs):
    if env_id in gym.registry:
        return

    kwargs = {
        "env_cfg_entry_point": f"{__name__}.{module_name}:{cfg_cls}",
        "rl_games_cfg_entry_point": f"{__name__}.agents:{env_id}.yaml",
    }
    kwargs.update(extra_kwargs)
    gym.register(
        id=env_id,
        entry_point=f"{__name__}.{module_name}:{env_cls}",
        disable_env_checker=True,
        kwargs=kwargs,
    )


_PANDA_VARIANT_PREFIXES = sorted(
    _PANDA_VARIANT_TASKS.items(),
    key=lambda item: len(item[0]),
    reverse=True,
)
for _filename in os.listdir(agents.__path__[0]):
    if not _filename.startswith("Panda-Table-") or not _filename.endswith(".yaml"):
        continue
    _env_id = _filename[:-5]
    for _task_id, (_module_name, _env_cls, _cfg_cls, _extra_kwargs) in _PANDA_VARIANT_PREFIXES:
        if _env_id.startswith(f"{_task_id}-"):
            _register_missing_panda_variant(_env_id, _module_name, _env_cls, _cfg_cls, _extra_kwargs)
            break
