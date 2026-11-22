# SPDX-License-Identifier: Apache-2.0
"""UR5 + Robotiq 2F-85 tabletop task registrations (safe before Isaac starts)."""

import gymnasium as gym

TASKS = {
    "Drawer": "Drawer",
    "Drawer-Stack": "DrawerStack",
    "Stack-2": "Stack2",
    "Stack-3": "Stack3",
    "Hang-Cup": "HangCup",
    "Can": "Can",
    "Pen-Cup": "PenCup",
    "EStop": "EStop",
    "EStop-Closed": "EStopClosed",
}

for _task, _class in TASKS.items():
    gym.register(
        id=f"UR5-Table-{_task}",
        entry_point=f"{__name__}.tasks:UR5{_class}Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks:UR5{_class}EnvCfg",
            "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
        },
    )
