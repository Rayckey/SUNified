# SPDX-License-Identifier: Apache-2.0
"""UR5 registration and object-free recorder contracts, without a simulator."""

import json
from types import SimpleNamespace

import gymnasium as gym
import h5py
import numpy as np
import torch

from kuafu.isaac.learned.ur5 import TASKS
from kuafu.isaac.learned.ur5.assets import JOINT_NAMES
from kuafu.data.collection.rgb_recorder import RGBTransitionRecorder
from kuafu.rl.isaac.task_registry import infer_robot_id_from_task, normalize_robot_id
from kuafu.paths import PROJECT_ROOT
from kuafu.pipeline.task_catalog import get_task, list_tasks


def test_ur5_tasks_resolve_to_their_own_embodiment():
    assert normalize_robot_id("ur5") == "ur5_robotiq"
    for task in TASKS:
        task_id = f"UR5-Table-{task}"
        assert infer_robot_id_from_task(task_id) == "ur5_robotiq"
        assert ".ur5.tasks:" in gym.spec(task_id).entry_point


def test_all_nine_published_tasks_select_bundled_programs():
    tasks = [get_task(row["id"], "ur5_robotiq") for row in list_tasks()
             if row["status"] == "registered"]
    assert len(tasks) == 9
    assert {task["environments"][0]["gym_id"] for task in tasks} == {
        f"UR5-Table-{name}" for name in TASKS
    }
    for task in tasks:
        program = task["example_program"]
        assert program is not None, task["id"]
        stages = list((PROJECT_ROOT / program["path"] / "program").glob("subtask_*"))
        assert len(stages) == program["stages"] > 0
        assert all((stage / "constraints.json").is_file() for stage in stages)


def test_ur5_recording_needs_only_joints_and_rgb(tmp_path):
    state = torch.zeros(1, 8)
    pixels = torch.zeros(1, 72, 128, 3, dtype=torch.uint8)
    camera = SimpleNamespace(data=SimpleNamespace(output={"rgb": pixels}), update=lambda *a, **kw: None)
    env = SimpleNamespace(
        num_envs=1, obs_buf={"policy": state, "robot_state": state},
        _actuated_joint_names=JOINT_NAMES, _logged_relative_joint=torch.zeros(1, 8),
        reward_buf=torch.zeros(1), _cached_success=torch.tensor([True]),
        scene=SimpleNamespace(sensors={"table_cam_0": camera, "table_cam_1": camera}),
        sim=SimpleNamespace(render=lambda: None),
    )
    path = tmp_path / "robot_rgb.hdf5"
    recorder = RGBTransitionRecorder(env, path, 1, {"env_name": "UR5-Table-Drawer", "control_dt": 1 / 30})
    for index in range(3):
        pixels.fill_(index)
        recorder.record_pre_step()
        state.add_(1)
        recorder.record_post_step()
    recorder.record_pre_reset(torch.tensor([0]))
    recorder.close()
    with h5py.File(path) as source:
        metadata = json.loads(source["data"].attrs["env_args"])
        assert metadata["robot_state_dim"] == 8
        assert metadata["robot_state_joint_names"] == list(JOINT_NAMES)
        demo = source["data/demo_0"]
        assert "full_state" not in demo["obs"]
        assert demo["obs/rgb"].shape == (3, 2, 72, 128, 3)
        np.testing.assert_array_equal(demo["obs/robot_state"][:, 0], [0, 1, 2])
        np.testing.assert_array_equal(demo["next_obs/robot_state"][:, 0], [1, 2, 3])
