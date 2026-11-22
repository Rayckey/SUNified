"""State recording must preserve transition timing across asynchronous resets."""

from types import SimpleNamespace

import h5py
import numpy as np
import torch

from kuafu.data.collection.state_recorder import StateTransitionRecorder


def test_success_budget_and_autoreset_do_not_contaminate_transitions(tmp_path):
    env = SimpleNamespace(
        num_envs=3,
        obs_buf={"full_state": torch.tensor([[0.], [10.], [20.]]),
                 "policy": torch.tensor([[0.], [10.], [20.]])},
        _logged_relative_joint=torch.ones(3, 9),
        _logged_joint_targets=torch.full((3, 9), 2.),
        reward_buf=torch.zeros(3),
        _cached_success=torch.tensor([True, False, True]),
    )
    path = tmp_path / "demos.hdf5"
    recorder = StateTransitionRecorder(env, path, target=1)
    for step in range(2):
        recorder.record_pre_step()
        # In-place mutation must not alter the saved pre-action observation/action.
        env.obs_buf["full_state"].add_(1)
        env.obs_buf["policy"].add_(1)
        env._logged_relative_joint.add_(1)
        env.reward_buf.fill_(step + 1)
        recorder.record_post_step()
    recorder.record_pre_step()
    recorder.record_pre_reset(torch.tensor([0, 1, 2]))
    env.obs_buf["full_state"].fill_(-1000)  # autoreset observation, not a successor
    recorder.record_post_reset(torch.tensor([0, 1, 2]))
    recorder.record_post_step()
    assert recorder.count == 1
    assert recorder.attempts == 3
    assert all(not rows for rows in recorder.episodes)
    recorder.close()
    with h5py.File(path) as data:
        assert list(data["data"]) == ["demo_0"]
        episode = data["data/demo_0"]
        np.testing.assert_array_equal(episode["obs/full_state"][:, 0], [0, 1])
        np.testing.assert_array_equal(episode["next_obs/full_state"][:, 0], [1, 2])
        np.testing.assert_array_equal(episode["actions/relative_joint"][:, 0], [1, 2])
        np.testing.assert_array_equal(episode["rewards"], [1, 2])
        assert episode.attrs["success"]
        assert episode.attrs["terminal_transition_omitted"]
        assert data["data"].attrs["total"] == 2


def test_mpc_records_partial_resets_failures_and_scene_states(tmp_path):
    positions = torch.tensor([[0.], [10.]])
    env = SimpleNamespace(
        num_envs=2, obs_buf={"full_state": positions, "policy": positions},
        _logged_relative_joint=torch.ones(2, 9),
        reward_buf=torch.ones(2), _cached_success=torch.tensor([True, False]),
        scene=SimpleNamespace(get_state=lambda **kwargs: {
            "articulation": {"robot": {"joint_position": positions}},
        }),
    )
    path = tmp_path / "mpc.hdf5"
    recorder = StateTransitionRecorder(env, path, 3, record_failures=True, record_scene=True)
    recorder.record_pre_step()
    positions.add_(1)
    recorder.record_post_step()
    recorder.record_pre_step()
    recorder.record_pre_reset(torch.tensor([0]))
    recorder.record_pre_reset(torch.tensor([0]))  # Native task and base reset hooks both call it.
    positions[0] = 100
    positions[1] += 1
    recorder.record_post_step()
    recorder.record_pre_step()
    positions.add_(1)
    recorder.record_post_step()
    recorder.record_pre_reset(torch.tensor([1]))
    recorder.record_pre_reset(torch.tensor([0]))
    assert recorder.count == recorder.attempts == 3
    assert recorder.exported_successful_episode_count == 2
    assert recorder.exported_failed_episode_count == 1
    recorder.close()
    with h5py.File(path) as file:
        assert file["data"].attrs["total"] == 5
        for index, (start, length, success) in enumerate(((0, 1, True), (10, 3, False), (100, 1, True))):
            group = file[f"data/demo_{index}"]
            assert group.attrs["num_samples"] == length
            assert group.attrs["success"] == success
            assert len(group["actions/relative_joint"]) == length
            np.testing.assert_array_equal(group["obs/state"][:, 0], np.arange(start, start + length))
            np.testing.assert_array_equal(group["next_obs/state"][:, 0], np.arange(start + 1, start + length + 1))
            np.testing.assert_array_equal(group["states/articulation/robot/joint_position"], group["next_obs/state"])
            assert group["initial_state/articulation/robot/joint_position"][0, 0] == start


def test_suppressed_reset_discards_partial_episode(tmp_path):
    env = SimpleNamespace(
        num_envs=1, obs_buf={"full_state": torch.zeros(1, 1)},
        _logged_relative_joint=torch.ones(1, 9), reward_buf=torch.ones(1),
        _cached_success=torch.tensor([True]),
    )
    recorder = StateTransitionRecorder(env, tmp_path / "demos.hdf5", 1)
    recorder.record_pre_step()
    recorder.record_post_step()
    recorder.cfg.export_in_record_pre_reset = False
    recorder.record_pre_reset(None)
    recorder.record_post_step()
    assert recorder.count == recorder.attempts == 0
    assert recorder.pending == [None]
    assert recorder.episodes == [[]]
    recorder.close()
