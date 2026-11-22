"""Learning keeps the cabinet, program and pre-action state from its MPC teacher."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from kuafu.agents import ToolService, jobs
from kuafu.data.generated_drawer import OBSERVATION_LAYOUT, combine_demonstrations, policy_observation, write_demonstration
from kuafu.pipeline.drawer_policy import check_inputs, finish, replay_command


def snapshot():
    return {"robot_joints": [0.1, 0.2], "robot_joint_velocities": [0.3, 0.4], "cabinet_joints": [0.5],
            "frames": {"top_drawer_handle_frame": {"position": [1, 2, 3], "quaternion_wxyz": [1, 0, 0, 0]}}}


def contract():
    return {"observation_layout": OBSERVATION_LAYOUT, "joint_names": ["a", "b"],
            "cabinet_joint_names": ["drawer"], "asset_hashes": {"asset": "original"},
            "program_hashes": {"program": "original"}, "target_travel_m": 0.1}


def test_recorded_transitions_preserve_pre_action_state_and_episode_boundaries(tmp_path):
    h5py = pytest.importorskip("h5py")
    obs = policy_observation(snapshot(), 2)
    np.testing.assert_allclose(obs, [.1, .2, .3, .4, .5, 1, 2, 3, 1, 0, 0, 0, 2])
    episodes = []
    for seed in (42, 43):
        path = tmp_path / f"{seed}.hdf5"
        write_demonstration(path, [obs, obs + 1], [[.6, .7], [.8, .9]], [obs + 1, obs + 2],
                            {**contract(), "seed": seed})
        episodes.append(path)
    combined = tmp_path / "combined.hdf5"
    result = combine_demonstrations(episodes, combined)
    assert result["episodes"] == 2 and result["transitions"] == 4
    with h5py.File(combined) as handle:
        assert sorted(handle["data"]) == ["demo_0", "demo_1"]
        np.testing.assert_array_equal(handle["data/demo_0/obs/state"][0], obs)
        np.testing.assert_array_equal(handle["data/demo_0/next_obs/state"][0], obs + 1)
        assert handle["data/demo_1/dones"][:].tolist() == [False, True]
        assert json.loads(handle["data/demo_1"].attrs["provenance"])["seed"] == 43
    with h5py.File(episodes[1], "a") as handle:
        handle.attrs["metadata"] = json.dumps({**contract(), "program_hashes": {"program": "edited"}})
    with pytest.raises(ValueError, match="different scene or program"):
        combine_demonstrations(episodes, tmp_path / "changed.hdf5")


def test_nonfinite_observation_is_rejected():
    state = snapshot()
    state["robot_joint_velocities"][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        policy_observation(state, 0)


def verified_service(tmp_path):
    service = ToolService(tmp_path / "artifacts", isaac_python=sys.executable)
    asset = tmp_path / "asset"
    asset.mkdir()
    for name in ("asset_dynamic.usda", "asset_geometry.usda", "semantics.json"):
        (asset / name).write_text("{}")
    context = {"asset": str(asset), "asset_hashes": jobs.asset_hashes(asset), "task": "Open top drawer 10 cm",
               "target_travel_m": .1, "position": [.6, -.5, .2], "yaw_deg": 180}
    directory = service._new("jobs")
    stage = directory / "input/program/subtask_0"
    stage.mkdir(parents=True)
    for name, value in {"task.txt": "Open drawer", "atomic.txt": "['Pull']", "constraints.json": '{}',
                        "status.txt": "status: unverified"}.items():
        (stage / name).write_text(value)
    (directory / "output").mkdir()
    jobs.write_json(directory / "output/scene.json", {"asset_hashes": context["asset_hashes"]})
    jobs.write_json(directory / "output/verification.json", {"passed": True, "program_passed": True,
                    "solver_passed": True, "physical": {"passed": True}, "model_requests": 0})
    request = {"kind": "verify", "context": context, "program_hashes": jobs.tree_hashes(directory / "input/program")}
    jobs.write_json(directory / "request.json", request)
    jobs.write_json(directory / "status.json", jobs.finish(directory, request, 0, None))
    return service, directory, asset


def test_agent_collection_snapshots_verified_inputs_and_rejects_changed_program(tmp_path, monkeypatch):
    service, verified, asset = verified_service(tmp_path)
    monkeypatch.setattr(jobs, "launch", lambda directory, request: jobs.write_json(directory / "request.json", request))
    response = service.call("kuafu_collect_drawer", {"verification_job": verified.name, "episodes": 2})
    assert response["ok"], response
    directory = Path(response["result"]["directory"])
    request = jobs.read_json(directory / "request.json")
    assert request["kind"] == "drawer-collect"
    check_inputs(directory, request)
    (asset / "semantics.json").write_text("edited original")
    check_inputs(directory, request)
    (directory / "input/program/subtask_0/task.txt").write_text("edited copy")
    with pytest.raises(ValueError, match="inputs changed"):
        check_inputs(directory, request)
    (verified / "accepted/program/subtask_0/task.txt").write_text("unverified replacement")
    rejected = service.call("kuafu_collect_drawer", {"verification_job": verified.name})
    assert not rejected["ok"] and "changed" in rejected["error"]["message"]


def test_learning_exit_code_cannot_replace_artifacts(tmp_path, monkeypatch):
    service, verified, _ = verified_service(tmp_path)
    monkeypatch.setattr(jobs, "launch", lambda directory, request: jobs.write_json(directory / "request.json", request))
    response = service.call("kuafu_collect_drawer", {"verification_job": verified.name})
    directory = Path(response["result"]["directory"])
    request = jobs.read_json(directory / "request.json")
    with pytest.raises(FileNotFoundError):
        jobs.finish(directory, request, 0, None)
    output = directory / "output"
    output.mkdir()
    jobs.write_json(output / "report.json", {"kind": "drawer-collect", "completed": True, "artifacts": {}})
    with pytest.raises(ValueError, match="required output"):
        finish(directory, request)
    command = replay_command(directory, request, output / "episode", 42)
    assert "--record-demo" in command and command[command.index("--max-requests") + 1] == "0"
    policy_command = replay_command(directory, request, output / "evaluation", 46, policy=output / "policy.pth")
    assert "--policy" in policy_command and "--record-demo" not in policy_command


def test_policy_uses_checkpoint_normalizers_and_checks_scene_identity(tmp_path):
    torch = pytest.importorskip("torch")
    from kuafu.bc.generated_drawer import GeneratedDrawerPolicy
    from kuafu.bc.training.model import MLPActorPrior

    model = MLPActorPrior(13, 2, (4,))
    for parameter in model.parameters():
        parameter.data.zero_()
    model.mean_layer.bias.data[:] = torch.tensor([.5, 1.])
    path = tmp_path / "policy.pth"
    torch.save({"generated_drawer_contract": contract(), "obs_dim": 13, "act_dim": 2, "hidden_dims": [4],
                "activation": "leaky_relu", "negative_slope": .01, "dropout": 0., "action_mode": "absolute_joint",
                "action_chunk_size": 1, "obs_history": 1, "model_state_dicts": {"model_state_dict_norm": model.state_dict()},
                "obs_normalizer": {"mean": torch.zeros(13), "std": torch.ones(13)},
                "act_normalizer": {"mean": torch.tensor([1., 2.]), "std": torch.tensor([2., 2.])}}, path)
    arguments = {key: contract()[key] for key in ("joint_names", "cabinet_joint_names", "asset_hashes", "program_hashes")}
    policy = GeneratedDrawerPolicy(path, device="cpu", **arguments)
    actual = policy.action(snapshot(), 2, torch.tensor([[-3., 3.], [-3., 3.]]))
    torch.testing.assert_close(actual, torch.tensor([2., 3.]))
    with pytest.raises(ValueError, match="training contract"):
        GeneratedDrawerPolicy(path, device="cpu", **{**arguments, "program_hashes": {"program": "wrong"}})

    stage = torch.load(path, weights_only=False)
    relative = dict(stage, action_mode="relative_joint")
    torch.save({"generated_drawer_contract": contract(), "stages": [stage, relative]}, path)
    policy = GeneratedDrawerPolicy(path, device="cpu", **arguments)
    limits = torch.tensor([[-10., 10.], [-10., 10.]])
    torch.testing.assert_close(policy.action(snapshot(), 0, limits), torch.tensor([2., 4.]))
    torch.testing.assert_close(policy.action(snapshot(), 1, limits), torch.tensor([2.1, 4.2]))
    with pytest.raises(ValueError, match="no network"):
        policy.action(snapshot(), 2, limits)


def test_policy_success_cannot_be_promoted_as_an_mpc_verification(tmp_path):
    _, directory, _ = verified_service(tmp_path)
    report = jobs.read_json(directory / "output/verification.json")
    report.update(controller="bc_policy", solver_passed=None, mpc_solves=0)
    jobs.write_json(directory / "output/verification.json", report)
    result = jobs.finish(directory, jobs.read_json(directory / "request.json"), 0, None)
    assert result["status"] == "failed"
    assert "accepted_run" not in result


def test_nearest_policy_keeps_validation_episodes_out_of_its_state_table(tmp_path):
    torch = pytest.importorskip("torch")
    h5py = pytest.importorskip("h5py")
    from kuafu.bc.generated_drawer import GeneratedDrawerPolicy, fit_nearest_policy

    episodes = []
    for index in range(4):
        obs = np.stack([policy_observation(snapshot(), stage) for stage in (0, 0, 1, 1)])
        obs[:, 0] += index
        path = tmp_path / f"demo{index}.hdf5"
        write_demonstration(path, obs, obs[:, :2] + .01, obs, contract())
        episodes.append(path)
    dataset = tmp_path / "data.hdf5"
    combine_demonstrations(episodes, dataset)
    checkpoint = tmp_path / "policy.pth"
    result = fit_nearest_policy(dataset, checkpoint)
    assert len(result["train_episodes"]) == 3 and len(result["validation_episodes"]) == 1
    assert not set(result["train_episodes"]) & set(result["validation_episodes"])
    before = torch.load(checkpoint, weights_only=False)
    with h5py.File(dataset, "a") as handle:
        handle[f"data/{result['validation_episodes'][0]}/obs/state"][:, 0] = 1e6
    fit_nearest_policy(dataset, checkpoint)
    after = torch.load(checkpoint, weights_only=False)
    for key in before["nearest"]:
        torch.testing.assert_close(before["nearest"][key], after["nearest"][key])
    arguments = {key: contract()[key] for key in ("joint_names", "cabinet_joint_names", "asset_hashes", "program_hashes")}
    policy = GeneratedDrawerPolicy(checkpoint, device="cpu", **arguments)
    state = before["nearest"]["observations"][0]
    torch.testing.assert_close(policy.nearest_action(state, 0), before["nearest"]["actions"][0])
    with pytest.raises(ValueError, match="no decisions"):
        policy.nearest_action(state, 2)
