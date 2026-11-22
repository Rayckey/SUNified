"""Exercise the task success predicate without starting Isaac Sim."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def test_estop_open_requires_unlock_retreat_and_stage_progress():
    path = Path(__file__).resolve().parents[1] / (
        "src/kuafu/isaac/learned/kinova/kinova_estop_closed_env_learned.py"
    )
    tree = ast.parse(path.read_text())
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_external_get_dones"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    predicate = namespace["_external_get_dones"]

    cap = torch.zeros(4, 3)
    fingers = torch.tensor([[0.10, 0., 0.], [0.04, 0., 0.],
                            [0.05, 0., 0.], [0.10, 0., 0.]])

    def object_position(name):
        assert name == "estop_cap_tip"
        return cap

    def robot_position(name):
        assert name == "panda_fingertip"
        return fingers

    env = SimpleNamespace(
        device="cpu",
        estop=SimpleNamespace(is_open=torch.tensor([False, True, True, True]),
                              update_state=lambda: None),
        _unlocked_once=torch.zeros(4, dtype=torch.bool),
        _success=torch.zeros(4, dtype=torch.bool),
        current_stage=torch.tensor([[3], [3], [3], [2]]),
        cfg=SimpleNamespace(sun_program_evaluator_stage_override=False),
        program_runtime=SimpleNamespace(
            controller_api_wrapper=SimpleNamespace(
                get_object_xyz_position=object_position,
                lookup_robot_frame_position=robot_position,
            ),
            compiled_program=SimpleNamespace(stage_count=lambda: 4),
        ),
    )
    assert predicate(env).tolist() == [False, False, True, False]

    # An earlier unlock is remembered; retreat still has to occur.
    env.estop.is_open.fill_(False)
    fingers[1, 0] = 0.06
    env.current_stage.fill_(3)
    assert predicate(env).tolist() == [False, True, True, True]

    # Evaluator overrides bypass the stage gate, not the physical criterion.
    env.current_stage.zero_()
    assert predicate(env).tolist() == [False, False, False, False]
    env.cfg.sun_program_evaluator_stage_override = True
    assert predicate(env).tolist() == [False, True, True, True]
    env.cfg.sun_program_evaluator_stage_override = False
    env.cfg.ignore_stage_gate_for_success = True
    assert predicate(env).tolist() == [False, True, True, True]
