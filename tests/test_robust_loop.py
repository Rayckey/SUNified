"""Exercise real learn-programs recovery with deterministic, simulator-free stages."""

from collections import deque
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kuafu.programs.llm.sun_program_plan import SUNProgramPlan
from kuafu.programs.robust_loop import run_robust_program_learning_loop


DRAWER_TASKS = [
    "approach handle", "contact handle", "grasp handle", "open drawer",
    "release handle", "retract from handle", "approach strawberry",
    "contact strawberry", "grasp strawberry", "lift strawberry",
    "carry strawberry", "lower strawberry", "release strawberry",
    "retract from strawberry",
]


class RunHarness:
    """Model/physics boundaries are fake; task lists and the runner are real."""

    def __init__(self, failures=(), *, stall_detection=True, tasks=None):
        # Failure entries are (stage index, is_replay), consumed in order.
        self.failures = deque(failures)
        self.tasks = list(DRAWER_TASKS if tasks is None else tasks)
        self.stall_detection = stall_detection
        self.plan_calls = 0
        self.reset_calls = 0
        self.loop_calls = 0
        self.groundings = []
        self.replays = []
        self.closed = False
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.unwrapped = self
        self.program_runtime = None

    def make_program_runtime(self):
        self.program_runtime = FakeProgramRuntime(self)
        return self.program_runtime

    def is_running(self):
        self.loop_calls += 1
        assert self.loop_calls < 1000, "Recovery did not converge in the test"
        return True

    def reset(self):
        self.reset_calls += 1
        self.episode_length_buf.zero_()

    def step(self, actions):
        self.program_runtime.stage_steps += 1
        self.episode_length_buf += 1
        return {}, torch.zeros(1), torch.tensor([False]), torch.tensor([False]), {}

    def close(self):
        self.closed = True

    def run(self, output_dir, *, max_counter=None):
        return run_robust_program_learning_loop(
            env=self,
            env_cfg=SimpleNamespace(sim=SimpleNamespace(device="cpu")),
            program_runtime=self.make_program_runtime(),
            make_program_runtime=self.make_program_runtime,
            update_data_in_controller=lambda *args, **kwargs: None,
            simulation_app=self,
            output_dir=str(output_dir),
            max_counter=max_counter,
            enable_stall_detection=self.stall_detection,
            constraint_call_interval=3,
            recompute_call_interval=1,
            task_update_uses_camera=False,
        )


class FakeProgramRuntime:
    """Implement SUNProgramExecutor's stage-state contract using the real SUNProgramPlan."""

    def __init__(self, harness):
        self.harness = harness
        self.tasks = SUNProgramPlan()
        self.learner = SimpleNamespace(
            _tasks=self.tasks, _constraints={}, prompt_manager=SimpleNamespace(),
        )
        self.program_learner = SimpleNamespace(_constraint_generator=self.learner)
        self.controller_api_wrapper = SimpleNamespace(
            device="cpu", batch=1, dtype=torch.float32,
            updatePOVState=lambda: None,
        )
        self.is_replay = False
        self.failing = False
        self.stage_steps = 0

    def callbackTask(self):
        self.harness.plan_calls += 1
        self.tasks.update_tasks(list(self.harness.tasks))
        return True

    def callbackAtomicActions(self, error=None):
        self.harness.groundings.append((self.snapshot_program_stage(), error))
        self.tasks.update_atomic_actions([f"Execute {self.tasks.current_task}"])
        self.is_replay = False
        return True

    def callbackConstraint(self):
        self.learner._constraints = {"Constraints": [{"op": "test_move"}]}
        return True

    def applyConstraint(self, **kwargs):
        self.stage_steps = 0
        stage_index = len(self.tasks.completed_tasks)
        # The saved stage index remains authoritative when reproducing the old
        # bug, where loading a replay erased completed_tasks as well as pending.
        if self.is_replay:
            stage_index = self.loaded_stage_index
        event = (stage_index, self.is_replay)
        self.failing = bool(self.harness.failures and self.harness.failures[0] == event)
        if self.failing:
            self.harness.failures.popleft()

    def computeControl(self):
        return np.zeros((2, 9), dtype=np.float32)

    def hasStalled(self):
        return self.failing

    def stallFailureReason(self):
        return "controller stalled (test)"

    def hasCollision(self):
        return False

    def quantifyPredicate(self, text=False, **kwargs):
        satisfied = torch.tensor([self.stage_steps > 0 and not self.failing])
        return (satisfied, "test predicate") if text else satisfied

    def terminateCurrentSubtask(self):
        self.tasks.pop_task()

    def hasMoreTasks(self):
        return self.tasks.has_tasks

    def snapshot_program_stage(self):
        return deepcopy({
            "task": self.tasks.current_task,
            "atomic_actions": self.tasks.current_atomic_actions,
            "constraints": self.learner._constraints,
            "pending_tasks": self.tasks.pending_tasks,
            "completed_tasks": self.tasks.completed_tasks,
            "completed_task_atomic_actions": self.tasks.completed_task_atomic_actions,
        })

    def set_program_stage(
        self, task, atomic_actions, constraints, *, pending_tasks=None,
        completed_tasks=None, completed_task_atomic_actions=None,
    ):
        # Same queue restoration semantics as SUNProgramExecutor.set_program_stage.
        self.tasks.update_tasks([task] + list(pending_tasks or []))
        self.tasks.update_atomic_actions(list(atomic_actions))
        self.tasks._completed_tasks = list(completed_tasks or [])
        self.tasks._completed_task_atomic_actions = deepcopy(completed_task_atomic_actions or [])
        self.learner._constraints = deepcopy(constraints)
        self.is_replay = False

    def record_program(self, task_path, atomic_path, constraints_path):
        Path(task_path).write_text(self.tasks.current_task, encoding="utf-8")
        Path(atomic_path).write_text(json.dumps(self.tasks.current_atomic_actions), encoding="utf-8")
        Path(constraints_path).write_text(json.dumps(self.learner._constraints), encoding="utf-8")

    def load_program_stage(self, path, **context):
        path = Path(path)
        stage = {
            "task": (path / "task.txt").read_text(encoding="utf-8"),
            "atomic_actions": json.loads((path / "atomic.txt").read_text(encoding="utf-8")),
            "constraints": json.loads((path / "constraints.json").read_text(encoding="utf-8")),
        }
        self.set_program_stage(**stage, **context)
        self.is_replay = True
        self.loaded_stage_index = int(path.name.removeprefix("subtask_"))
        self.harness.replays.append((self.loaded_stage_index, self.snapshot_program_stage()))
        return stage


def assert_complete_run(harness, output_dir):
    assert not harness.failures, "The requested failure sequence was not exercised"
    assert harness.program_runtime.tasks.completed_tasks == harness.tasks
    assert not harness.program_runtime.tasks.tasks
    assert harness.plan_calls == 1
    assert harness.closed
    assert (output_dir / "success.txt").read_text() == "True"
    assert f"completed_subtasks: {len(harness.tasks)}\n" in (output_dir / "stats.txt").read_text()
    for index in range(len(harness.tasks)):
        status = (output_dir / f"program/subtask_{index}/status.txt").read_text()
        assert "status: succeeded" in status
        assert f"subtask_index: {index}\n" in status


@pytest.mark.parametrize("stall_detection", [True, False], ids=["stall", "timeout"])
@pytest.mark.parametrize("failures", [
    [(12, False), (11, True)],
    [(10, False), (11, False), (10, True)],
    [(3, False), (0, True)],
    [(12, False), (11, True), (10, True)],
    [(12, False), (11, True), (11, False)],
], ids=["trial4", "trial5", "first-replay-stage", "nested-replay-failures", "correction-fails-again"])
def test_replay_failure_preserves_remaining_plan(tmp_path, failures, stall_detection):
    harness = RunHarness(failures, stall_detection=stall_detection)
    assert harness.run(tmp_path)
    assert_complete_run(harness, tmp_path)
    assert harness.reset_calls == len(failures)

    # Rewind completed history to the replay position and retain the entire
    # suffix, including the original failed action and every subsequent action.
    for index, snapshot in harness.replays:
        assert snapshot["task"] == harness.tasks[index]
        assert snapshot["completed_tasks"] == harness.tasks[:index]
        assert snapshot["pending_tasks"] == harness.tasks[index + 1:]
        assert [entry["task"] for entry in snapshot["completed_task_atomic_actions"]] == harness.tasks[:index]

    for snapshot, error in harness.groundings:
        if error is not None:
            index = len(snapshot["completed_tasks"])
            assert snapshot["task"] == harness.tasks[index]
            assert snapshot["pending_tasks"] == harness.tasks[index + 1:]


@pytest.mark.parametrize("failures", [[], [(0, False)], [(11, False)], [(13, False)]],
                         ids=["no-failure", "first-stage", "normal-recovery", "last-stage"])
def test_normal_completion_and_recovery(tmp_path, failures):
    harness = RunHarness(failures)
    assert harness.run(tmp_path)
    assert_complete_run(harness, tmp_path)


def test_replay_uses_positions_not_unique_task_descriptions(tmp_path):
    harness = RunHarness([(2, False), (1, True)], tasks=["move", "move", "move", "release"])
    assert harness.run(tmp_path)
    assert_complete_run(harness, tmp_path)


def test_total_step_budget_stops_before_grounding_another_stage(tmp_path):
    harness = RunHarness(tasks=["approach", "grasp"])
    with pytest.raises(RuntimeError, match="Maximum control steps reached"):
        harness.run(tmp_path, max_counter=1)
    assert len(harness.groundings) == 1
    assert harness.plan_calls == 1
    assert "control_steps: 1" in (tmp_path / "stats.txt").read_text()
    assert (tmp_path / "success.txt").read_text() == "False"


def test_last_budgeted_step_can_complete_the_program(tmp_path):
    harness = RunHarness(tasks=["approach"])
    assert harness.run(tmp_path, max_counter=1)
    assert_complete_run(harness, tmp_path)


def test_step_budget_survives_restarts(tmp_path):
    harness = RunHarness([(1, False)], tasks=["approach", "grasp"])
    with pytest.raises(RuntimeError, match="Maximum control steps reached"):
        harness.run(tmp_path, max_counter=2)
    assert harness.reset_calls == 1
    assert "control_steps: 2" in (tmp_path / "stats.txt").read_text()
