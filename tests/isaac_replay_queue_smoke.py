"""Check the recovery regression with real SUNProgramExecutor stage-state APIs, no LLM calls.

Run with isaaclab.sh -p tests/isaac_replay_queue_smoke.py --headless.
AppLauncher enables SUNProgramExecutor's simulator imports; model/control outcomes remain
deterministic test doubles. This is not a physical task-success test.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

from kuafu.mpc.simulation.sun_program_executor import SUNProgramExecutor
from test_robust_loop import FakeProgramRuntime, RunHarness, assert_complete_run


class ActualStateProgramRuntime(FakeProgramRuntime, SUNProgramExecutor):
    snapshot_program_stage = SUNProgramExecutor.snapshot_program_stage
    set_program_stage = SUNProgramExecutor.set_program_stage

    def __init__(self, harness):
        super().__init__(harness)
        self._object_grasp_hold_counter = None

    def load_program_stage(self, path, **context):
        stage = SUNProgramExecutor.load_program_stage(self, path, **context)
        self.is_replay = True
        self.loaded_stage_index = int(Path(path).name.removeprefix("subtask_"))
        self.harness.replays.append((self.loaded_stage_index, self.snapshot_program_stage()))
        return stage

    def callbackAtomicActions(self, error=None):
        if error is not None:
            history = self.tasks.completed_task_atomic_actions
            expected = history[-1]["atomic_actions"] if history else []
            assert self.tasks.last_completed_atomic_actions == expected
        return super().callbackAtomicActions(error=error)


class ActualStateHarness(RunHarness):
    def make_program_runtime(self):
        self.program_runtime = ActualStateProgramRuntime(self)
        return self.program_runtime


try:
    cases = {
        "trial4": [(12, False), (11, True)],
        "trial5": [(10, False), (11, False), (10, True)],
        "nested_replay_failures": [(12, False), (11, True), (10, True)],
        "first_replay_stage": [(3, False), (0, True)],
    }
    for name, failures in cases.items():
        with TemporaryDirectory(prefix="kuafu-replay-queue-") as run_dir:
            harness = ActualStateHarness(failures)
            assert harness.run(Path(run_dir))
            assert_complete_run(harness, Path(run_dir))
            for index, snapshot in harness.replays:
                assert snapshot["completed_tasks"] == harness.tasks[:index]
                assert snapshot["pending_tasks"] == harness.tasks[index + 1:]
            print(f"SUN_PROGRAM_EXECUTOR_QUEUE_CHECK {name}: passed all 14 stages", flush=True)
    print("SUN_PROGRAM_EXECUTOR_QUEUE_CHECK all 4 cases passed; no model calls", flush=True)
finally:
    launcher.app.close()
