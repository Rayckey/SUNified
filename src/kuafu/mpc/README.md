# Model predictive control

Use MPC to turn a SUN Program into robot actions and measured feedback for your
agent. A failed stage returns its predicate breakdown and solver trace; a passing
program is checked against actual object motion before downstream collection.

The current action solver uses **CasADi / IPOPT**, with **Pinocchio** kinematics
and bounded inverse kinematics. Torch evaluates program predicates and residuals
across environments. This module is the verification backend used by Kuafu's
agent tools.

## Install

From the repository root:

```bash
python -m pip install -e ".[agents,programs,mpc,isaac]"
```

See [runtime setup](../../../docs/INSTALL.md) for the CPU and Isaac environments.

## Input and output

| Direction | Contract | Concrete example |
|---|---|---|
| Input | Scene state, robot model and compiled SUN Program | [input.json](examples/input.json) |
| Output | Control trajectory, solver trace and physical verification | [output.json](examples/output.json) |

Supply observed robot/object frames, an articulated robot model, and SUN stages
with atomic actions and constraints. Positions use meters; quaternions use
**wxyz**. Frame names come from scene inspection. The receiver assigns numeric
tolerances and owns the physical acceptance check.

## Use

For an agent that has inspected the scene and submitted its stages:

```python
from kuafu.agents import ToolService

tools = ToolService(run_id="my-task")
job = tools.call("kuafu_verify_program", {"program_id": "ID_FROM_kuafu_submit_program"})
result = tools.call("kuafu_job_status", {"job_id": job["result"]["job_id"]})
```

Poll until the job finishes. A successful result includes `accepted_run` and its
program identity. Read these files in the job's output directory:

| File | What it tells your agent |
|---|---|
| `verification.json` | Per-stage predicates, physical task result, joint travel and errors |
| `mpc_solves.json` | Solve success, finite actions and IK diagnostics |
| `trajectory.json` | Measured robot/object states throughout execution |
| `final_table_cam_0.png` | Final scene for visual review |
| `progress.json` | Last active stage and operation for interruption recovery |

For a complete command and recorded rollout, run the
[prepared cabinet example](../../../examples/02_program_to_rollout/README.md).
For your own agent proposals, follow the [agent interface](../../../docs/AGENT_INTERFACE.md).
The bundled generated-scene adapter checks top-drawer opening; registered task
families use their corresponding scene and success adapters.

## Source map

| Directory | Responsibility |
|---|---|
| `controller/` | CasADi MPC, solver utilities and bounded IK |
| `simulation/` | SUN Program execution, stage transitions and simulator feedback |
| `torch/` | Batched predicate and residual evaluation |

Run `python -m pytest -q tests/test_bounded_ik.py tests/test_generated_drawer_verification.py`
for kinematic limits and independent physical acceptance checks, then run the
cabinet replay for changes to control behavior.
