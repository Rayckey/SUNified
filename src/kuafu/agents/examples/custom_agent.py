"""Connect any program-authoring agent to Kuafu's measured repair loop.

Implement ``propose_program(observation)`` with your model/framework and call
``solve_drawer``. The callback returns JSON matching observation['contract']['schema'].
The callback owns model access and inference limits.
"""

import time
from collections.abc import Callable

from kuafu.agents import ToolService


def wait_for_job(kuafu: ToolService, job_id: str) -> dict:
    deadline = time.monotonic() + kuafu.timeout_s + 30
    while time.monotonic() < deadline:
        response = kuafu.call("kuafu_job_status", {"job_id": job_id})
        if not response["ok"]:
            raise RuntimeError(response["error"])
        if response["result"]["status"] not in {"queued", "running"}:
            return response["result"]
        time.sleep(2)
    kuafu.call("kuafu_cancel_job", {"job_id": job_id})
    raise TimeoutError(f"No final result for job {job_id}; inspect its logs")


def solve_drawer(
    propose_program: Callable[[dict], dict],
    asset: str,
    *,
    task: str = "Open the top drawer by 10 cm.",
    target_travel_m: float = 0.10,
    max_revisions: int = 3,
    kuafu: ToolService | None = None,
) -> dict:
    """Let the caller's agent revise proposals against a fixed physical objective."""
    if max_revisions < 1:
        raise ValueError("max_revisions must be positive")
    kuafu = kuafu or ToolService()
    if kuafu.run_id is None:
        created = kuafu.call("kuafu_create_run", {"task": task})
        if not created["ok"]:
            return created
        kuafu.run_id = created["result"]["run_id"]
    contract = kuafu.call("kuafu_contract", {"kind": "program"})["result"]
    started = kuafu.call("kuafu_inspect_drawer", {"asset": asset, "task": task, "target_travel_m": target_travel_m})
    if not started["ok"]:
        return started
    job_id = started["result"]["job_id"]
    context = wait_for_job(kuafu, job_id)
    if context["status"] != "succeeded":
        return {"solved": False, "feedback": context}
    feedback = None
    for revision in range(max_revisions):
        proposal = propose_program(
            {"task": task, "scene": context["scene"], "contract": contract, "feedback": feedback, "revision": revision}
        )
        submitted = kuafu.call("kuafu_submit_program", {"context_job": job_id, "program": proposal})
        if not submitted["ok"]:
            feedback = submitted
            continue
        verification = kuafu.call("kuafu_verify_program", {"program_id": submitted["result"]["program_id"]})
        if not verification["ok"]:
            feedback = verification
            continue
        feedback = wait_for_job(kuafu, verification["result"]["job_id"])
        if feedback["status"] == "succeeded" and feedback.get("accepted_run"):
            return {"solved": True, "run_id": kuafu.run_id,
                    "accepted_run": feedback["accepted_run"], "verification": feedback}
    return {"solved": False, "run_id": kuafu.run_id, "feedback": feedback, "reason": "Revision budget exhausted"}
