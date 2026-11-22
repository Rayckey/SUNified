"""Supervise the pretrained-encoder download and two LiLa-WAM training stages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from kuafu.dynamics.lilawam_data import file_hash, write_json
from kuafu.paths import resolve_artifact_path


def run(root, token_file=None, wait_for_access=False):
    """Wait without reserving a GPU, then train with the configured pretrained encoder."""
    from omegaconf import OmegaConf
    from kuafu.dynamics.lilawam import download_encoder

    root = resolve_artifact_path(root, "LiLa-WAM pipeline")
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "pipeline.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("This LiLa-WAM pipeline is already running") from None
    recorded_steps = [json.loads(path.read_text()).get("step", 0)
                      for path in (root / "stage1/progress.json", root / "stage2/progress.json") if path.exists()]
    state = {"pid": os.getpid(), "production_training_started": any(step > 0 for step in recorded_steps)}

    def status(phase, **fields):
        state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
        write_json(root / "pipeline_status.json", state)
        print(json.dumps(state), flush=True)

    def command(arguments, log_name, phase):
        status(phase)
        with (root / log_name).open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", "-m", "kuafu.dynamics.lilawam", *arguments],
                                     stdout=log, stderr=subprocess.STDOUT)
            state["child_pid"] = child.pid
            status(phase)
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    if phase.startswith("running_stage"):
                        progress = root / phase.removeprefix("running_") / "progress.json"
                        if progress.exists():
                            observed = json.loads(progress.read_text())
                            step = observed.get("step", 0)
                            status(phase, production_training_started=state["production_training_started"] or step > 0,
                                   observed_optimizer_step=step, target_updates=observed.get("target"))
        state.pop("child_pid", None)
        if code:
            status("needs_attention", failed_phase=phase, exit_code=code, log=str(root / log_name))
            raise RuntimeError(f"{phase} exited {code}; inspect {root / log_name}")

    try:
        config = OmegaConf.load(root / "stage1.yaml")
        encoder = Path(config.model.vision_encoder.checkpoint_path)
        # A successful pinned download writes provenance only after hashing the complete weights.
        while not (encoder / "kuafu_provenance.json").is_file():
            status("checking_encoder_access")
            try:
                download_encoder(encoder, token_file)
            except Exception as error:
                code = getattr(getattr(error, "response", None), "status_code", None)
                if code in (401, 403):
                    phase = "waiting_for_encoder_access"
                elif (code is not None and code >= 500) or type(error).__name__ in (
                        "ConnectError", "ConnectTimeout", "ReadTimeout", "LocalEntryNotFoundError"):
                    phase = "waiting_for_network"
                else:
                    status("needs_attention", failed_phase="encoder_download", error_type=type(error).__name__,
                           http_status=code)
                    raise RuntimeError("Encoder download failed; credentials and request details were not logged") from None
                status(phase, http_status=code, error_type=type(error).__name__, retry_seconds=300,
                       access_url="https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m")
                if not wait_for_access:
                    return
                time.sleep(300)

        status("verifying_inputs", retry_seconds=None, http_status=None, error_type=None)
        provenance = json.loads((encoder / "kuafu_provenance.json").read_text())
        if (provenance["revision"] != "ea8dc2863c51be0a264bab82070e3e8836b02d51"
                or provenance["weights_sha256"] != file_hash(encoder / "model.safetensors")):
            raise ValueError("Pretrained encoder provenance mismatch")
        manifest = json.loads(Path(config.kuafu.manifest).read_text())
        for task, spec in manifest["tasks"].items():
            if spec["source"]["sha256"] != file_hash(spec["source"]["path"]):
                raise ValueError(f"Recording changed since preparation: {task}")

        gpu = int(str(config.kuafu.device).split(":")[1])

        def wait_for_gpu():
            stable = 0
            while stable < 2:
                result = subprocess.check_output(["nvidia-smi", f"--id={gpu}",
                    "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
                free, utilization = [int(value.strip()) for value in result.split(",")]
                stable = stable + 1 if free >= 25000 and utilization <= 10 else 0
                if stable < 2:
                    status("waiting_for_gpu", gpu=gpu, free_mib=free, utilization_percent=utilization)
                    time.sleep(30)

        wait_for_gpu()
        command(["conditions", "--config", str(root / "stage1.yaml")], "conditions.log", "preparing_task_conditions")
        for stage in (1, 2):
            directory = root / f"stage{stage}"
            progress_path = directory / "progress.json"
            previous = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            if previous.get("phase") == "complete":
                continue
            wait_for_gpu()
            args = ["train", "--config", str(root / f"stage{stage}.yaml"), "--output", str(directory)]
            if (directory / "last.pt").exists():
                args += ["--resume", str(directory / "last.pt")]
            elif directory.exists():
                raise ValueError(f"Incomplete initialization without a checkpoint at {directory}; inspect before restarting")
            elif stage == 2:
                args += ["--init-from", str(root / "stage1" / "best.pt")]
            status(f"starting_stage{stage}", stage=stage)
            command(args, f"stage{stage}.log", f"running_stage{stage}")
            finished = json.loads(progress_path.read_text())
            if finished.get("phase") != "complete" or finished.get("step", 0) < 1:
                raise RuntimeError(f"Stage {stage} exited without completing optimizer updates")
            state["production_training_started"] = True
        status("complete", stage=2, evaluation=str(root / "stage2" / "evaluation.json"))
    except Exception as error:
        if state.get("phase") != "needs_attention":
            status("needs_attention", error_type=type(error).__name__)
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--token-file")
    parser.add_argument("--wait-for-access", action="store_true")
    arguments = parser.parse_args()
    run(arguments.root, arguments.token_file, arguments.wait_for_access)


if __name__ == "__main__":
    main()
