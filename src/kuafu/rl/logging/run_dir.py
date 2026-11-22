"""Run directory management + jsonl logging + tensorboard feedback reader."""

from __future__ import annotations

import datetime
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kuafu.paths import DEFAULT_EXTERNAL_ARTIFACT_ROOT
from kuafu.rl.isaac.task_registry import (
    resolve_artifact_robot_id,
    resolve_execution_robot_id,
    task_log_root,
)


@dataclass(frozen=True)
class RunDirs:
    root_dir: str
    openai_payloads_path: str
    prompt_transcript_path: str
    trials_path: str
    iterations_path: str
    videos_path: str
    tb_dir: str
    checkpoints_dir: str
    best_path: str


def _write_run_metadata(
    root_dir: str,
    argv: list[str],
    *,
    launch_script_path: str | None = None,
    execution_robot_id: str | None = None,
    artifact_robot_id: str | None = None,
    log_base_dir: str | None = None,
) -> None:
    """Write run_params.json (full command line) and optionally copy launch script into root_dir."""
    params_path = os.path.join(root_dir, "run_params.json")
    ts = datetime.datetime.now().isoformat()
    payload = {
        "timestamp_iso": ts,
        "argv": argv,
        "command_line": " ".join(argv),
        "execution_robot_id": execution_robot_id,
        "artifact_robot_id": artifact_robot_id,
        "log_base_dir": log_base_dir,
    }
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if launch_script_path and os.path.isfile(launch_script_path):
        dest = os.path.join(root_dir, os.path.basename(launch_script_path))
        shutil.copy2(launch_script_path, dest)


def _default_log_base_dir() -> str:
    explicit = os.environ.get("KUAFU_RESIDUAL_LOG_ROOT", "").strip()
    if explicit:
        return str(Path(explicit).expanduser().resolve())

    task_root = os.environ.get("KUAFU_TASK_ARTIFACT_ROOT", "").strip()
    if task_root:
        return str(Path(task_root).expanduser().resolve() / "residual" / "runs")

    configured_root = os.environ.get("KUAFU_ARTIFACT_ROOT", "").strip()
    base = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else DEFAULT_EXTERNAL_ARTIFACT_ROOT.resolve()
    )
    return str(base / "standalone" / "residual_rl" / "runs")


def make_run_dirs(
    *,
    task: str,
    argv: list[str] | None = None,
    launch_script_path: str | None = None,
    run_timestamp: str | None = None,
    execution_robot_id: str | None = None,
    artifact_robot_id: str | None = None,
    log_base_dir: str | None = None,
    legacy_fr3_finray_layout: bool = True,
) -> RunDirs:
    effective_log_base_dir = log_base_dir or _default_log_base_dir()
    ts = run_timestamp or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    resolved_execution_robot_id = resolve_execution_robot_id(task, execution_robot_id)
    resolved_artifact_robot_id = resolve_artifact_robot_id(artifact_robot_id)
    task_root = task_log_root(
        effective_log_base_dir,
        task,
        resolved_execution_robot_id,
        legacy_fr3_finray_layout=legacy_fr3_finray_layout,
    )
    root = os.path.abspath(os.path.join(task_root, ts))
    os.makedirs(root, exist_ok=True)
    if argv is not None:
        _write_run_metadata(
            root,
            argv,
            launch_script_path=launch_script_path,
            execution_robot_id=resolved_execution_robot_id,
            artifact_robot_id=resolved_artifact_robot_id,
            log_base_dir=effective_log_base_dir,
        )
    tb_dir = os.path.join(root, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    ckpt_dir = os.path.join(root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return RunDirs(
        root_dir=root,
        openai_payloads_path=os.path.join(root, "openai_payloads.jsonl"),
        prompt_transcript_path=os.path.join(root, "prompt_transcript.md"),
        trials_path=os.path.join(root, "trials.jsonl"),
        iterations_path=os.path.join(root, "iterations.jsonl"),
        videos_path=os.path.join(root, "videos.jsonl"),
        tb_dir=tb_dir,
        checkpoints_dir=ckpt_dir,
        best_path=os.path.join(root, "best.json"),
    )


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def _format_message_content_for_transcript(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        rendered_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                rendered_parts.append(str(part.get("text", "")))
            elif isinstance(part, dict) and part.get("type") == "image_url":
                image_url = part.get("image_url")
                image_url_data = image_url if isinstance(image_url, dict) else {}
                detail = image_url_data.get("detail", "unknown")
                url = str(image_url_data.get("url", ""))
                if url.startswith("data:"):
                    url = "<attached image omitted from transcript>"
                rendered_parts.append(f"[image_url detail={detail}: {url}]")
            else:
                rendered_parts.append(json.dumps(part, ensure_ascii=False, default=str))
        return "\n\n".join(rendered_parts)
    return json.dumps(content, ensure_ascii=False, default=str)

def append_prompt_transcript(
    *,
    path: str,
    header: dict[str, Any],
    messages: list[dict[str, Any]],
    raw_outputs: list[str],
) -> None:
    """Append a human-readable prompt/response block (markdown)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n---\n")
        f.write(f"- timestamp: {header.get('timestamp')}\n")
        if "kind" in header:
            f.write(f"- kind: {header.get('kind')}\n")
        if "stage_id" in header:
            f.write(f"- stage_id: {header.get('stage_id')}\n")
        if "iter" in header:
            f.write(f"- iter: {header.get('iter')}\n")
        if "best_run_idx" in header:
            f.write(f"- best_run_idx: {header.get('best_run_idx')}\n")
        f.write("\n")

        f.write("### Messages\n")
        for m in messages:
            role = m.get("role", "unknown")
            content = _format_message_content_for_transcript(m.get("content", ""))
            f.write(f"\n#### {role}\n\n")
            f.write("```text\n")
            f.write(content.rstrip() + "\n")
            f.write("```\n")

        f.write("\n### Raw outputs\n")
        for i, out in enumerate(raw_outputs):
            f.write(f"\n#### output_{i}\n\n")
            f.write("```text\n")
            f.write((out or "").rstrip() + "\n")
            f.write("```\n")


def load_tensorboard_scalars(path: str) -> dict[str, list[float]]:
    """Load scalar tags from a tensorboard log directory."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as e:
        raise RuntimeError("Missing dependency `tensorboard`. Install via: `pip install tensorboard`.") from e

    data = defaultdict(list)
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    for tag in event_acc.Tags().get("scalars", []):
        for ev in event_acc.Scalars(tag):
            data[tag].append(ev.value)
    return data


def make_tb_writer(tb_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as e:
        raise RuntimeError("Missing TensorBoard integration. Ensure `torch` and `tensorboard` are installed.") from e

    return SummaryWriter(tb_dir)

