"""Video export for LLM feedback (no shell commands).

We run a short *evaluation playback* from a saved RL-Games checkpoint and write
per-env mp4s to disk. The returned string is formatted for direct inclusion in
the LLM prompt (YAML-ish list of rollouts).

Important Isaac constraint:
- Isaac Sim must be created in a fresh process via AppLauncher.
  Therefore we do the work in a spawned Python process (not a subprocess cmd).
"""

from __future__ import annotations

import os
import multiprocessing
from pathlib import Path
from typing import Any


def _pin_single_gpu_for_export(dev: str) -> str:
    """Restrict the exporter process to one visible GPU before Isaac boots.

    Kuafu training workers already pin themselves to a single GPU via
    ``CUDA_VISIBLE_DEVICES``. The video exporter used during global finetuning
    runs in its own spawned process, so it must do the same explicitly.

    This avoids Isaac/Kit trying to initialize interop across every visible GPU,
    which is fragile in Docker and can surface as CUDA init failures even when
    normal RL training works.
    """
    dev_in = str(dev or "cuda:0")
    if not dev_in.startswith("cuda"):
        return dev_in

    requested_idx = 0
    if ":" in dev_in:
        tail = dev_in.split(":", 1)[1].strip()
        first = tail.split(",", 1)[0].strip()
        if first.isdigit():
            requested_idx = int(first)

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    chosen: str | None = None
    if cvd:
        ids = [s.strip() for s in cvd.split(",") if s.strip()]
        if ids:
            if 0 <= requested_idx < len(ids):
                chosen = ids[requested_idx]
            else:
                chosen = ids[0]
    else:
        chosen = str(requested_idx)

    if chosen is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = chosen
        return "cuda:0"
    return dev_in


def _export_pov_videos_entry(
    qobj: Any,
    task: str,
    checkpoint_path: str,
    out_dir: str | Path,
    video_length: int,
    num_envs_to_record: int,
    dev: str,
    headless: bool,
    seed: int,
    execution_robot_id: str | None,
    all_stages_weights: dict[int, dict[int, float]] | None,
    all_stages_parameters: dict[int, dict[int, dict[str, object]]] | None,
    extra_sys_paths: list[str] | None = None,
) -> None:
    """Spawned-process entry point for exporting POV videos.

    Defined at module top level so it is picklable under the 'spawn' start method.
    ``extra_sys_paths`` is needed because 'spawn' does NOT inherit ``sys.path``.
    """
    import sys
    for p in (extra_sys_paths or []):
        if p not in sys.path:
            sys.path.insert(0, p)
    result = None
    try:
        pinned_dev = _pin_single_gpu_for_export(str(dev))
        print(
            "[residual_rl][video] exporter child start: "
            f"task={task} checkpoint={checkpoint_path} output_dir={out_dir} "
            f"device_in={dev} pinned_device={pinned_dev} "
            f"video_length={video_length} num_envs_to_record={num_envs_to_record} "
            f"headless={headless} seed={seed} execution_robot_id={execution_robot_id} "
            f"has_all_stages_weights={bool(all_stages_weights)} "
            f"has_all_stages_parameters={bool(all_stages_parameters)}",
            flush=True,
        )
        from kuafu.rl.video.pov_eval import PovEvalConfig, export_pov_videos

        mp4s = export_pov_videos(
            PovEvalConfig(
                task=str(task),
                checkpoint_path=str(checkpoint_path),
                output_dir=str(out_dir),
                video_length=int(video_length),
                num_envs_to_record=int(num_envs_to_record),
                device=str(pinned_dev),
                headless=bool(headless),
                seed=int(seed),
                execution_robot_id=execution_robot_id,
                all_stages_weights=all_stages_weights,
                all_stages_parameters=all_stages_parameters,
            ),
            _close_app=False,
        )
        print(f"[residual_rl][video] exporter child success: mp4_count={len(mp4s)}", flush=True)
        result = {"ok": True, "mp4s": mp4s}
    except BaseException as e:
        print(f"[residual_rl][video] exporter child exception: {type(e).__name__}: {e}", flush=True)
        result = {"ok": False, "err": f"{type(e).__name__}: {e}"}
    finally:
        qobj.put(result if result is not None else {"ok": False, "err": "process crashed before producing a result"})
        try:
            qobj.close()
            qobj.join_thread()
        except BaseException as exc:
            print(
                f"[residual_rl][video] WARNING: failed to close exporter result queue: {type(exc).__name__}: {exc}",
                flush=True,
            )
        import os
        os._exit(0)


def generate_video_feedback(
    *,
    task: str,
    checkpoint_path: str,
    video_length: int,
    num_envs_to_record: int,
    output_dir: str,
    device: str | None,
    headless: bool,
    seed: int,
    execution_robot_id: str | None = None,
    all_stages_weights: dict[int, dict[int, float]] | None = None,
    all_stages_parameters: dict[int, dict[int, dict[str, object]]] | None = None,
    gemini_enabled: bool = True,
    gemini_model: str = "gemini-2.0-flash",
    gemini_api_key: str | None = None,
    context_text: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate videos and return (prompt_block, metadata)."""
    meta: dict[str, Any] = {
        "task": str(task),
        "checkpoint_path": str(checkpoint_path),
        "video_length": int(video_length),
        "num_envs_to_record": int(num_envs_to_record),
        "output_dir": str(output_dir),
        "device": str(device) if device is not None else None,
        "headless": bool(headless),
        "seed": int(seed),
        "execution_robot_id": execution_robot_id,
        "all_stages_weights": all_stages_weights,
        "all_stages_parameters": all_stages_parameters,
        "ok": False,
        "mp4s": [],
        "error": None,
        "gemini": None,
    }

    if not os.path.exists(checkpoint_path):
        meta["error"] = f"checkpoint missing on disk: {checkpoint_path}"
        return (f"\n[residual_rl] Video feedback: {meta['error']}\n", meta)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = str(device) if device is not None else "cuda:0"
    # Multi-GPU strings like "cuda:0,1,2,3" are valid for RL training but video
    # export spawns a single Isaac Sim instance — use only the first GPU.
    if "," in dev:
        prefix, ids = dev.split(":", 1) if ":" in dev else ("cuda", dev)
        dev = f"{prefix}:{ids.split(',')[0]}"

    import sys
    extra_paths = list(sys.path)

    ctx = multiprocessing.get_context("spawn")
    q: Any = ctx.Queue()
    p = ctx.Process(
        target=_export_pov_videos_entry,
        args=(
            q,
            str(task),
            str(checkpoint_path),
            str(out_dir),
            int(video_length),
            int(num_envs_to_record),
            str(dev),
            bool(headless),
            int(seed),
            execution_robot_id,
            all_stages_weights,
            all_stages_parameters,
            extra_paths,
        ),
    )
    p.daemon = False
    print(
        "[residual_rl][video] launching exporter subprocess: "
        f"task={task} checkpoint={checkpoint_path} output_dir={output_dir} "
        f"device={dev} video_length={video_length} num_envs_to_record={num_envs_to_record} "
        f"execution_robot_id={execution_robot_id} "
        f"has_all_stages_weights={bool(all_stages_weights)} "
        f"has_all_stages_parameters={bool(all_stages_parameters)}",
        flush=True,
    )
    p.start()
    p.join(timeout=max(300.0, float(video_length) * 2.0))
    if p.is_alive():
        try:
            p.terminate()
        except Exception as exc:
            raise RuntimeError("[residual_rl][video] exporter timed out and could not be terminated.") from exc
        meta["error"] = "exporter timed out"
        return ("\n[residual_rl] Video feedback: exporter timed out.\n", meta)

    try:
        payload = q.get_nowait()
    except Exception:
        exit_code = getattr(p, "exitcode", "?")
        print(f"[residual_rl][video] exporter subprocess exited without payload: exit_code={exit_code}", flush=True)
        meta["error"] = f"exporter returned no payload (exit_code={exit_code})"
        return (f"\n[residual_rl] Video feedback: exporter returned no payload (exit_code={exit_code}).\n", meta)
    print(
        "[residual_rl][video] exporter subprocess returned payload: "
        f"exit_code={getattr(p, 'exitcode', '?')} ok={bool(isinstance(payload, dict) and payload.get('ok'))}",
        flush=True,
    )

    if not isinstance(payload, dict) or not payload.get("ok"):
        meta["error"] = payload.get("err") if isinstance(payload, dict) else str(payload)
        return (f"\n[residual_rl] Video feedback generation failed: {meta['error']}\n", meta)

    mp4s = list(payload.get("mp4s") or [])
    meta["ok"] = True
    meta["mp4s"] = mp4s
    if not mp4s:
        meta["error"] = f"no mp4s produced (output_dir={str(out_dir)})"
        return (f"\n[residual_rl] Video feedback: {meta['error']}.\n", meta)

    # Gemini video-to-text reflection is required whenever gemini_enabled=True.
    gemini_payload = None
    if gemini_enabled:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or gemini_api_key):
            meta["ok"] = False
            meta["error"] = "Gemini video summarization required but no GEMINI_API_KEY/GOOGLE_API_KEY is configured"
            return (f"\n[residual_rl] Video feedback generation failed: {meta['error']}\n", meta)
        try:
            from kuafu.rl.video.gemini import summarize_videos_with_gemini
            from kuafu.rl.video.video_prompts import GEMINI_OUTPUT_INSTRUCTIONS

            prompt = (context_text or "").strip()
            if not prompt:
                prompt = (
                    "You are analyzing RL rollout videos from Isaac Lab. "
                    "Summarize what happens, whether the task seems achieved, "
                    "and the earliest failure mode you see."
                    + GEMINI_OUTPUT_INSTRUCTIONS
                )
            gemini_payload = summarize_videos_with_gemini(
                video_paths=[str(p) for p in mp4s],
                context_text=prompt,
                model=str(gemini_model),
                api_key=gemini_api_key,
            )
            meta["gemini"] = gemini_payload
        except Exception as e:
            gemini_payload = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            meta["gemini"] = gemini_payload

    # Prompt block: list cameras, then append the single Gemini summary.
    lines: list[str] = []
    for pth in mp4s:
        cam = Path(str(pth)).name.split("_grid_")[0]
        lines.append(f"- camera: {cam}")
        lines.append(f"  episode_id_or_seed: {int(seed)}")
        lines.append(f"  video: {pth}")

    combined = ""
    if isinstance(gemini_payload, dict) and gemini_payload.get("ok"):
        combined = str(gemini_payload.get("combined_text") or "").strip()
    elif gemini_enabled:
        meta["ok"] = False
        gemini_error = None
        if isinstance(gemini_payload, dict):
            gemini_error = gemini_payload.get("error")
        if not gemini_error:
            gemini_error = "unknown Gemini failure"
        meta["error"] = f"Gemini video summarization failed: {gemini_error}"
        return (f"\n[residual_rl] Video feedback generation failed: {meta['error']}\n", meta)
    if combined:
        indented = "\n".join("  " + ln for ln in combined.splitlines())
        lines.append("gemini_analysis: |")
        lines.append(indented if indented else "  (empty)")
    elif gemini_enabled:
        meta["ok"] = False
        meta["error"] = "Gemini video summarization returned empty text"
        return (f"\n[residual_rl] Video feedback generation failed: {meta['error']}\n", meta)

    return ("\n".join(lines) + "\n", meta)
