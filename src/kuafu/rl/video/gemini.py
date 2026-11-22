"""Gemini-based video-to-text reflection.

This module is intentionally isolated from the OpenAI loop: it ONLY converts
rollout videos (mp4) into compact textual diagnostics that Kuafu can embed into
its existing text-only tuning prompts.

All camera videos for a single evaluation are sent in ONE Gemini request so the
model can reason about the scene holistically across camera angles.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from kuafu.common.prompt_context import text_for_model

_TAG = "[residual_rl] Gemini"
_FILE_POLL_INTERVAL = 5
_FILE_POLL_MAX_WAIT = 120


def _get_api_key(explicit: str | None) -> str | None:
    return explicit or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _make_display_name(video_path: str) -> str:
    """Build a unique Gemini display_name from the video path.

    e.g. ``…/videos/global_iter0_postbest/table_cam_0_grid_3x3.mp4``
    →  ``global_iter0_postbest__table_cam_0_grid_3x3.mp4``
    """
    p = Path(video_path)
    return f"{p.parent.name}__{p.name}"


def _wait_for_active(client, file_ref, timeout: float = _FILE_POLL_MAX_WAIT) -> bool:
    """Poll until an uploaded Gemini file reaches ACTIVE state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        f = client.files.get(name=file_ref.name)
        state = getattr(f, "state", None)
        if state is not None and str(state).upper().endswith("ACTIVE"):
            return True
        if state is not None and str(state).upper().endswith("FAILED"):
            return False
        time.sleep(_FILE_POLL_INTERVAL)
    return False


def _try_delete(client, remote_name: str) -> None:
    """Best-effort deletion of an uploaded Gemini file."""
    try:
        client.files.delete(name=remote_name)
        print(f"{_TAG} deleted remote file {remote_name}", flush=True)
    except Exception as exc:
        print(f"{_TAG} failed to delete {remote_name}: {exc}", flush=True)


def summarize_videos_with_gemini(
    *,
    video_paths: list[str],
    context_text: str,
    model: str = "gemini-2.0-flash",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Upload all camera videos, send ONE request, return JSON-serializable payload.

    Flow: upload all → poll all ACTIVE → single generate_content → cleanup all.
    """
    out: dict[str, Any] = {"model": str(model), "summaries": [], "combined_text": "", "ok": False, "error": None}

    key = _get_api_key(api_key)
    if not key:
        out["error"] = "missing GEMINI_API_KEY (or GOOGLE_API_KEY)"
        for p in video_paths:
            out["summaries"].append({"video_path": str(p), "ok": False, "text": "", "error": out["error"]})
        return out

    if not video_paths:
        out["error"] = "no video paths provided"
        return out

    uploaded: list[tuple[str, str, Any]] = []  # (local_path, display_name, file_ref)

    try:
        context_text = text_for_model(context_text)
        from google import genai  # type: ignore
        from google.genai import types as genai_types  # type: ignore

        client = genai.Client(api_key=key)

        # --- Phase 1: Upload all videos ---
        for idx, vp in enumerate(video_paths):
            display_name = _make_display_name(vp)
            print(f"{_TAG} [{idx + 1}/{len(video_paths)}] uploading {vp} (display_name={display_name}) ...", flush=True)
            f = client.files.upload(
                file=str(vp),
                config=genai_types.UploadFileConfig(display_name=display_name),
            )
            uploaded.append((str(vp), display_name, f))
            print(f"{_TAG} [{idx + 1}/{len(video_paths)}] uploaded → {f.name}", flush=True)

        # --- Phase 2: Wait for all to become ACTIVE ---
        print(f"{_TAG} waiting for all {len(uploaded)} files to become ACTIVE ...", flush=True)
        all_active = True
        for local_path, display_name, fref in uploaded:
            if not _wait_for_active(client, fref):
                print(f"{_TAG} TIMEOUT: {fref.name} ({display_name}) never reached ACTIVE", flush=True)
                all_active = False

        if not all_active:
            out["error"] = "one or more uploaded files never reached ACTIVE"
            for local_path, display_name, fref in uploaded:
                out["summaries"].append({"video_path": local_path, "ok": False, "text": "", "error": out["error"]})
            return out

        print(f"{_TAG} all files ACTIVE", flush=True)

        # --- Phase 3: Build contents and make ONE request ---
        video_listing = "\n".join(
            f"  - camera {idx + 1}/{len(uploaded)}: {dn} ({lp})"
            for idx, (lp, dn, _) in enumerate(uploaded)
        )
        prompt = (
            f"{context_text}\n\n"
            f"--- ATTACHED VIDEOS ({len(uploaded)} camera angles of the SAME rollout) ---\n"
            f"{video_listing}\n"
            f"The videos are attached in the same order as listed above.\n"
        )

        contents: list[Any] = [str(prompt)]
        for _, _, fref in uploaded:
            contents.append(fref)

        print(f"{_TAG} generating content with {model} ({len(uploaded)} videos in one request) ...", flush=True)
        resp = client.models.generate_content(model=str(model), contents=contents)
        text = getattr(resp, "text", None) or str(resp)
        text = str(text).strip()
        print(f"{_TAG} done ({len(text)} chars)", flush=True)

        out["ok"] = True
        out["combined_text"] = text
        for local_path, display_name, fref in uploaded:
            out["summaries"].append({"video_path": local_path, "ok": True, "text": text, "error": None})

    except Exception as e:
        print(f"{_TAG} ERROR: {type(e).__name__}: {e}", flush=True)
        out["error"] = f"{type(e).__name__}: {e}"
        for vp in video_paths:
            if not any(s["video_path"] == str(vp) for s in out["summaries"]):
                out["summaries"].append({"video_path": str(vp), "ok": False, "text": "", "error": out["error"]})

    finally:
        # --- Phase 4: Cleanup all uploaded files ---
        if uploaded:
            try:
                from google import genai as _g  # type: ignore
                cleanup_client = _g.Client(api_key=key)
                for _, _, fref in uploaded:
                    _try_delete(cleanup_client, fref.name)
            except Exception as exc:
                print(
                    f"{_TAG} WARNING: failed to clean up uploaded Gemini files: {type(exc).__name__}: {exc}",
                    flush=True,
                )

    return out
