"""A bounded observe -> construct -> render -> critique -> revise loop."""

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from kuafu.paths import resolve_artifact_path

from .articulation import posed_asset
from .export import export_asset
from .geometry import build_asset
from .io import digest, local_file, write_json
from .model import ModelOutputError, OpenAIModel, VisionModel
from .prompts import BUILD_INSTRUCTIONS, REVIEW_INSTRUCTIONS
from .render import render_views
from .schema import PhysicsConfig, PreparedInput, Review, RunConfig, SceneSpec
from .semantics import require_agent_semantics

LOG = logging.getLogger(__name__)


def reconstruct(
    workspace: Path, config: RunConfig, physics: PhysicsConfig, model: VisionModel | None = None
) -> dict:
    workspace = resolve_artifact_path(workspace, "Asset reconstruction workspace")
    inputs = PreparedInput.model_validate_json((workspace / "input.json").read_text())
    frames = []
    for frame in inputs.frames:
        path = local_file(workspace, frame.path)
        if digest(path) != frame.sha256:
            raise ValueError(f"Prepared frame changed: {frame.path}; prepare a fresh workspace")
        frames.append(path)
    if model is None:
        model = OpenAIModel(config, workspace / "cache")
    state = {
        "status": "running",
        "config": config.model_dump(),
        "physics": physics.model_dump(),
        "input_sha256": digest(workspace / "input.json"),
        "iterations": [],
    }
    write_json(workspace / "run.json", state)
    prompt = (
        f"Object description:\n{inputs.description}\n"
        f"Required motion: {inputs.motion}. Additional clips show the same object or its removed parts.\n"
        f"Known overall dimension: {inputs.scale.model_dump() if inputs.scale else 'none; estimate scale'}\n"
        "Frames in image order (path, video index, seconds): "
        f"{[(f.path, f.source_index, round(f.time_s, 3)) for f in inputs.frames]}\n"
        f"Capture warnings: {inputs.warnings}"
    )
    previous, feedback, previous_images = None, "", []
    best = None
    try:
        for index in range(config.iterations):
            iteration = workspace / "iterations" / f"{index:02d}"
            iteration.mkdir(parents=True, exist_ok=True)
            record = {"index": index}
            state["iterations"].append(record)
            build_prompt = prompt
            if previous is not None:
                build_prompt += "\nPrevious candidate:\n" + previous.model_dump_json()
            if feedback:
                build_prompt += "\nFeedback to address:\n" + feedback
            LOG.info("Candidate %d/%d: inferring geometry", index + 1, config.iterations)
            try:
                spec = model.complete(SceneSpec, BUILD_INSTRUCTIONS, build_prompt, frames + previous_images)
                if inputs.motion == "articulated" and spec.articulation is None:
                    raise ValueError(
                        "Articulated motion was requested: supply separate links and moving joints"
                    )
                if inputs.motion == "rigid" and spec.articulation is not None:
                    raise ValueError("Rigid motion was requested: do not invent an articulation")
                require_agent_semantics(spec)
                previous = spec
                write_json(iteration / "scene.json", spec)
                asset = build_asset(spec, inputs.scale)
            except (ValidationError, ModelOutputError, ValueError) as error:
                feedback = "Geometry/schema validation failed: " + str(error)[:6000]
                record.update(status="invalid_geometry", error=feedback)
                write_json(workspace / "run.json", state)
                LOG.warning("%s", feedback)
                continue
            LOG.info("Candidate %d: rendering visuals and collisions", index + 1)
            render_views(asset, iteration)
            write_json(iteration / "geometry.json", asset.metadata)
            previous_images = [iteration / "preview.png", iteration / "collisions.png"]
            if spec.articulation:
                render_views(posed_asset(asset), iteration / "articulation_open")
                previous_images.append(iteration / "articulation_open/preview.png")
            # Keep an exportable candidate even if a later API request fails.
            if best is None:
                best = (float("-inf"), index, spec, asset, None)
            review_prompt = (
                prompt
                + "\nCandidate:\n"
                + spec.model_dump_json()
                + "\nGeometry metadata:\n"
                + json.dumps(asset.metadata)
            )
            try:
                review = model.complete(Review, REVIEW_INSTRUCTIONS, review_prompt, frames + previous_images)
            except (ValidationError, ModelOutputError) as error:
                feedback = "Review was invalid. Re-check shape and functional openings: " + str(error)[:3000]
                record.update(status="invalid_review", error=feedback)
                write_json(workspace / "run.json", state)
                continue
            write_json(iteration / "review.json", review)
            feedback = review.model_dump_json()
            record.update(status="reviewed", score=review.score, acceptable=review.acceptable)
            write_json(workspace / "run.json", state)
            LOG.info(
                "Candidate %d: visual score %.2f; %d issues", index + 1, review.score, len(review.issues)
            )
            if review.score > best[0]:
                best = (review.score, index, spec, asset, review)
            if review.acceptable and review.score >= config.target_score:
                # A candidate that clears both gates wins, including score ties.
                best = (review.score, index, spec, asset, review)
                break
    except Exception as error:
        state["error"] = f"{type(error).__name__}: {error}"
        if best is None:
            state["status"] = "failed"
            write_json(workspace / "run.json", state)
            raise
        LOG.warning("Agent stopped early; exporting the best available candidate: %s", error)
    if best is None:
        state["status"] = "failed"
        write_json(workspace / "run.json", state)
        raise RuntimeError(
            "No valid geometry was produced. Inspect iterations and cache, then reconstruct again."
        )
    _, index, spec, asset, review = best
    try:
        metadata = export_asset(asset, spec, workspace / "final", physics, inputs.scale)
    except Exception as error:
        state.update(status="failed", error=f"Export failed: {error}")
        write_json(workspace / "run.json", state)
        raise
    accepted = bool(review and review.acceptable and review.score >= config.target_score)
    state.update(
        status="complete" if accepted and "error" not in state else "needs_review",
        best_iteration=index,
        review=review.model_dump() if review else None,
        artifacts="final",
        metadata=metadata,
        api_calls=getattr(model, "api_calls", None),
        cache_hits=getattr(model, "cache_hits", None),
    )
    write_json(workspace / "run.json", state)
    return state
