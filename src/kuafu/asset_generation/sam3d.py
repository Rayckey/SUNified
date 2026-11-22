"""Masked video frames -> isolated SAM 3D worker -> reviewable textured GLB.

SAM 3D Objects is a single-image model, not a multi-view video solver. Language
guides object/frame/mask selection; it is recorded here, not passed as a text prompt
to a model that doesn't accept one. No procedural fallback is labelled SAM 3D.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .io import digest, fresh_directory, local_file, resolve_artifact_path, write_json
from .schema import PreparedInput

DEFAULT_ROOT = resolve_artifact_path(os.getenv("SAM3D_ROOT") or "assets/sam3d", "SAM 3D runtime")


def polygon_mask(size: tuple[int, int], annotation: dict) -> np.ndarray:
    """Rasterize pixel-coordinate include/exclude polygons; never infer a mask silently."""
    if set(annotation) - {"include", "exclude", "notes"} or not annotation.get("include"):
        raise ValueError("Mask annotation needs include polygons and optional exclude polygons/notes")
    width, height = size
    mask = np.zeros((height, width), np.uint8)
    for key, value in (("include", 255), ("exclude", 0)):
        for polygon in annotation.get(key, []):
            points = np.asarray(polygon, dtype=float)
            if (
                points.ndim != 2
                or points.shape[1] != 2
                or len(points) < 3
                or not np.isfinite(points).all()
                or np.any(points < 0)
                or np.any(points >= [width, height])
            ):
                raise ValueError("Polygons need at least 3 finite, in-image [x, y] points")
            cv2.fillPoly(mask, [np.round(points).astype(np.int32)], value)
    return mask


def validate_mask(mask: np.ndarray, size: tuple[int, int]) -> None:
    if mask.shape != (size[1], size[0]):
        raise ValueError("Mask must match the selected frame's height and width exactly")
    if not set(np.unique(mask)).issubset({0, 255}):
        raise ValueError("Mask must be binary: 0 background, 255 object; do not supply a color overlay")
    if np.count_nonzero(mask) < 64 or np.all(mask):
        raise ValueError("Mask must contain an object (at least 64 pixels) and some background")


def prepare_sam3d(workspace: Path, frame: int, mask_path: Path, out: Path) -> Path:
    workspace = workspace.resolve()
    prepared = PreparedInput.model_validate_json((workspace / "input.json").read_text())
    if not 0 <= frame < len(prepared.frames):
        raise ValueError(f"--frame is a zero-based selected-frame ordinal: 0..{len(prepared.frames) - 1}")
    selected = prepared.frames[frame]
    source = local_file(workspace, selected.path)
    if digest(source) != selected.sha256:
        raise ValueError("Selected frame changed after video preparation; prepare it again")
    with Image.open(source) as loaded:
        rgb = loaded.convert("RGB")
    annotation = None
    if mask_path.suffix.lower() == ".json":
        annotation = json.loads(mask_path.read_text())
        mask = polygon_mask(rgb.size, annotation)
    else:
        with Image.open(mask_path) as loaded:
            if loaded.mode not in ("1", "L"):
                raise ValueError("Supply a single-channel binary mask PNG, not an RGB image")
            mask = np.array(loaded.convert("L"))
    validate_mask(mask, rgb.size)
    out = fresh_directory(out)
    rgb.save(out / "image.png")
    Image.fromarray(mask).save(out / "mask.png")
    rgba = rgb.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask))
    rgba.save(out / "cutout.png")
    overlay = np.array(rgb).copy()
    inside = mask > 0
    overlay[inside] = (0.7 * overlay[inside] + 0.3 * np.array([45, 230, 125])).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 35, 65), 2)
    Image.fromarray(overlay).save(out / "mask_review.jpg", quality=92)
    if annotation is not None:
        write_json(out / "mask_annotation.json", annotation)
    else:
        shutil.copy2(mask_path, out / "source_mask.png")
    job = {
        "version": 1,
        "backend": "facebook/sam-3d-objects",
        "image": "image.png",
        "image_sha256": digest(out / "image.png"),
        "mask": "mask.png",
        "mask_sha256": digest(out / "mask.png"),
        "description": prepared.description,
        "language_role": "object selection and later review; SAM 3D has no text-prompt input",
        "source_video": str(prepared.source_video),
        "source_video_sha256": prepared.source_sha256,
        "source_frame": selected.model_dump(mode="json"),
        "selected_frame_ordinal": frame,
        "prepared_input_sha256": digest(workspace / "input.json"),
        "mask_source": "manual_polygon" if annotation is not None else "supplied_binary_mask",
        "status": "prepared_not_inferred",
        "scale_basis": "uncalibrated; preserve estimated scene scale during simulation import",
    }
    write_json(out / "job.json", job)
    return out / "job.json"


def validate_job(job_path: Path) -> dict:
    job = json.loads(job_path.read_text())
    if job.get("version") != 1 or job.get("backend") != "facebook/sam-3d-objects":
        raise ValueError("Not a supported SAM 3D job")
    for field in ("image", "mask"):
        path = local_file(job_path.parent, job[field])
        if digest(path) != job[f"{field}_sha256"]:
            raise ValueError(f"SAM 3D {field} changed after preparation")
    with Image.open(local_file(job_path.parent, job["image"])) as image:
        size = image.size
    with Image.open(local_file(job_path.parent, job["mask"])) as image:
        validate_mask(np.asarray(image), size)
    return job


def worker_environment(root: Path, gpu: int) -> dict:
    if gpu < 0:
        raise ValueError("GPU index must be non-negative")
    root = resolve_artifact_path(root, "SAM 3D runtime")
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(key, None)
    environment.update(
        CONDA_PREFIX=str(root / "env"),
        CUDA_HOME=str(root / "env"),
        PATH=str(root / "env/bin") + os.pathsep + environment.get("PATH", ""),
        PYTHONPATH=str(root / "repo"),
        CUDA_VISIBLE_DEVICES=str(gpu),
        LIDRA_SKIP_INIT="true",
        ATTN_BACKEND="xformers",
        SPARSE_ATTN_BACKEND="xformers",
        SPARSE_BACKEND="spconv",
        HF_HUB_CACHE=str(root / "cache/huggingface"),
        TORCH_HOME=str(root / "cache/torch"),
        TORCH_EXTENSIONS_DIR=str(root / "cache/extensions"),
        XDG_CACHE_HOME=str(root / "cache/xdg"),
        TMPDIR=str(root / "tmp"),
        MAX_JOBS="4",
        CPATH=str(root / "env/targets/x86_64-linux/include"),
        LIBRARY_PATH=str(root / "env/targets/x86_64-linux/lib"),
        TORCH_CUDA_ARCH_LIST="7.5",
    )
    # Keep HF_HOME and HF_TOKEN untouched: normal local authentication must still work.
    return environment


def worker_command(root: Path) -> list[str]:
    python = root.resolve() / "env/bin/python"
    if not python.is_file() or not (root / "repo/sam3d_objects").is_dir():
        raise RuntimeError("SAM 3D runtime is missing; run scripts/assets/setup_sam3d.sh first")
    return [str(python), str(Path(__file__).with_name("sam3d_worker.py")), "--root", str(root.resolve())]


def sam3d_maintenance(root: Path, action: str, gpu: int = 0, kernel_test: bool = False) -> int:
    command = worker_command(root) + [f"--{action}"]
    if kernel_test:
        command.append("--kernel-test")
    return subprocess.run(command, env=worker_environment(root, gpu), check=False).returncode


def run_sam3d(
    job_path: Path,
    out: Path,
    root: Path,
    gpu: int,
    mask_reviewed: bool,
    seed: int = 42,
    config: Path | None = None,
    timeout_s: int = 3600,
) -> Path:
    if not mask_reviewed:
        raise ValueError("Inspect mask_review.jpg/cutout.png, then pass --mask-reviewed")
    if timeout_s <= 0:
        raise ValueError("Timeout must be positive")
    job_path = job_path.resolve()
    validate_job(job_path)
    config = (config or root / "models/sam3d/checkpoints/pipeline.yaml").resolve()
    if not config.is_file():
        raise RuntimeError(
            "SAM 3D checkpoints are unavailable. Request access to facebook/sam-3d-objects, "
            "authenticate locally with `hf auth login`, then run `kuafu assets sam3d-fetch`. "
            "Never paste a token into chat. Existing approved weights can be supplied with --config."
        )
    command = worker_command(root) + ["--job", str(job_path), "--config", str(config), "--seed", str(seed)]
    out = fresh_directory(out)
    command.extend(["--out", str(out)])
    state = {
        "backend": "facebook/sam-3d-objects",
        "job_sha256": digest(job_path),
        "mask_reviewed": True,
        "status": "running",
        "seed": seed,
    }
    write_json(out / "run.json", state)
    try:
        with (out / "inference.log").open("w") as log:
            result = subprocess.run(
                command,
                env=worker_environment(root, gpu),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"SAM 3D failed (exit {result.returncode}); see {out / 'inference.log'}")
        mesh = out / "visual.glb"
        if not mesh.is_file() or not (out / "provenance.json").is_file():
            raise RuntimeError("SAM 3D worker did not produce a verified textured mesh")
        state.update(status="needs_visual_and_alignment_review", visual_sha256=digest(mesh))
        write_json(out / "run.json", state)
        return mesh
    except Exception as error:
        state.update(status="failed", error=str(error))
        write_json(out / "run.json", state)
        raise
