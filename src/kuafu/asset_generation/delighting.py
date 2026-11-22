"""Local intrinsic estimation -> bounded lighting correction of original pixels.

The model supplies an albedo *guide*, not replacement texture pixels. A smooth
log-RGB gain transfers its estimated lighting removal to the original photograph.
Fine source detail survives the correction; unobserved detail is not recovered.
Torch/diffusers are optional and imported only when inference is requested.
"""

import json
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pydantic import Field

from .io import digest, fresh_directory, local_file, resolve_artifact_path, write_json
from .sam3d import polygon_mask, validate_mask
from .schema import PreparedInput, StrictModel

MODEL_ID = "prs-eth/marigold-iid-lighting-v1-1"
MODEL_REVISION = "08c3930bb641abf786ba44ce92547507ebefbc16"
ALGORITHM = "intrinsic_guided_log_gain_v1"
TARGETS = ("albedo", "shading", "residual")


class DelightConfig(StrictModel):
    steps: int = Field(default=4, ge=1, le=50)
    ensemble_size: int = Field(default=1, ge=1, le=8)
    processing_resolution: int = Field(default=768, ge=128, le=1536)
    seed: int = Field(default=42, ge=0, le=2**32 - 13)
    strength: float = Field(default=1.0, ge=0, le=1)
    max_gain: float = Field(default=16.0, ge=1, le=32)
    smooth_fraction: float = Field(default=0.004, gt=0, le=0.05)
    crop_margin: float = Field(default=0.08, ge=0, le=0.5)


def srgb_to_linear(rgb):
    """Normalized sRGB -> linear RGB; intrinsic model outputs are already linear."""
    rgb = np.asarray(rgb, np.float32)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    rgb = np.clip(np.asarray(rgb, np.float32), 0, 1)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def _rgb8(linear):
    return np.round(np.clip(linear_to_srgb(linear), 0, 1) * 255).astype(np.uint8)


def correct_lighting(image, albedo, mask, config: DelightConfig):
    """Apply a smooth, mask-normalized gain to source pixels in linear light.

    Smoothing the *correction*, not the photograph, preserves fine photographed
    grain/printing that the model may soften. The model remains responsible for
    separating large markings from lighting. This is estimated, not measured,
    albedo. Gain limits avoid unbounded amplification of deep-shadow noise.
    """
    image, albedo, mask = np.asarray(image), np.asarray(albedo), np.asarray(mask)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Lighting correction needs an RGB uint8 image")
    if (
        albedo.shape != image.shape
        or not np.isfinite(albedo).all()
        or np.any(albedo < 0)
        or np.any(albedo > 1)
    ):
        raise ValueError("Albedo must be finite linear RGB in [0, 1], matching the source image")
    if mask.shape != image.shape[:2] or mask.dtype != bool or not mask.any():
        raise ValueError("Lighting correction needs a nonempty boolean mask matching the image")
    source = srgb_to_linear(image / 255.0)
    # Epsilon stabilizes black ink and quantized deep shadows; no new nonzero
    # source detail is invented at black pixels.
    epsilon = 1 / 4096
    log_gain = np.log(np.maximum(albedo, epsilon)) - np.log(np.maximum(source, epsilon))
    sigma = max(0.5, min(mask.shape) * config.smooth_fraction)
    weight = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    log_gain = cv2.GaussianBlur(log_gain * mask[:, :, None], (0, 0), sigma)
    log_gain /= np.maximum(weight[:, :, None], 1e-6)
    gain = np.exp(np.clip(log_gain * config.strength, -20, 20))
    limited = (gain.max(axis=2) > config.max_gain) | (gain.min(axis=2) < 1 / config.max_gain)
    # Limit brightening jointly across RGB to avoid yellow/magenta clipped-gain
    # casts in deep colored shadows. Extreme attenuation is bounded as well.
    gain *= np.minimum(1, config.max_gain / np.maximum(gain.max(axis=2), 1e-6))[:, :, None]
    gain = np.clip(gain, 1 / config.max_gain, config.max_gain)
    corrected_linear = source * gain
    corrected = image.copy()
    corrected[mask] = _rgb8(corrected_linear)[mask]
    return corrected, {
        "object_pixels": int(mask.sum()),
        "smoothing_sigma_px": float(sigma),
        "gain_limited_fraction": float(limited[mask].mean()),
        "input_clipped_fraction": float((image.max(axis=2)[mask] >= 254).mean()),
        "input_dark_fraction": float((source.mean(axis=2)[mask] < 0.01).mean()),
        "output_clipped_fraction": float((corrected_linear.max(axis=2)[mask] > 1).mean()),
        "gain_quantiles": np.quantile(gain[mask], [0.01, 0.5, 0.99]).tolist(),
    }


def _crop_box(mask, margin):
    yy, xx = np.nonzero(mask)
    pad = max(8, round(max(np.ptp(xx) + 1, np.ptp(yy) + 1) * margin))
    return [
        max(0, int(xx.min()) - pad),
        max(0, int(yy.min()) - pad),
        min(mask.shape[1], int(xx.max()) + pad + 1),
        min(mask.shape[0], int(yy.max()) + pad + 1),
    ]


def _recipe_inputs(workspace, recipe_path):
    recipe = json.loads(recipe_path.read_text())
    entries = recipe.get("views", [])
    if recipe.get("version") != 1 or not 1 <= len(entries) <= 12:
        raise ValueError("Photo recipe needs version 1 and 1..12 registered views")
    if len({entry["frame"] for entry in entries}) != len(entries):
        raise ValueError("De-lighting requires unique source frames in the photo recipe")
    prepared = PreparedInput.model_validate_json((workspace / "input.json").read_text())
    frames = {frame.path: frame for frame in prepared.frames}
    inputs = []
    for entry in entries:
        path = local_file(workspace, entry["frame"])
        if entry["frame"] not in frames or digest(path) != frames[entry["frame"]].sha256:
            raise ValueError("Source frame changed or is absent from the prepared video manifest")
        with Image.open(path) as loaded:
            image = np.array(loaded.convert("RGB"))
        mask_path = local_file(workspace, entry["mask"])
        if mask_path.suffix.lower() == ".json":
            mask = polygon_mask((image.shape[1], image.shape[0]), json.loads(mask_path.read_text()))
        else:
            with Image.open(mask_path) as loaded:
                mask = np.array(loaded)
        validate_mask(mask, (image.shape[1], image.shape[0]))
        inputs.append(
            {
                "frame": entry["frame"],
                "frame_sha256": digest(path),
                "mask": entry["mask"],
                "mask_sha256": digest(mask_path),
                "image": image,
                "inside": mask > 0,
            }
        )
    return inputs


class MarigoldPredictor:
    """Pinned, public safetensors model; no remote Python code or image upload."""

    def __init__(self, config, device="auto", cache_dir=None, local_files_only=False):
        cache_dir = resolve_artifact_path(cache_dir or "assets/models/intrinsics", "Intrinsic model cache")
        try:
            import torch
            from diffusers import MarigoldIntrinsicsPipeline
        except ImportError as error:
            raise RuntimeError(
                "Install optional local inference dependencies: pip install -e '.[delight]'"
            ) from error
        self.torch, self.config = torch, config
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("De-lighting currently supports cpu or cuda[:index]")
        if self.device.type == "cuda" and (
            not torch.cuda.is_available() or (self.device.index or 0) >= torch.cuda.device_count()
        ):
            raise RuntimeError(
                "Requested CUDA device is not available; select --device cpu or an available GPU"
            )
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.pipe = MarigoldIntrinsicsPipeline.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            variant="fp16",
            torch_dtype=dtype,
            use_safetensors=True,
            token=False,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ).to(self.device)
        properties = self.pipe.target_properties
        if tuple(properties["target_names"]) != TARGETS or any(
            properties[name]["prediction_space"] != "linear" for name in TARGETS
        ):
            raise RuntimeError("Unexpected intrinsic model targets or color space")
        self.pipe.set_progress_bar_config(disable=True)
        self.info = {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": "CreativeML Open RAIL++-M",
            "model_card": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
            "device": str(self.device),
            "dtype": str(dtype),
            "runtime_versions": {
                name: version(name) for name in ("torch", "diffusers", "transformers", "huggingface-hub")
            },
        }

    def predict(self, image, seed):
        with self.torch.inference_mode():
            result = self.pipe(
                Image.fromarray(image),
                num_inference_steps=self.config.steps,
                ensemble_size=self.config.ensemble_size,
                processing_resolution=self.config.processing_resolution,
                match_input_resolution=True,
                generator=self.torch.Generator(self.device).manual_seed(seed),
                output_type="np",
            )
        prediction = result.prediction
        # Diffusers NumPy output flattens batch * targets, not [batch, targets, ...].
        if prediction.shape != (3, *image.shape):
            raise RuntimeError(f"Unexpected intrinsic prediction shape: {prediction.shape}")
        return dict(zip(TARGETS, prediction, strict=True))


def _comparison(path, source, corrected, prediction):
    panels = [source, corrected, *[_rgb8(prediction[name]) for name in TARGETS]]
    titles = [
        "Original video",
        "Corrected original pixels",
        "Model albedo guide",
        "Shading (relative)",
        "Residual (relative)",
    ]
    width = 300
    height = round(source.shape[0] * width / source.shape[1])
    sheet = Image.new("RGB", (width * len(panels), height + 30), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (panel, title) in enumerate(zip(panels, titles, strict=True)):
        sheet.paste(Image.fromarray(panel).resize((width, height)), (i * width, 30))
        draw.text((i * width + 8, 8), title, fill="black")
    sheet.save(path, quality=93)


def delight_frames(
    workspace: Path,
    recipe_path: Path,
    out: Path,
    config: DelightConfig | None = None,
    *,
    device="auto",
    cache_dir=None,
    local_files_only=False,
    predictor=None,
):
    """Create a reusable, hash-bound corrected-frame manifest; never alter inputs."""
    config = config or DelightConfig()
    workspace = workspace.resolve()
    input_hash, recipe_hash = digest(workspace / "input.json"), digest(recipe_path)
    inputs = _recipe_inputs(workspace, recipe_path)
    out = fresh_directory(out)  # Refuse overwrite before loading/downloading any model.
    predictor = predictor or MarigoldPredictor(config, device, cache_dir, local_files_only)
    views = []
    for index, entry in enumerate(inputs):
        print(f"De-lighting {entry['frame']} ({index + 1}/{len(inputs)}) ...", flush=True)
        image, inside = entry["image"], entry["inside"]
        box = _crop_box(inside, config.crop_margin)
        x0, y0, x1, y1 = box
        crop, crop_mask = image[y0:y1, x0:x1], inside[y0:y1, x0:x1]
        prediction = predictor.predict(crop, config.seed + index)
        if set(prediction) != set(TARGETS) or any(
            value.shape != crop.shape
            or not np.isfinite(value).all()
            or np.any(value < 0)
            or np.any(value > 1)
            for value in prediction.values()
        ):
            raise RuntimeError("Intrinsic predictions must be finite, aligned linear RGB in [0, 1]")
        corrected, report = correct_lighting(crop, prediction["albedo"], crop_mask, config)
        full_frame = image.copy()
        full_frame[y0:y1, x0:x1] = corrected
        relative = f"frame_{index:02d}.png"
        Image.fromarray(full_frame).save(out / relative)
        comparison = f"comparison_{index:02d}.jpg"
        _comparison(out / comparison, crop, corrected, prediction)
        intrinsic_file = f"intrinsics_{index:02d}.npz"
        np.savez_compressed(
            out / intrinsic_file, **{key: value.astype(np.float16) for key, value in prediction.items()}
        )
        views.append(
            {
                **{key: value for key, value in entry.items() if key not in {"image", "inside"}},
                "size": [image.shape[1], image.shape[0]],
                "crop_box": box,
                "seed": config.seed + index,
                "corrected": relative,
                "corrected_sha256": digest(out / relative),
                "comparison": comparison,
                "intrinsics": intrinsic_file,
                "metrics": report,
            }
        )
    manifest = {
        "version": 1,
        "status": "complete_needs_appearance_review",
        "algorithm": ALGORITHM,
        "model": predictor.info,
        "model_calls": len(views),
        "config": config.model_dump(),
        "input_sha256": input_hash,
        "recipe_sha256": recipe_hash,
        "color_spaces": {"source": "sRGB", "intrinsics": "linear RGB", "corrected": "sRGB"},
        "views": views,
        "files_sha256": {path.name: digest(path) for path in sorted(out.iterdir()) if path.is_file()},
        "warnings": [
            "Albedo and lighting are model estimates, not photometric measurements; "
            "colors can drift between views.",
            "Fine grain/printing is retained through source-pixel correction, "
            "but detail preservation is not guaranteed.",
            "Clipped highlights, deep shadows and unseen surfaces contain unrecoverable information.",
            "Gain limits can leave residual shading; fine lighting edges and camera noise can remain.",
            "Geometry, masks, camera registration, UV coverage and simulation validation "
            "are unchanged by this stage.",
        ],
    }
    if (
        digest(workspace / "input.json") != input_hash
        or digest(recipe_path) != recipe_hash
        or any(
            digest(local_file(workspace, entry[key])) != entry[f"{key}_sha256"]
            for entry in inputs
            for key in ("frame", "mask")
        )
    ):
        raise ValueError("Inputs changed during de-lighting; refusing to publish a reusable manifest")
    # No consumable manifest exists until every view and diagnostic has succeeded.
    write_json(out / "manifest.json", manifest)
    return out / "manifest.json"


def load_delighting(workspace: Path, recipe_path: Path, manifest_path: Path):
    """Verify source identity, complete coverage, output hashes and pixel alignment."""
    workspace, manifest_path = workspace.resolve(), manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("version") != 1
        or manifest.get("algorithm") != ALGORITHM
        or manifest.get("status") != "complete_needs_appearance_review"
    ):
        raise ValueError("Unsupported or incomplete de-lighting manifest")
    DelightConfig.model_validate(manifest["config"])
    if manifest.get("input_sha256") != digest(workspace / "input.json") or manifest.get(
        "recipe_sha256"
    ) != digest(recipe_path):
        raise ValueError("De-lighting manifest belongs to different or changed inputs/recipe")
    inputs = _recipe_inputs(workspace, recipe_path)
    views = manifest.get("views", [])
    if len(views) != len(inputs) or manifest.get("model_calls") != len(views):
        raise ValueError("De-lighting manifest must cover every recipe view exactly once")
    for relative, expected in manifest["files_sha256"].items():
        if digest(local_file(manifest_path.parent, relative)) != expected:
            raise ValueError(f"Changed de-lighting output: {relative}")
    images = {}
    for source, view in zip(inputs, views, strict=True):
        if any(view.get(key) != source[key] for key in ("frame", "frame_sha256", "mask", "mask_sha256")):
            raise ValueError("De-lighting frame/mask changed or view order does not match the recipe")
        path = local_file(manifest_path.parent, view["corrected"])
        if (
            digest(path) != view["corrected_sha256"]
            or manifest["files_sha256"].get(view["corrected"]) != view["corrected_sha256"]
        ):
            raise ValueError("Corrected-frame hash does not match the de-lighting manifest")
        with Image.open(path) as loaded:
            if loaded.mode != "RGB":
                raise ValueError("Corrected frames must be RGB images")
            image = np.array(loaded)
        original = source["image"]
        if image.shape != original.shape or view["size"] != [original.shape[1], original.shape[0]]:
            raise ValueError("Corrected frame dimensions no longer match the original video")
        if not np.array_equal(image[~source["inside"]], original[~source["inside"]]):
            raise ValueError("De-lighting changed pixels outside the reviewed object mask")
        images[source["frame"]] = image
    return images, manifest
