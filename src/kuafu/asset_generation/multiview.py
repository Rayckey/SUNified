"""Expand reviewed camera anchors into additional, quality-gated video views.

No appearance is generated here. Sparse forward/backward optical flow tracks
landmarks through every source frame; dense backward flow transports masks.
New masks and registrations remain estimates and require visual inspection.
"""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pydantic import Field

from .geometry import build_asset
from .io import digest, fresh_directory, local_file, write_json
from .phototexture import depth_buffer, fit_camera, read_mask
from .schema import PreparedInput, SceneSpec, StrictModel
from .video import contact_sheet


class TrackConfig(StrictModel):
    max_views: int = Field(default=12, ge=2, le=12)
    max_fb_error_px: float = Field(default=2.0, gt=0, le=10)
    max_focal_ratio: float = Field(default=1.5, ge=1, le=3)
    min_mask_iou: float = Field(default=0.72, gt=0, le=1)
    max_decoded_frames: int = Field(default=600, ge=2, le=2000)
    max_decoded_megapixels: int = Field(default=600, ge=1, le=2000)


def track_points(previous, current, points, valid, max_fb_error_px=2.0):
    """Bidirectional LK; a lost landmark cannot silently become valid again."""
    params = dict(
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.001),
        minEigThreshold=1e-6,
    )
    points = np.asarray(points, np.float32).reshape(-1, 1, 2)
    forward, forward_ok, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **params)
    if forward is None:
        return points[:, 0], np.zeros(len(points), bool)
    backward, backward_ok, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, **params)
    if backward is None:
        return forward[:, 0], np.zeros(len(points), bool)
    error = np.linalg.norm(backward[:, 0] - points[:, 0], axis=1)
    h, w = current.shape
    xy = forward[:, 0]
    keep = (
        np.asarray(valid, bool)
        & forward_ok.ravel().astype(bool)
        & backward_ok.ravel().astype(bool)
        & (error < max_fb_error_px)
        & np.isfinite(xy).all(axis=1)
        & (xy >= 1).all(axis=1)
        & (xy < [w - 1, h - 1]).all(axis=1)
    )
    # Invalid non-finite points must not poison the next OpenCV call.
    return np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0), keep


def camera_rejection(camera, seed_camera, max_rms, config):
    if camera["reprojection_rms_px"] > max_rms:
        return "reprojection_error"
    ratio = camera["focal_px"] / seed_camera["focal_px"]
    if not 1 / config.max_focal_ratio <= ratio <= config.max_focal_ratio:
        return "focal_drift"
    return None


def warp_masks(masks, backward_flow):
    """Use current-to-previous flow for nearest-neighbor binary-mask transport."""
    h, w = backward_flow.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    return {
        key: cv2.remap(
            mask,
            xx + backward_flow[..., 0],
            yy + backward_flow[..., 1],
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        for key, mask in masks.items()
    }


def refine_mask(image, propagated):
    """Refine a narrow uncertain silhouette band; never invent a new object."""
    kernel = np.ones((9, 9), np.uint8)
    inner = cv2.erode(propagated, kernel) > 0
    outer = cv2.dilate(propagated, kernel) > 0
    if inner.sum() < 64 or outer.all():
        raise ValueError("Tracked mask lost its foreground/background seeds")
    labels = np.full(propagated.shape, cv2.GC_BGD, np.uint8)
    labels[outer] = cv2.GC_PR_BGD
    labels[propagated > 0] = cv2.GC_PR_FGD
    labels[inner] = cv2.GC_FGD
    cv2.setRNGSeed(0)
    cv2.grabCut(image, labels, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
    return (np.isin(labels, [cv2.GC_FGD, cv2.GC_PR_FGD]) * 255).astype(np.uint8)


def _decode(video, prepared, size, config):
    if prepared.frame_count > config.max_decoded_frames:
        raise ValueError("Video exceeds the bounded tracking frame limit; use a shorter clip")
    if prepared.frame_count * size[0] * size[1] > config.max_decoded_megapixels * 1_000_000:
        raise ValueError("Video exceeds the bounded tracking RAM limit; prepare smaller frames")
    result = []
    capture = cv2.VideoCapture(str(video))
    try:
        for _ in range(prepared.frame_count):
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Could not decode every source frame needed for continuous tracking")
            if prepared.crop is not None:
                h, w = frame.shape[:2]
                x0, y0, x1, y1 = prepared.crop
                frame = frame[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
            result.append(cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()
    return result


def expand_photo_views(
    workspace: Path,
    recipe_path: Path,
    out: Path,
    config=None,
    *,
    extra_anchors: Path | None = None,
    video: Path | None = None,
):
    config = config or TrackConfig()
    workspace, out = workspace.resolve(), out.resolve()
    if not out.is_relative_to(workspace) or out == workspace:
        raise ValueError("Tracking output must be a new child of the workspace")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Output must be a new or empty directory")
    recipe = json.loads(recipe_path.read_text())
    if recipe.get("version") != 1 or not recipe.get("views"):
        raise ValueError("Expected a version-1 recipe with reviewed seed views")
    seeds = copy.deepcopy(recipe["views"])
    if extra_anchors is not None:
        extra = json.loads(extra_anchors.read_text())
        if extra.get("version") != 1:
            raise ValueError("Extra anchors must use version 1")
        seeds.extend(extra["views"])
    seed_paths = [entry["frame"] for entry in seeds]
    if len(set(seed_paths)) != len(seeds) or not 1 <= len(seeds) <= config.max_views:
        raise ValueError("Use unique seed frames, no more than max_views")
    prepared_path = workspace / "input.json"
    prepared_hash = digest(prepared_path)
    prepared = PreparedInput.model_validate_json(prepared_path.read_text())
    if prepared.reference_videos:
        raise ValueError("Track each video in its own prepared workspace; optical flow cannot cross clips")
    video = (video or Path(prepared.source_video)).resolve()
    if not video.is_file() or digest(video) != prepared.source_sha256:
        raise ValueError("Tracking requires the unchanged original video (or a hash-identical --video)")
    frames = {f.path: f for f in prepared.frames}
    size = None
    for frame in prepared.frames:
        path = local_file(workspace, frame.path)
        if digest(path) != frame.sha256:
            raise ValueError("Prepared frame pixels changed")
        with Image.open(path) as image:
            if size is not None and size != image.size:
                raise ValueError("Prepared frame dimensions must agree")
            size = image.size
    scene_path = local_file(workspace, recipe["scene"])
    asset = build_asset(SceneSpec.model_validate_json(scene_path.read_text()))
    mesh = asset.mesh()
    inputs = {str(recipe_path.resolve()): digest(recipe_path), str(scene_path): digest(scene_path)}
    if extra_anchors:
        inputs[str(extra_anchors.resolve())] = digest(extra_anchors)
    seed_data = []
    for entry in seeds:
        if entry["frame"] not in frames:
            raise ValueError("Seed frame absent from prepared manifest")
        xyz = np.asarray([p["xyz"] for p in entry["landmarks"]], float)
        points = xyz * asset.metadata["scale_factor"] + asset.metadata["translation_m"]
        pixels = np.asarray([p["uv"] for p in entry["landmarks"]], np.float32)
        camera = fit_camera(points, pixels, size)
        if camera["reprojection_rms_px"] > recipe.get("max_reprojection_rms_px", 12):
            raise ValueError(f"Seed camera fit too poor: {entry['frame']}")
        annotations = {
            "object": entry["mask"],
            **{f"region_{k}": v for k, v in entry.get("regions", {}).items()},
        }
        annotations.update(
            {
                f"occluder_{i}": item["mask"] if "mask" in item else {"include": [item["polygon"]]}
                for i, item in enumerate(entry.get("occluders", []))
            }
        )
        masks = {k: read_mask(workspace, v, size) for k, v in annotations.items()}
        for value in annotations.values():
            if isinstance(value, str):
                path = local_file(workspace, value)
                inputs[str(path)] = digest(path)
        seed_data.append(dict(entry=entry, points=points, xyz=xyz, pixels=pixels, camera=camera, masks=masks))
    # Validate all inputs before decoding or creating output.
    gray = _decode(video, prepared, size, config)
    out = fresh_directory(out)
    candidates, rejected = {}, []
    targets = {f.index: f for f in prepared.frames if f.path not in seed_paths}
    for seed_id, seed in enumerate(seed_data):
        start = frames[seed["entry"]["frame"]].index
        print(f"Tracking anchor {seed['entry']['frame']} ...", flush=True)
        for step in (-1, 1):
            xy, valid = seed["pixels"].copy(), np.ones(len(seed["pixels"]), bool)
            for index in range(start + step, len(gray) if step == 1 else -1, step):
                xy, valid = track_points(gray[index - step], gray[index], xy, valid, config.max_fb_error_px)
                if valid.sum() < 6:
                    rejected.append(
                        dict(seed=seed["entry"]["frame"], source_index=index, reason="lost_landmarks")
                    )
                    break
                if index not in targets:
                    continue
                frame = targets[index]
                try:
                    camera = fit_camera(seed["points"][valid], xy[valid], size)
                    reason = camera_rejection(
                        camera, seed["camera"], recipe.get("max_reprojection_rms_px", 12), config
                    )
                except ValueError:
                    reason, camera = "degenerate_camera", None
                if reason:
                    rejected.append(
                        dict(
                            frame=frame.path,
                            seed=seed["entry"]["frame"],
                            reason=reason,
                            rms_px=None if camera is None else camera["reprojection_rms_px"],
                        )
                    )
                    continue
                quality = camera["reprojection_rms_px"] + 0.025 * abs(index - start)
                if frame.path not in candidates or quality < candidates[frame.path]["quality"]:
                    candidates[frame.path] = dict(
                        frame=frame,
                        camera=camera,
                        seed_id=seed_id,
                        quality=quality,
                        xyz=seed["xyz"][valid],
                        xy=xy[valid],
                    )

    # Farthest-camera-direction sampling adds angular diversity, not just frame count.
    def direction(camera):
        d = np.asarray(camera["camera_center"]) - mesh.centroid
        return d / np.linalg.norm(d)

    selected = []
    directions = [direction(s["camera"]) for s in seed_data]
    while candidates and len(selected) + len(seeds) < config.max_views:

        def priority(c):
            novelty = min(1 - np.dot(direction(c["camera"]), d) for d in directions)
            return novelty / (1 + c["quality"] / 12)

        best = max(candidates.values(), key=priority)
        del candidates[best["frame"].path]
        selected.append(best)
        directions.append(direction(best["camera"]))
    entries = [dict(s, registration_source="reviewed_landmarks") for s in seeds]
    accepted_reports = []
    flow_engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    for seed_id, seed in enumerate(seed_data):
        start = frames[seed["entry"]["frame"]].index
        for step in (-1, 1):
            pending = {
                c["frame"].index: c
                for c in selected
                if c["seed_id"] == seed_id and (c["frame"].index - start) * step > 0
            }
            if not pending:
                continue
            stop = max(pending) if step == 1 else min(pending)
            masks = {k: v.copy() for k, v in seed["masks"].items()}
            for index in range(start + step, stop + step, step):
                flow = flow_engine.calc(gray[index], gray[index - step], None)
                masks = warp_masks(masks, flow)
                if index not in pending:
                    continue
                candidate = pending[index]
                frame = candidate["frame"]
                rgb = np.array(Image.open(workspace / frame.path).convert("RGB"))
                refined = refine_mask(rgb, masks["object"])
                silhouette = np.isfinite(depth_buffer(mesh, candidate["camera"], size))
                intersection = np.sum(silhouette & (refined > 0))
                iou = float(intersection / max(1, np.sum(silhouette | (refined > 0))))
                if iou < config.min_mask_iou:
                    rejected.append(dict(frame=frame.path, reason="mask_camera_disagreement", mask_iou=iou))
                    continue
                folder = out / Path(frame.path).stem
                folder.mkdir()
                saved = {}
                for name, mask in {**masks, "object": refined}.items():
                    # Fail closed if a required material region disappears.
                    if np.count_nonzero(mask) < 64:
                        raise ValueError(f"Tracked material region disappeared: {frame.path}/{name}")
                    path = folder / f"{name}.png"
                    Image.fromarray(mask).save(path)
                    saved[name] = path.relative_to(workspace).as_posix()
                entry = dict(
                    frame=frame.path,
                    mask=saved["object"],
                    landmarks=[
                        dict(xyz=p.tolist(), uv=q.tolist())
                        for p, q in zip(candidate["xyz"], candidate["xy"], strict=True)
                    ],
                    registration_source="forward_backward_lk_from_reviewed_anchor",
                    tracking_seed=seed["entry"]["frame"],
                    regions={name: saved[f"region_{name}"] for name in seed["entry"].get("regions", {})},
                    occluders=[
                        dict(mask=saved[f"occluder_{i}"], parts=item["parts"])
                        for i, item in enumerate(seed["entry"].get("occluders", []))
                    ],
                )
                entries.append(entry)
                accepted_reports.append(
                    dict(frame=frame.path, seed=entry["tracking_seed"], mask_iou=iou, **candidate["camera"])
                )
                overlay = rgb.copy()
                for binary, color in ((refined > 0, (255, 80, 20)), (silhouette, (20, 240, 230))):
                    contours, _ = cv2.findContours(
                        binary.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(overlay, contours, -1, color, 2)
                for xy in candidate["xy"]:
                    cv2.circle(overlay, tuple(np.rint(xy).astype(int)), 4, (30, 255, 30), -1)
                Image.fromarray(overlay).save(folder / "registration.jpg", quality=92)
                print(
                    f"Accepted {frame.path}: RMS {candidate['camera']['reprojection_rms_px']:.2f}px, "
                    f"mask IoU {iou:.3f}",
                    flush=True,
                )
    if digest(prepared_path) != prepared_hash or any(digest(Path(p)) != h for p, h in inputs.items()):
        raise ValueError("An input changed during tracking; refusing to publish a recipe")
    if digest(video) != prepared.source_sha256 or any(
        digest(workspace / f.path) != f.sha256 for f in prepared.frames
    ):
        raise ValueError("Video/frame pixels changed during tracking")
    entries.sort(key=lambda entry: frames[entry["frame"]].index)
    recipe["views"] = entries
    recipe["multiview"] = dict(
        algorithm="bidirectional_lk_dis_masks_v1",
        seed_count=len(seeds),
        tracked_count=len(entries) - len(seeds),
        review_required=True,
    )
    write_json(out / "recipe.json", recipe)
    write_json(
        out / "report.json",
        dict(
            version=1,
            config=config.model_dump(),
            input_sha256=prepared_hash,
            source_video_sha256=prepared.source_sha256,
            inputs=inputs,
            seed_count=len(seeds),
            registered_views=len(entries),
            accepted=accepted_reports,
            rejected=rejected,
            status="needs_registration_and_mask_review",
            model_calls=0,
            warnings=[
                "Optical flow can drift despite passing numeric gates; review every overlay.",
                "Extra views do not reveal surfaces absent from the source video.",
            ],
        ),
    )
    previews = sorted(out.glob("frame_*/registration.jpg"))
    if previews:
        contact_sheet(previews, [p.parent.name for p in previews], out / "registrations.jpg")
    return out / "recipe.json"
