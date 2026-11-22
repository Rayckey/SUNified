"""Ungated video-pixel texture baking onto an existing approximate rigid mesh.

Reviewed 2D/3D landmarks fit each camera. A mesh depth buffer rejects occluded
surfaces, and masked video pixels are projected into an xatlas UV atlas. Unseen
texels remain explicitly marked, not described as recovered texture or albedo.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import distance_transform_edt, find_objects, label
from scipy.optimize import least_squares

from .delighting import load_delighting
from .geometry import build_asset
from .io import digest, fresh_directory, local_file, write_json
from .sam3d import polygon_mask, validate_mask
from .schema import PhysicsConfig, PreparedInput, SceneSpec
from .textured import build_textured_asset

DEFAULT_FUSION = "robust"


def _validate_fusion(fusion, view_count):
    if fusion not in {"best", "robust"}:
        raise ValueError("Fusion must be 'robust' or 'best'")
    if fusion == "robust" and view_count < 3:
        raise ValueError(
            "Robust fusion requires at least three registered views; run track-photo-views "
            "or explicitly use --fusion best for legacy one/two-view recipes"
        )


def project(points, camera):
    rotation = np.asarray(camera["rotation"])
    transformed = np.asarray(points) @ rotation.T + np.asarray(camera["translation"])
    z = transformed[:, 2]
    pixels = transformed[:, :2] / np.maximum(z[:, None], 1e-9)
    pixels = pixels * camera["focal_px"] + np.asarray(camera["principal_point"])
    return pixels, z


def fit_camera(points, pixels, size):
    points, pixels = np.asarray(points, float), np.asarray(pixels, float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 6
        or pixels.shape != (len(points), 2)
        or not np.isfinite(points).all()
        or not np.isfinite(pixels).all()
        or np.linalg.matrix_rank(points - points.mean(axis=0)) < 3
    ):
        raise ValueError("Camera needs 6+ finite, non-coplanar 3D landmarks and matching image pixels")
    width, height = size
    if np.any(pixels < 0) or np.any(pixels >= [width, height]):
        raise ValueError("Camera landmarks must lie inside the selected image")
    principal = np.asarray([width / 2, height / 2])
    longest = max(size)

    def residual(parameters):
        rvec, tvec, focal = parameters[:3], parameters[3:6], np.exp(parameters[6])
        rotation, _ = cv2.Rodrigues(rvec)
        camera = {"rotation": rotation, "translation": tvec, "focal_px": focal, "principal_point": principal}
        prediction, depth = project(points, camera)
        errors = (prediction - pixels).ravel()
        return np.concatenate([errors, np.minimum(depth, 0) * 1e5])

    candidates = []
    for focal in (0.5 * longest, longest, 2 * longest, 4 * longest):
        intrinsic = np.array([[focal, 0, principal[0]], [0, focal, principal[1]], [0, 0, 1.0]])
        try:
            ok, rvec, tvec = cv2.solvePnP(points, pixels, intrinsic, None, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            continue
        if not ok:
            continue
        initial = np.r_[rvec.ravel(), tvec.ravel(), np.log(focal)]
        solution = least_squares(
            residual,
            initial,
            bounds=(np.r_[[-np.inf] * 6, np.log(longest * 0.25)], np.r_[[np.inf] * 6, np.log(longest * 8)]),
            max_nfev=1000,
        )
        rotation, _ = cv2.Rodrigues(solution.x[:3])
        camera = {
            "rotation": rotation.tolist(),
            "translation": solution.x[3:6].tolist(),
            "focal_px": float(np.exp(solution.x[6])),
            "principal_point": principal.tolist(),
            "camera_center": (-rotation.T @ solution.x[3:6]).tolist(),
        }
        prediction, depth = project(points, camera)
        if np.any(depth <= 0):
            continue
        error = np.linalg.norm(prediction - pixels, axis=1)
        camera.update(
            reprojection_rms_px=float(np.sqrt(np.mean(error**2))),
            reprojection_max_px=float(error.max()),
            projected_landmarks=prediction.tolist(),
        )
        candidates.append(camera)
    if not candidates:
        raise ValueError("Could not fit a camera in front of these landmarks")
    return min(candidates, key=lambda c: c["reprojection_rms_px"])


def triangle_pixels(triangle, width, height):
    """Pixel centers and barycentric coordinates inside one image/UV triangle."""
    a, b, c = np.asarray(triangle)
    x0, y0 = np.maximum(np.floor(np.min(triangle, axis=0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(np.max(triangle, axis=0)).astype(int), [width - 1, height - 1])
    denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(denominator) < 1e-10 or x1 < x0 or y1 < y0:
        return None
    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    x, y = xx + 0.5, yy + 0.5
    w0 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
    w1 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
    w2 = 1 - w0 - w1
    inside = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8)
    return yy[inside], xx[inside], np.column_stack([w0[inside], w1[inside], w2[inside]])


def depth_buffer(mesh, camera, size):
    width, height = size
    xy, depth = project(mesh.vertices, camera)
    result = np.full((height, width), np.inf, np.float32)
    for face in mesh.faces:
        if np.min(depth[face]) <= 0:
            continue
        pixels = triangle_pixels(xy[face], width, height)
        if pixels is None:
            continue
        yy, xx, barycentric = pixels
        z = 1 / np.sum(barycentric / depth[face], axis=1)
        result[yy, xx] = np.minimum(result[yy, xx], z)
    return result


def sample_color_guard(samples, guard):
    """Reject impossible cross-material samples; never recolor accepted video pixels."""
    values = np.asarray(samples, float)
    maximum, minimum = values.max(axis=1), values.min(axis=1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1)
    allowed = (saturation <= guard.get("max_saturation", 1)) & (maximum >= 255 * guard.get("min_value", 0))
    if "min_red_ratio" in guard:
        allowed &= values[:, 0] >= guard["min_red_ratio"] * values[:, 1:].max(axis=1)
    return allowed


def read_mask(workspace, annotation, size):
    """Read a validated polygon annotation or workspace-local binary mask file."""
    if isinstance(annotation, str):
        path = local_file(workspace, annotation)
        if path.suffix.lower() == ".json":
            annotation = json.loads(path.read_text())
        else:
            with Image.open(path) as source:
                mask = np.array(source)
            validate_mask(mask, size)
            return mask
    mask = polygon_mask(size, annotation)
    validate_mask(mask, size)
    return mask


def fuse_view_samples(colors, low_colors, scores):
    """Robust low-frequency fusion, retaining detail from one aligned source.

    Arrays are V x N x 3 sRGB (0..255), and V x N nonnegative scores.
    Require two agreeing observations. No averaging of lettering/grain detail.
    Unsupported texels remain unseen, to be handled by the explicit fill policy.
    """
    count = colors.shape[1]
    result = np.zeros((count, 3), np.uint8)
    source = np.full(count, -1, np.int16)
    for start in range(0, count, 32000):
        stop = min(count, start + 32000)
        score = scores[:, start:stop].copy()
        valid = score > 0
        use = np.flatnonzero(valid.sum(axis=0) >= 2)
        if not len(use):
            continue
        rgb = colors[:, start:stop][:, use].astype(np.float32)
        low = low_colors[:, start:stop][:, use]
        score = score[:, use]
        valid = valid[:, use]
        median = np.nanmedian(np.where(valid[..., None], low, np.nan), axis=0)
        # A broad color gate suppresses conflicting desk/highlight samples,
        # without treating dark printing as an outlier when views agree on it.
        # Compare BROAD color, not individual ink/grain pixels: imperfect
        # registration must not classify real high-frequency markings as dirt.
        inlier = valid & (np.max(np.abs(low - median), axis=2) <= 36)
        supported = inlier.sum(axis=0) >= 2
        score *= inlier
        best = np.argmax(score, axis=0)
        indices = np.arange(len(use))
        weights = score / np.maximum(score.sum(axis=0), 1e-20)
        broad = np.sum(low * weights[..., None], axis=0)
        detail = rgb[best, indices] - low[best, indices]
        selected = start + use[supported]
        result[selected] = np.rint(np.clip(broad + detail, 0, 255))[supported].astype(np.uint8)
        source[selected] = best[supported]
    return result, source


def bake_mesh(mesh, views, resolution, depth_tolerance, texel_density=None, *, fusion=DEFAULT_FUSION):
    _validate_fusion(fusion, len(views))
    try:
        import xatlas
    except ImportError as error:
        raise RuntimeError("Install the ungated UV library: pip install 'xatlas>=0.0.9,<1'") from error

    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    packing = xatlas.PackOptions()
    packing.resolution = 0
    packing.texels_per_unit = texel_density or resolution / float(mesh.extents.max())
    packing.padding = 4
    packing.rotate_charts = True
    atlas.generate(pack_options=packing)
    if atlas.atlas_count != 1:
        raise RuntimeError("UV mesh needs multiple atlas pages; refusing overlapping texture coordinates")
    mapping, faces, uv = atlas[0]
    baked = trimesh.Trimesh(vertices=mesh.vertices[mapping], faces=faces, process=False)
    # xatlas may split vertices, never change their positions or face order.
    if not np.allclose(baked.vertices[baked.faces], mesh.vertices[mesh.faces]):
        raise RuntimeError("UV unwrapping changed triangle geometry")
    width, height = atlas.width, atlas.height
    xy = np.column_stack([uv[:, 0] * width, (1 - uv[:, 1]) * height])
    points = np.zeros((height, width, 3), np.float32)
    normals = np.zeros_like(points)
    surface = np.zeros((height, width), bool)
    for face_index, face in enumerate(faces):
        pixels = triangle_pixels(xy[face], width, height)
        if pixels is None:
            continue
        yy, xx, weights = pixels
        points[yy, xx] = weights @ baked.vertices[face]
        normals[yy, xx] = baked.face_normals[face_index]
        surface[yy, xx] = True
    yy, xx = np.nonzero(surface)
    locations, directions = points[yy, xx], normals[yy, xx]
    base_rgb = np.asarray(mesh.visual.face_colors[0, :3], np.uint8)
    texture = np.full((height, width, 3), base_rgb, np.uint8)
    best_score = np.zeros(len(locations), np.float32)
    source_view = np.full((height, width), -1, np.int16)
    if fusion == "robust":
        all_colors = np.zeros((len(views), len(locations), 3), np.uint8)
        all_low = np.zeros((len(views), len(locations), 3), np.float32)
        all_scores = np.zeros((len(views), len(locations)), np.float32)
    for view_index, view in enumerate(views):
        projected, z = project(locations, view["camera"])
        px = np.round(projected[:, 0]).astype(int)
        py = np.round(projected[:, 1]).astype(int)
        image_height, image_width = view["image"].shape[:2]
        inside = (px >= 1) & (px < image_width - 1) & (py >= 1) & (py < image_height - 1) & (z > 0)
        px = np.clip(px, 0, image_width - 1)
        py = np.clip(py, 0, image_height - 1)
        direction = np.asarray(view["camera"]["camera_center"]) - locations
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        cosine = np.sum(directions * direction, axis=1)
        sample_mask = view["mask"].copy()
        part_name = mesh.metadata["name"]
        if part_name in view.get("regions", {}):
            sample_mask &= view["regions"][part_name]
        for occluder in view.get("occluders", []):
            if part_name not in occluder["parts"]:
                sample_mask &= ~occluder["mask"]
        visible = (np.abs(z - view["depth"][py, px]) < depth_tolerance) & sample_mask[py, px]
        score = np.maximum(cosine, 0) ** 4
        if fusion == "robust":
            rms = view["camera"].get("reprojection_rms_px", 0)
            margin = distance_transform_edt(sample_mask)
            score *= np.minimum(margin[py, px] / max(4, 2 * rms), 1) ** 2
            score /= 1 + (rms / 6) ** 2
            # Normalized filtering never brings background pixels into albedo.
            rows, cols = np.nonzero(sample_mask)
            sigma = max(2, min(np.ptp(rows), np.ptp(cols)) * 0.025) if len(rows) else 2
            weight = cv2.GaussianBlur(sample_mask.astype(np.float32), (0, 0), sigma)
            rgb = view.get("texture_image", view["image"]).astype(np.float32)
            broad = cv2.GaussianBlur(rgb * sample_mask[..., None], (0, 0), sigma)
            broad /= np.maximum(weight[..., None], 1e-6)
        accepted = inside & visible & (cosine > 0.15)
        if fusion == "best":
            accepted &= score > best_score
        indices = np.flatnonzero(accepted)
        if not len(indices):
            continue
        # Always test material guards on the original RGB. Lighting correction
        # must not silently change which views/texels pass the recipe's guards.
        for start in range(0, len(indices), 16000):
            selected = indices[start : start + 16000]
            samples = cv2.remap(
                view["image"],
                projected[selected, 0].astype(np.float32)[:, None],
                projected[selected, 1].astype(np.float32)[:, None],
                cv2.INTER_LINEAR,
            )
            valid = sample_color_guard(samples[:, 0], view.get("color_guards", {}).get(part_name, {}))
            if "texture_image" in view:
                samples = cv2.remap(
                    view["texture_image"],
                    projected[selected, 0].astype(np.float32)[:, None],
                    projected[selected, 1].astype(np.float32)[:, None],
                    cv2.INTER_LINEAR,
                )
            selected = selected[valid]
            if fusion == "robust":
                all_colors[view_index, selected] = samples[valid, 0]
                all_low[view_index, selected] = (
                    cv2.remap(
                        broad,
                        projected[selected, 0].astype(np.float32)[:, None],
                        projected[selected, 1].astype(np.float32)[:, None],
                        cv2.INTER_LINEAR,
                    )[:, 0]
                    if len(selected)
                    else np.empty((0, 3))
                )
                all_scores[view_index, selected] = score[selected]
            else:
                texture[yy[selected], xx[selected]] = samples[valid, 0]
                best_score[selected] = score[selected]
                source_view[yy[selected], xx[selected]] = view_index
    if fusion == "robust":
        fused, sources = fuse_view_samples(all_colors, all_low, all_scores)
        supported = sources >= 0
        texture[yy[supported], xx[supported]] = fused[supported]
        source_view[yy, xx] = sources
    observed = source_view >= 0
    unseen_policy = "nearest color within observed charts; unseen charts keep a flat prior"
    if fusion == "robust" and observed.any():
        # Calibrate only the explicitly UNSEEN prior to this part's observed color.
        # Do not paint over any measured grain, logo or other known detail.
        prior = np.rint(np.median(texture[observed], axis=0)).astype(np.uint8)
        texture[surface & ~observed] = prior
        unseen_policy = "nearest color within observed charts; unseen charts use the observed part median"
    # Continue nearby observed colors within each UV chart to avoid hard seams.
    # This is explicit nearest-color inference, not newly observed texture.
    inferred = np.zeros_like(observed)
    components, _ = label(surface)
    for component_id, bounds in enumerate(find_objects(components), 1):
        if bounds is None:
            continue
        chart = components[bounds] == component_id
        known = observed[bounds] & chart
        if not known.any():
            continue
        _, nearest = distance_transform_edt(~known, return_indices=True)
        fill = chart & ~known
        region = texture[bounds]
        region[fill] = region[nearest[0][fill], nearest[1][fill]]
        inferred[bounds][fill] = True
    # Pad atlas gutters without changing the measured-coverage classification.
    if observed.any():
        distance, nearest = distance_transform_edt(~surface, return_indices=True)
        padding = (~surface) & (distance <= 6)
        texture[padding] = texture[nearest[0][padding], nearest[1][padding]]
    baked.visual = trimesh.visual.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(texture),
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            roughnessFactor=mesh.metadata["roughness"],
            metallicFactor=mesh.metadata["metallic"],
            alphaMode="OPAQUE",
        ),
    )
    baked.metadata = mesh.metadata.copy()
    coverage = np.zeros((height, width, 3), np.uint8)
    coverage[surface] = [225, 89, 56]
    coverage[observed] = [52, 187, 117]
    return (
        baked,
        Image.fromarray(coverage),
        {
            "surface_texels": int(surface.sum()),
            "observed_texels": int(observed.sum()),
            "observed_uv_fraction": float(observed.sum() / max(1, surface.sum())),
            "inferred_continuation_texels": int(inferred.sum()),
            "texture_dimensions": [width, height],
            "source_view_texels": [int(np.sum(source_view == i)) for i in range(len(views))],
            "unseen_policy": unseen_policy,
            "fusion": fusion,
            "minimum_observations": 2 if fusion == "robust" else 1,
        },
    )


def bake_video_textures(
    workspace: Path,
    recipe_path: Path,
    out: Path,
    resolution=768,
    *,
    delighting_manifest: Path | None = None,
    fusion=DEFAULT_FUSION,
):
    if not 128 <= resolution <= 2048:
        raise ValueError("Texture resolution must be between 128 and 2048")
    workspace = workspace.resolve()
    recipe = json.loads(recipe_path.read_text())
    if recipe.get("version") != 1 or not 1 <= len(recipe.get("views", [])) <= 12:
        raise ValueError("Photo recipe needs version 1 and 1..12 registered views")
    if len({entry["frame"] for entry in recipe["views"]}) != len(recipe["views"]):
        raise ValueError("Registered views must use distinct prepared frames")
    _validate_fusion(fusion, len(recipe["views"]))
    scene_path = local_file(workspace, recipe["scene"])
    spec = SceneSpec.model_validate_json(scene_path.read_text())
    asset = build_asset(spec)
    mesh = asset.mesh()
    prepared = PreparedInput.model_validate_json((workspace / "input.json").read_text())
    source_frames = {frame.path: frame for frame in prepared.frames}
    corrected_images, delighting = {}, None
    if delighting_manifest is not None:
        corrected_images, delighting = load_delighting(workspace, recipe_path, delighting_manifest)
    views = []
    reports = []
    for entry in recipe["views"]:
        frame = local_file(workspace, entry["frame"])
        if entry["frame"] not in source_frames or digest(frame) != source_frames[entry["frame"]].sha256:
            raise ValueError("Frame is absent from the prepared video manifest or its pixels have changed")
        image = np.array(Image.open(frame).convert("RGB"))
        mask_file = local_file(workspace, entry["mask"])
        mask = read_mask(workspace, entry["mask"], (image.shape[1], image.shape[0]))
        points = np.asarray([p["xyz"] for p in entry["landmarks"]], float)
        points = points * asset.metadata["scale_factor"] + np.asarray(asset.metadata["translation_m"])
        pixels = np.asarray([p["uv"] for p in entry["landmarks"]], float)
        camera = fit_camera(points, pixels, (image.shape[1], image.shape[0]))
        if camera["reprojection_rms_px"] > recipe.get("max_reprojection_rms_px", 12):
            raise ValueError(
                f"Camera fit too poor for {frame.name}: {camera['reprojection_rms_px']:.1f} px RMS"
            )
        depth = depth_buffer(mesh, camera, (image.shape[1], image.shape[0]))
        # Registration uncertainty is largest at the silhouette. Reject a band
        # around it instead of baking desk pixels into the object's edge charts.
        erosion = 2 * int(np.ceil(camera["reprojection_rms_px"] + 1)) + 1
        safe_mask = cv2.erode(mask, np.ones((erosion, erosion), np.uint8)) > 0
        size = (image.shape[1], image.shape[0])
        regions = {
            name: read_mask(workspace, annotation, size) > 0
            for name, annotation in entry.get("regions", {}).items()
        }
        occluders = [
            {
                "mask": cv2.dilate(
                    read_mask(
                        workspace, item["mask"] if "mask" in item else {"include": [item["polygon"]]}, size
                    ),
                    np.ones((5, 5), np.uint8),
                )
                > 0,
                "parts": item["parts"],
            }
            for item in entry.get("occluders", [])
        ]
        views.append(
            {
                "camera": camera,
                "image": image,
                "mask": safe_mask,
                "depth": depth,
                "regions": regions,
                "occluders": occluders,
                "color_guards": recipe.get("color_guards", {}),
            }
        )
        if delighting is not None:
            views[-1]["texture_image"] = corrected_images[entry["frame"]]
        reports.append(
            {
                "frame": entry["frame"],
                "frame_sha256": digest(frame),
                "mask_sha256": digest(mask_file),
                "registration_source": entry.get("registration_source", "reviewed_landmarks"),
                **camera,
            }
        )
    out = fresh_directory(out)
    for i, (view, entry) in enumerate(zip(views, recipe["views"], strict=True)):
        overlay = view["image"].copy()
        # Show silhouette edges, not the dense triangulation, plus fitted anchors.
        contours, _ = cv2.findContours(
            np.isfinite(view["depth"]).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (20, 240, 230), 2)
        for anchor, predicted in zip(entry["landmarks"], reports[i]["projected_landmarks"], strict=True):
            cv2.circle(overlay, tuple(np.round(anchor["uv"]).astype(int)), 4, (255, 55, 45), -1)
            cv2.circle(overlay, tuple(np.round(predicted).astype(int)), 4, (40, 240, 100), 1)
        Image.fromarray(overlay).save(out / f"registration_{i:02d}.jpg", quality=94)
    write_json(out / "cameras.json", reports)
    write_json(out / "recipe.json", recipe)
    results = []
    glb = trimesh.Scene()
    tolerance = float(np.max(mesh.extents)) * 0.006
    for i, part in enumerate(asset.visuals):
        print(f"Baking {part.metadata['name']} ...", flush=True)
        textured, coverage, report = bake_mesh(
            part, views, resolution, tolerance, resolution / float(mesh.extents.max()), fusion=fusion
        )
        textured.visual.material.baseColorTexture.save(out / f"texture_{i:03d}.png")
        coverage.save(out / f"coverage_{i:03d}.png")
        textured.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        glb.add_geometry(textured, node_name=f"Part_{i:03d}", geom_name=f"Part_{i:03d}")
        results.append({"part": part.metadata["name"], **report})
    glb.export(out / "visual.glb")
    provenance = {
        "backend": "video_pixel_projection",
        "model_calls": 0,
        "texture_source": "actual video pixels projected through landmark-fitted cameras",
        "scene_sha256": digest(scene_path),
        "recipe_sha256": digest(recipe_path),
        "source_video_sha256": prepared.source_sha256,
        "source_views": reports,
        "parts": results,
        "visual_sha256": digest(out / "visual.glb"),
        "geometry_changed": False,
        "view_fusion": fusion,
        "photometric_calibration": False,
        "metric_calibration": False,
        "unseen_policy": "nearest color within observed charts; unseen charts keep a flat prior",
        "warnings": [
            "Input lighting/shadows remain in sampled colors; this is not calibrated albedo.",
            "View registration depends on approximate geometry and reviewed or tracked landmarks; "
            "inspect overlays.",
            "Coverage maps: green observed, orange unseen prior, black atlas gutter.",
        ],
    }
    if fusion == "robust":
        provenance["unseen_policy"] = (
            "nearest color within observed charts; unseen charts use observed part median, "
            "or original flat prior if no supported samples exist"
        )
        provenance["warnings"].extend(
            [
                "Robust fusion requires two agreeing views; adjacent video observations are correlated.",
                "Broad color is blended in sRGB; fine texture comes from one highest-scoring inlier view.",
                "Entirely unseen charts remain estimated flat colors, not recovered texture.",
            ]
        )
    if delighting is not None:
        write_json(out / "delighting.json", delighting)
        provenance.update(
            model_calls=delighting["model_calls"],
            bake_model_calls=0,
            texture_source="original video pixels with intrinsic-guided lighting correction, "
            "projected through fitted cameras",
            delighting={
                # Hash the saved, canonical copy of the data actually validated
                # and used, even if the input JSON was formatted differently.
                "manifest_sha256": digest(out / "delighting.json"),
                **{key: delighting[key] for key in ("algorithm", "model", "config", "views", "color_spaces")},
            },
        )
        provenance["warnings"][0] = (
            "Lighting removal is an estimated albedo correction, not photometric calibration; "
            "colors, fine detail and residual shading need review."
        )
    write_json(out / "provenance.json", provenance)
    physics = PhysicsConfig.model_validate(recipe.get("physics", {}))
    metadata = build_textured_asset(
        out / "visual.glb",
        scene_path,
        out / "asset",
        physics,
        visual_links={f"Part_{i:03d}": mesh.metadata["link"] for i, mesh in enumerate(asset.visuals)}
        if asset.metadata.get("articulation")
        else None,
    )
    metadata.update(
        alignment_method="original mesh geometry unchanged; UV vertex splitting only",
        status="needs_texture_review_and_physics_validation",
        texture_coverage=results,
    )
    write_json(out / "asset/asset.json", metadata)
    return out / "asset/asset_dynamic.usda"
