import argparse
import json
import logging
import os
import sys
from pathlib import Path

from kuafu.paths import resolve_artifact_path

from . import __version__
from .delighting import DelightConfig, delight_frames
from .demo import run_demo
from .export import export_asset
from .geometry import build_asset
from .io import fresh_directory
from .multiview import TrackConfig, expand_photo_views
from .phototexture import DEFAULT_FUSION, bake_video_textures
from .pipeline import reconstruct
from .reviewed import build_reviewed, package_reviewed
from .sam3d import DEFAULT_ROOT, prepare_sam3d, run_sam3d, sam3d_maintenance
from .schema import PhysicsConfig, RunConfig, Scale, SceneSpec
from .textured import build_textured_asset
from .video import prepare


def _scale_args(parser):
    parser.add_argument("--size-m", type=float, help="Known overall bounding dimension, in meters")
    parser.add_argument("--size-axis", choices=["x", "y", "z", "longest"], default="z")


def _physics_args(parser):
    parser.add_argument(
        "--mass-kg", type=float, help="Total object/assembly mass (default: provisional 0.2 kg)"
    )
    parser.add_argument(
        "--estimated-mass", action="store_true", help="Mark --mass-kg as an assumption, not a measurement"
    )
    parser.add_argument("--friction", type=float, default=0.5)
    parser.add_argument("--restitution", type=float, default=0.0)


def _agent_args(parser):
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-6-astra"))
    parser.add_argument("--iterations", type=int, default=3, help="At most two model requests per iteration")
    parser.add_argument("--target-score", type=float, default=0.85)
    parser.add_argument("--max-output-tokens", type=int, default=16000)
    parser.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high", "xhigh", "max"], default="high"
    )
    parser.add_argument("--timeout-s", type=float, default=300)
    _physics_args(parser)


def _input_args(parser):
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--reference-video",
        type=Path,
        action="append",
        default=[],
        help="Additional clip of this assembly or a removed part (up to three)",
    )
    parser.add_argument(
        "--motion",
        choices=["auto", "rigid", "articulated"],
        default="auto",
        help="Infer motion by default; articulated rejects a rigid-only candidate",
    )
    description = parser.add_mutually_exclusive_group(required=True)
    description.add_argument("--description")
    description.add_argument("--description-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=12, help="Total sampled frames across all clips")
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--crop", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"))
    _scale_args(parser)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="kuafu assets", description="Video + language to Kuafu simulation assets")
    root.add_argument("--version", action="version", version=__version__)
    sub = root.add_subparsers(dest="command", required=True)
    _input_args(sub.add_parser("prepare", help="Extract views and a manifest; no API key required"))
    run = sub.add_parser("run", help="Prepare a video and run the vision reconstruction agent")
    _input_args(run)
    _agent_args(run)
    resume = sub.add_parser("reconstruct", help="Run/retry an existing workspace; reuse cached responses")
    resume.add_argument("workspace", type=Path)
    _agent_args(resume)
    build = sub.add_parser("build-scene", help="Rebuild an editable scene JSON offline")
    build.add_argument("scene", type=Path)
    build.add_argument("--out", type=Path, required=True)
    _scale_args(build)
    _physics_args(build)
    for name, help_text in (
        ("build-reviewed", "Replay a saved batch of reviewed scenes without new model inference"),
        ("package-reviewed", "Package reviewed assets after hash-verified physics tests"),
    ):
        reviewed = sub.add_parser(name, help=help_text)
        reviewed.add_argument("manifest", type=Path)
        reviewed.add_argument(
            "--output-name", default="final", help="New child output directory per workspace"
        )
    demo = sub.add_parser("demo", help="Run a synthetic, scripted pipeline demo without API calls")
    demo.add_argument("--out", type=Path, required=True)
    sub.add_parser("turntable", help="Render a spinning preview; see 'turntable --help'")
    validate = sub.add_parser("validate", help="Check an exported asset in Isaac Sim and record USD hashes")
    validate.add_argument("--asset", type=Path, required=True)
    validate.add_argument("--report", type=Path, help="Defaults to physics_smoke.json beside the asset")
    validate.add_argument("--python", help="Isaac Python executable; defaults to KUAFU_ISAAC_PYTHON or this Python")
    validate.add_argument("--steps", type=int, default=240)
    validate.add_argument("--probe-radius", type=float)
    validate.add_argument("--cavity-floor-z", type=float)
    sub.add_parser("schema", help="Print the editable scene's JSON Schema")
    masked = sub.add_parser("sam3d-prepare", help="Prepare one selected video frame and an object-only mask")
    masked.add_argument("workspace", type=Path)
    masked.add_argument("--frame", type=int, required=True, help="Zero-based ordinal in input.json frames")
    masked.add_argument("--mask", type=Path, required=True, help="Binary PNG or include/exclude polygon JSON")
    masked.add_argument("--out", type=Path, required=True)
    for action in ("doctor", "fetch"):
        maintenance = sub.add_parser(f"sam3d-{action}", help=f"SAM 3D runtime {action}; no OpenAI calls")
        maintenance.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        maintenance.add_argument("--gpu", type=int, default=0)
        if action == "doctor":
            maintenance.add_argument("--kernel-test", action="store_true")
    sam = sub.add_parser("sam3d-run", help="Generate a UV-textured candidate with official SAM 3D weights")
    sam.add_argument("job", type=Path)
    sam.add_argument("--out", type=Path, required=True)
    sam.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sam.add_argument("--gpu", type=int, default=0, help="Physical GPU index; requires 32 GiB free")
    sam.add_argument(
        "--config", type=Path, help="Trusted official local pipeline.yaml, if already downloaded"
    )
    sam.add_argument("--seed", type=int, default=42)
    sam.add_argument("--timeout-s", type=int, default=3600)
    sam.add_argument("--mask-reviewed", action="store_true")
    textured = sub.add_parser(
        "build-textured", help="Pair a textured GLB visual with editable collision geometry"
    )
    textured.add_argument("visual", type=Path)
    textured.add_argument("--collision-scene", type=Path, required=True)
    textured.add_argument(
        "--visual-links",
        type=Path,
        help="JSON map from GLB mesh node to articulated link; required for articulated imports",
    )
    textured.add_argument("--out", type=Path, required=True)
    textured.add_argument("--visual-rotation-deg", type=float, nargs=3, default=[0, 0, 0])
    textured.add_argument("--visual-scale-axis", choices=["x", "y", "z", "longest"], default="longest")
    _scale_args(textured)
    _physics_args(textured)
    photo = sub.add_parser(
        "bake-video-textures", help="Bake actual video pixels onto an existing mesh; no model account"
    )
    photo.add_argument("workspace", type=Path)
    photo.add_argument("--recipe", type=Path, required=True, help="Reviewed frame masks and 2D/3D landmarks")
    photo.add_argument("--out", type=Path, required=True)
    photo.add_argument("--resolution", type=int, default=768)
    photo.add_argument(
        "--fusion",
        choices=["best", "robust"],
        default=DEFAULT_FUSION,
        help="robust (default): 3+ registered views, two-view support and detail-preserving color fusion; "
        "best: explicit legacy single-view selection",
    )
    photo.add_argument(
        "--delighting", type=Path, help="Verified delight-frames manifest; default uses raw RGB"
    )
    track = sub.add_parser(
        "track-photo-views", help="Expand reviewed anchors using quality-gated dense video tracking"
    )
    track.add_argument("workspace", type=Path)
    track.add_argument("--recipe", type=Path, required=True)
    track.add_argument("--extra-anchors", type=Path, help="Additional reviewed version-1 seed views")
    track.add_argument("--video", type=Path, help="Hash-identical source video if its original path moved")
    track.add_argument("--out", type=Path, required=True, help="New directory inside the workspace")
    track.add_argument("--max-views", type=int, default=12)
    delight = sub.add_parser(
        "delight-frames",
        help="Estimate/remove baked lighting while retaining source texture detail; local model",
    )
    delight.add_argument("workspace", type=Path)
    delight.add_argument("--recipe", type=Path, required=True)
    delight.add_argument("--out", type=Path, required=True)
    delight.add_argument("--device", default="auto", help="auto, cpu or cuda:0 (etc.)")
    delight.add_argument("--cache-dir", type=Path, help="Public model download cache, about 2.6 GB")
    delight.add_argument(
        "--local-files-only", action="store_true", help="Use cached weights without network access"
    )
    for name in ("steps", "ensemble_size", "processing_resolution", "seed"):
        delight.add_argument(
            f"--{name.replace('_', '-')}", type=int, default=DelightConfig.model_fields[name].default
        )
    for name in ("strength", "max_gain", "smooth_fraction", "crop_margin"):
        delight.add_argument(
            f"--{name.replace('_', '-')}", type=float, default=DelightConfig.model_fields[name].default
        )
    return root


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "turntable":
        from .turntable import main as turntable_main

        try:
            return turntable_main(argv[1:])
        except (ValueError, RuntimeError, OSError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        for field in ("out", "workspace", "job", "root", "cache_dir"):
            if getattr(args, field, None) is not None:
                setattr(args, field, resolve_artifact_path(getattr(args, field), f"Asset {field}"))
        if args.command == "validate":
            from .validation import validate_asset

            report = validate_asset(
                args.asset, report=args.report, python=args.python, steps=args.steps,
                probe_radius=args.probe_radius, cavity_floor_z=args.cavity_floor_z,
            )
            print(f"Physics passed: {report}")
            return 0
        if args.command == "track-photo-views":
            print(
                expand_photo_views(
                    args.workspace,
                    args.recipe,
                    args.out,
                    TrackConfig(max_views=args.max_views),
                    extra_anchors=args.extra_anchors,
                    video=args.video,
                )
            )
            print("Inspect every registration and mask before de-lighting or baking the expanded recipe.")
            return 0
        if args.command == "delight-frames":
            config = DelightConfig(**{key: getattr(args, key) for key in DelightConfig.model_fields})
            print(
                delight_frames(
                    args.workspace,
                    args.recipe,
                    args.out,
                    config,
                    device=args.device,
                    cache_dir=args.cache_dir,
                    local_files_only=args.local_files_only,
                )
            )
            print("Review comparison images, then pass manifest.json to bake-video-textures --delighting.")
            return 0
        if args.command == "bake-video-textures":
            print(
                bake_video_textures(
                    args.workspace,
                    args.recipe,
                    args.out,
                    args.resolution,
                    delighting_manifest=args.delighting,
                    fusion=args.fusion,
                )
            )
            print("Video textures baked. Inspect registration/coverage maps and validate the new USD.")
            return 0
        if args.command == "sam3d-prepare":
            print(prepare_sam3d(args.workspace, args.frame, args.mask, args.out))
            print("Inspect mask_review.jpg and cutout.png before inference. No model has run yet.")
            return 0
        if args.command in ("sam3d-doctor", "sam3d-fetch"):
            return sam3d_maintenance(
                args.root, args.command.removeprefix("sam3d-"), args.gpu, getattr(args, "kernel_test", False)
            )
        if args.command == "sam3d-run":
            print(
                run_sam3d(
                    args.job,
                    args.out,
                    args.root,
                    args.gpu,
                    args.mask_reviewed,
                    args.seed,
                    args.config,
                    args.timeout_s,
                )
            )
            print("Candidate needs visual and alignment review; not a validated simulation asset yet.")
            return 0
        if args.command == "schema":
            print(json.dumps(SceneSpec.model_json_schema(), indent=2))
            return 0
        if args.command == "demo":
            state = run_demo(args.out)
            print(f"Offline demo: {state['workspace']}/final/asset_dynamic.usda")
            print("Scripted responses exercised the pipeline; no AI inference was performed.")
            return 0
        if args.command in ("build-reviewed", "package-reviewed"):
            operation = build_reviewed if args.command == "build-reviewed" else package_reviewed
            for path in operation(args.manifest, args.output_name):
                print(path)
            return 0
        scale = (
            Scale(size_m=args.size_m, axis=args.size_axis)
            if getattr(args, "size_m", None) is not None
            else None
        )
        if args.command in ("prepare", "run"):
            description = (
                args.description if args.description is not None else args.description_file.read_text()
            )
            prepare(
                args.video,
                description,
                args.out,
                args.frames,
                args.max_side,
                args.crop,
                scale,
                args.reference_video,
                args.motion,
            )
            if args.command == "prepare":
                print(f"Prepared: {args.out.resolve()}/input.json")
                return 0
            args.workspace = args.out
        physics = PhysicsConfig(
            mass_kg=args.mass_kg,
            mass_is_estimate=args.estimated_mass,
            friction=args.friction,
            restitution=args.restitution,
        )
        if args.command == "build-textured":
            build_textured_asset(
                args.visual,
                args.collision_scene,
                args.out,
                physics,
                args.visual_rotation_deg,
                args.visual_scale_axis,
                scale,
                visual_links=json.loads(args.visual_links.read_text()) if args.visual_links else None,
            )
            print(f"Textured candidate: {args.out.resolve()}/asset_dynamic.usda")
            print("Review visual/collider alignment and rerun physics validation before use.")
            return 0
        if args.command == "build-scene":
            spec = SceneSpec.model_validate_json(args.scene.read_text())
            asset = build_asset(spec, scale)
            out = fresh_directory(args.out)
            export_asset(asset, spec, out, physics, scale)
            print(f"Built: {out}/asset_dynamic.usda")
            return 0
        config = RunConfig(**{key: getattr(args, key) for key in RunConfig.model_fields})
        state = reconstruct(args.workspace, config, physics)
        print(f"{state['status']}: {args.workspace.resolve()}/final/asset_dynamic.usda")
        print(f"Preview: {args.workspace.resolve()}/final/viewer.html")
        if state["status"] == "needs_review":
            print("A usable candidate was exported, but it did not pass the agent's acceptance gate.")
            return 2
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        # Provider/network errors still leave run.json and successful cache entries available.
        print(f"error ({type(error).__name__}): {error}", file=sys.stderr)
        return 1
