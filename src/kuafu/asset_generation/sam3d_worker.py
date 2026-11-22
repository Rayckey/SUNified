"""Standalone worker: run with the separate SAM 3D environment, not Isaac's Python.

The official model/configs execute trusted Python through Hydra and torch.load.
Only use approved checkpoints from Meta or a trusted local copy. This is not a
sandbox for arbitrary model repositories. No tokens are logged or saved here.
"""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_targets(config):
    """Reject non-model Hydra call targets before instantiating trusted configs."""
    if isinstance(config, dict):
        target = config.get("_target_")
        if target is not None and (
            not isinstance(target, str)
            or "__" in target
            or not target.startswith(("sam3d_objects.", "torch.nn.", "torchvision.", "moge."))
        ):
            raise ValueError(f"Unapproved Hydra target: {target}")
        for value in config.values():
            check_targets(value)
    elif isinstance(config, list):
        for value in config:
            check_targets(value)


def fetch(root):
    from huggingface_hub import HfApi, get_token, snapshot_download

    if not get_token():
        raise RuntimeError(
            "No local Hugging Face authentication. Request access at "
            "https://huggingface.co/facebook/sam-3d-objects and run `hf auth login` locally. "
            "Do not paste your token into chat."
        )
    repo = "facebook/sam-3d-objects"
    info = HfApi().model_info(repo)
    destination = root / "models/sam3d"
    snapshot_download(
        repo_id=repo,
        revision=info.sha,
        allow_patterns=["checkpoints/*"],
        local_dir=destination,
        max_workers=1,
    )
    manifest = {
        "repository": repo,
        "revision": info.sha,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            str(p.relative_to(destination)): sha256(p)
            for p in sorted((destination / "checkpoints").rglob("*"))
            if p.is_file()
        },
    }
    (destination / "download.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Approved checkpoint snapshot downloaded to {destination / 'checkpoints'}")


def gpu_kernels():
    """Small real GPU operations; this tests compiled libraries, NOT model inference."""
    import nvdiffrast.torch as dr
    import torch
    import xformers.ops
    from gsplat import rasterization
    from pytorch3d.renderer.mesh import rasterize_meshes
    from pytorch3d.structures import Meshes
    from spconv.pytorch import SparseConvTensor, SubMConv3d

    q = torch.randn(1, 16, 2, 32, device="cuda", dtype=torch.float16)
    attention = xformers.ops.memory_efficient_attention(q, q, q)
    assert torch.isfinite(attention).all()
    vertices = torch.tensor([[[-0.5, -0.5, 1], [0.5, -0.5, 1], [0, 0.5, 1]]], device="cuda")
    faces = torch.tensor([[[0, 1, 2]]], device="cuda")
    pix, *_ = rasterize_meshes(Meshes(vertices, faces), image_size=16)
    assert (pix >= 0).any()
    features = torch.ones(4, 8, device="cuda")
    indices = torch.tensor(
        [[0, 1, 1, 1], [0, 1, 1, 2], [0, 1, 2, 1], [0, 2, 1, 1]], dtype=torch.int32, device="cuda"
    )
    sparse = SparseConvTensor(features, indices, [4, 4, 4], 1)
    conv = SubMConv3d(8, 8, 3, padding=1).cuda()
    assert torch.isfinite(conv(sparse).features).all()
    colors, alpha, _ = rasterization(
        means=torch.tensor([[0.0, 0.0, 2.0]], device="cuda"),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
        scales=torch.full((1, 3), 0.1, device="cuda"),
        opacities=torch.full((1,), 0.8, device="cuda"),
        colors=torch.tensor([[1.0, 0.5, 0.2]], device="cuda"),
        viewmats=torch.eye(4, device="cuda")[None],
        Ks=torch.tensor([[[20.0, 0.0, 8.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]]], device="cuda"),
        width=16,
        height=16,
    )
    assert torch.isfinite(colors).all() and alpha.max() > 0
    clip_vertices = torch.tensor([[[-0.5, -0.5, 0, 1], [0.5, -0.5, 0, 1], [0, 0.5, 0, 1]]], device="cuda")
    raster, _ = dr.rasterize(dr.RasterizeCudaContext(), clip_vertices, faces[0].int(), resolution=[16, 16])
    assert (raster[..., 3] > 0).any()
    torch.cuda.synchronize()
    return [
        "float16_xformers_attention",
        "pytorch3d_cuda_rasterization",
        "spconv",
        "gsplat",
        "nvdiffrast_cuda",
    ]


def doctor(root, kernel_test=False):
    report = {"runtime": str(root), "dependencies": {}, "model_inference_tested": False}
    for name in (
        "torch",
        "torchvision",
        "xformers.ops",
        "pytorch3d._C",
        "spconv.pytorch",
        "kaolin",
        "nvdiffrast.torch",
        "gsplat",
        "utils3d",
        "moge",
        "hydra",
        "sam3d_objects.pipeline.inference_pipeline_pointmap",
    ):
        try:
            module = importlib.import_module(name)
            report["dependencies"][name] = {"ok": True, "version": getattr(module, "__version__", None)}
        except Exception as error:
            report["dependencies"][name] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    try:
        import torch
        from huggingface_hub import get_token

        report["hf_authenticated"] = bool(get_token())
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            capability = torch.cuda.get_device_capability()
            report["gpu"] = {
                "name": torch.cuda.get_device_name(),
                "capability": capability,
                "free_gib": round(free / 2**30, 2),
                "total_gib": round(total / 2**30, 2),
                "selected_physical_gpu": os.getenv("CUDA_VISIBLE_DEVICES"),
                "dtype": "float16" if capability[0] < 8 else "bfloat16",
            }
            if kernel_test:
                report["tested_kernels"] = gpu_kernels()
    except Exception as error:
        report["gpu_error"] = f"{type(error).__name__}: {error}"
    report["checkpoint_config_exists"] = (root / "models/sam3d/checkpoints/pipeline.yaml").is_file()
    report["runtime_imports_ok"] = all(value["ok"] for value in report["dependencies"].values())
    report["ready_for_inference_attempt"] = bool(
        report["runtime_imports_ok"]
        and report.get("cuda_available")
        and report["checkpoint_config_exists"]
        and not report.get("gpu_error")
    )
    print(json.dumps(report, indent=2))
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs/doctor.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["ready_for_inference_attempt"] else 2


def infer(args):
    import numpy as np
    import torch
    import trimesh
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from PIL import Image

    job_path = args.job.resolve()
    job = json.loads(job_path.read_text())
    files = {}
    for field in ("image", "mask"):
        path = (job_path.parent / job[field]).resolve()
        if not path.is_relative_to(job_path.parent) or sha256(path) != job[f"{field}_sha256"]:
            raise ValueError(f"Missing, unsafe or changed {field}")
        files[field] = path
    image = np.asarray(Image.open(files["image"]).convert("RGB"))
    mask = np.asarray(Image.open(files["mask"]))
    if mask.shape != image.shape[:2] or not set(np.unique(mask)).issubset({0, 255}):
        raise ValueError("Mask/image dimensions or binary encoding invalid")
    if np.count_nonzero(mask) < 64 or np.all(mask):
        raise ValueError("Mask must contain both object and background")
    rgba = np.concatenate([image, mask[..., None]], axis=-1)
    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3D requires an available CUDA GPU")
    free, _ = torch.cuda.mem_get_info()
    if free < 32 * 2**30:
        raise RuntimeError("Less than 32 GiB GPU memory is free; choose an idle GPU and retry")
    capability = torch.cuda.get_device_capability()
    dtype = "float16" if capability[0] < 8 else "bfloat16"
    # Official code automatically forces flash_attn on certain GPUs. This runtime
    # deliberately uses the installed xformers implementation on every architecture.
    from sam3d_objects.model.backbone.tdfy_dit.modules import attention, sparse
    from sam3d_objects.pipeline import inference_pipeline  # noqa: F401

    attention.set_backend("xformers")
    sparse.set_attn("xformers")
    config = OmegaConf.load(args.config)
    config.workspace_dir = str(args.config.parent)
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.dtype = dtype
    config.shape_model_dtype = dtype
    check_targets(OmegaConf.to_container(config, resolve=True))
    used_files = {"pipeline.yaml": sha256(args.config)}
    for key, value in config.items():
        if key.endswith(("_config_path", "_ckpt_path")) and value is not None:
            path = Path(value)
            if not path.is_absolute():
                path = args.config.parent / path
            if not path.is_file() or path.stat().st_size < 16:
                raise ValueError(f"Missing or incomplete model file: {path}")
            used_files[key] = sha256(path)
            if key.endswith("_config_path"):
                check_targets(OmegaConf.to_container(OmegaConf.load(path), resolve=True))
    pipeline = instantiate(config)
    # No fill-holes postprocess: it can destroy real holder/tray cavities.
    # UV unwrapping + texture baking still run (the latter needs nvdiffrast).
    output = pipeline.run(
        rgba,
        None,
        args.seed,
        stage1_only=False,
        with_mesh_postprocess=False,
        with_texture_baking=True,
        with_layout_postprocess=False,
        use_vertex_color=False,
    )
    mesh = output.get("glb")
    geometries = list(mesh.geometry.values()) if isinstance(mesh, trimesh.Scene) else [mesh]
    for geometry in geometries:
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
            raise RuntimeError("SAM 3D did not return a triangle mesh")
        visual = geometry.visual
        if (
            visual.kind != "texture"
            or visual.uv is None
            or getattr(visual.material, "baseColorTexture", None) is None
        ):
            raise RuntimeError("SAM 3D returned no UV texture; refusing to label vertex colors textured")
        if not np.isfinite(geometry.vertices).all() or not np.isfinite(visual.uv).all():
            raise RuntimeError("SAM 3D produced non-finite geometry/UVs")
    mesh.export(args.out / "visual.glb")
    revision = subprocess.check_output(
        ["git", "-C", str(args.root / "repo"), "rev-parse", "HEAD"], text=True
    ).strip()
    provenance = {
        "backend": "facebook/sam-3d-objects",
        "repo_revision": revision,
        "job_sha256": sha256(job_path),
        "model_files_sha256": used_files,
        "seed": args.seed,
        "dtype": dtype,
        "attention_backend": "xformers",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "with_texture_baking": True,
        "with_mesh_postprocess": False,
        "use_vertex_color": False,
        "with_layout_postprocess": False,
        "input_views": 1,
        "glb_up_axis": "Y",
        "scale_basis": "uncalibrated",
        "texture_source": "SAM 3D generated appearance baked to UV atlas, not direct video projection",
        "visual_sha256": sha256(args.out / "visual.glb"),
        "visual_triangles": sum(len(g.faces) for g in geometries),
        "status": "needs_visual_and_alignment_review",
        "isaac_physics_tested": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Textured candidate: {args.out / 'visual.glb'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--doctor", action="store_true")
    action.add_argument("--fetch", action="store_true")
    action.add_argument("--job", type=Path)
    parser.add_argument("--kernel-test", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.doctor:
        return doctor(args.root, args.kernel_test)
    if args.fetch:
        fetch(args.root)
    else:
        if args.out is None or args.config is None:
            parser.error("inference requires --out and --config")
        infer(args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"SAM 3D error: {type(error).__name__}: {error}", file=sys.stderr)
        raise
