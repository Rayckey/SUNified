"""Compare selected video crops and approximate render views; no pose registration.

Usage: python scripts/assets/compare_views.py --workspace /external/0029 --scene scene_v1.json
       --views comparison_views.json --out comparison_v1.jpg
All arguments except workspace are relative to that workspace. A view config is a list
of {frame, crop: [left,top,right,bottom], azimuth, elevation, label} records in JPEG pixels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from kuafu.asset_generation.geometry import build_asset
from kuafu.asset_generation.io import local_file, resolve_artifact_path
from kuafu.asset_generation.render import render_mesh
from kuafu.asset_generation.schema import SceneSpec


def compare(workspace: Path, scene: Path, views: Path, out: Path):
    out = resolve_artifact_path(out, "Asset comparison image")
    mesh = build_asset(SceneSpec.model_validate_json(scene.read_text())).mesh()
    config = json.loads(views.read_text())
    if not 1 <= len(config) <= 6:
        raise ValueError("Use one to six reference views")
    width = len(config) * 480
    sheet = Image.new("RGB", (width, 760), "#eff2f7")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        title = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = title = ImageFont.load_default()
    draw.text((24, 16), f"{workspace.name} | video-only reconstruction", font=title, fill="#172033")
    draw.text(
        (24, 51),
        "Top: video crops. Bottom: approximate render views; cameras and lighting are uncalibrated.",
        font=font,
        fill="#48566b",
    )
    for i, view in enumerate(config):
        x = i * 480
        photo = ImageOps.contain(
            Image.open(local_file(workspace, view["frame"])).crop(view["crop"]), (420, 285)
        )
        sheet.paste(photo, (x + (480 - photo.width) // 2, 92 + (285 - photo.height) // 2))
        rendered = render_mesh(mesh, view["azimuth"], view["elevation"], 640)
        mask = np.any(np.array(rendered) != [239, 242, 247], axis=2)
        yy, xx = np.nonzero(mask)
        cropped = rendered.crop((int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1))
        fitted = ImageOps.contain(cropped, (420, 285))
        sheet.paste(fitted, (x + (480 - fitted.width) // 2, 407 + (285 - fitted.height) // 2))
        draw.text((x + 24, 710), view["label"], font=font, fill="#172033")
    sheet.save(out, quality=94)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--views", default="comparison_views.json")
    parser.add_argument("--out", default="comparison.jpg")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = (workspace / args.out).resolve()
    if not output.is_relative_to(workspace):
        parser.error("Output must remain inside the workspace")
    compare(workspace, local_file(workspace, args.scene), local_file(workspace, args.views), output)
