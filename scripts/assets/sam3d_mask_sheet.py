"""Render staged mask cutouts for review; does not call any reconstruction model."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from kuafu.asset_generation.io import local_file, resolve_artifact_path
from kuafu.asset_generation.sam3d import validate_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = resolve_artifact_path(args.out, "SAM 3D mask sheet")
    manifest = json.loads(args.manifest.read_text())
    sheet = Image.new("RGB", (400 * len(manifest["jobs"]), 500), "#e9edf2")
    draw = ImageDraw.Draw(sheet)
    for index, entry in enumerate(manifest["jobs"]):
        job_path = local_file(args.manifest.parent, entry["job"])
        validate_job(job_path)
        with Image.open(job_path.parent / "cutout.png") as image:
            cutout = image.crop(image.getbbox())
            cutout.thumbnail((370, 400))
            sheet.paste(
                cutout, (index * 400 + (400 - cutout.width) // 2, 50 + (400 - cutout.height) // 2), cutout
            )
        draw.text((index * 400 + 20, 20), f"{entry['id']} | masked video frame", fill="#15232c")
        draw.text((index * 400 + 20, 467), "Input only - no 3D inference", fill="#15232c")
    if args.out.exists():
        raise ValueError("Choose a new mask sheet path; refusing to overwrite")
    sheet.save(args.out)


if __name__ == "__main__":
    main()
