"""Known-answer demo: exercises the pipeline without claiming AI inference."""

from pathlib import Path

import cv2
import numpy as np

from .geometry import build_asset
from .io import fresh_directory, write_json
from .pipeline import reconstruct
from .render import render_mesh
from .schema import PhysicsConfig, Review, RunConfig, Scale, SceneSpec
from .video import prepare


def demo_scene() -> SceneSpec:
    material = {"color": {"r": 0.12, "g": 0.43, "b": 0.72}, "roughness": 0.3, "metallic": 0.0}
    pose = {"position": {"x": 0, "y": 0, "z": 0}, "rotation_deg": {"x": 0, "y": 0, "z": 0}}
    return SceneSpec.model_validate(
        {
            "name": "Demo mug",
            "summary": "Blue hollow mug with a curved handle; synthetic reference asset.",
            "scale_basis": "user_measurement",
            "assumptions": ["Demo dimensions are known by construction."],
            "semantics": {
                "name": "mug",
                "description": "Blue hollow mug with a curved side handle",
                "axes": [{"name": "up", "direction": {"x": 0, "y": 0, "z": 1}, "description": "up"}],
                "frames": [
                    {
                        "name": "mug_center_frame",
                        "role": "center",
                        "description": "Center of the mug's usable interior",
                        "position": {"x": 0, "y": 0, "z": 0.0525},
                        "size": {"x": 0.07, "y": 0.07, "z": 0.095},
                        "evidence": "Known synthetic inner radius and floor height.",
                    },
                    {
                        "name": "mug_handle_frame",
                        "role": "grasp",
                        "description": "Middle of the outer curved handle bar",
                        "position": {"x": 0.078, "y": 0, "z": 0.05},
                        "axes": [
                            {
                                "name": "bar",
                                "direction": {"x": 0, "y": 0, "z": 1},
                                "description": "the handle bar direction",
                            }
                        ],
                        "evidence": "Known synthetic handle centerline.",
                    },
                ],
            },
            "parts": [
                {
                    "name": "cup_body",
                    **material,
                    **pose,
                    "evidence": "Known synthetic cup cross section.",
                    "geometry": {
                        "kind": "lathe",
                        "profile": [
                            {"radius": r, "z": z}
                            for r, z in [
                                (0, 0),
                                (0.04, 0),
                                (0.04, 0.1),
                                (0.035, 0.1),
                                (0.035, 0.005),
                                (0, 0.005),
                            ]
                        ],
                    },
                },
                {
                    "name": "handle",
                    **material,
                    **pose,
                    "evidence": "Known synthetic handle centerline.",
                    "geometry": {
                        "kind": "sweep",
                        "radius": 0.005,
                        "points": [
                            {"x": x, "y": 0, "z": z}
                            for x, z in [
                                (0.043, 0.082),
                                (0.06, 0.082),
                                (0.074, 0.07),
                                (0.078, 0.05),
                                (0.074, 0.03),
                                (0.06, 0.018),
                                (0.043, 0.018),
                            ]
                        ],
                    },
                },
            ],
        }
    )


class DemoModel:
    """Scripted responses test orchestration; they do not infer a model from the video."""

    def __init__(self):
        self.proposals = 0
        self.api_calls = 0
        self.cache_hits = 0

    def complete(self, schema, instructions, prompt, images):
        if schema is SceneSpec:
            self.proposals += 1
            spec = demo_scene()
            if self.proposals == 1:
                data = spec.model_dump()
                data["parts"][0]["geometry"] = {"kind": "cylinder", "radius": 0.04, "height": 0.1}
                data["parts"][0]["position"]["z"] = 0.05
                return SceneSpec.model_validate(data)
            return spec
        corrected = self.proposals > 1
        return Review(
            score=0.95 if corrected else 0.4,
            acceptable=corrected,
            issues=[] if corrected else ["The cup is solid; it needs an open interior."],
            suggested_changes=[] if corrected else ["Use a lathe with inner and outer walls."],
            uncertainty=["Scripted demo score; not a real model evaluation."],
        )


def run_demo(out: Path) -> dict:
    out = fresh_directory(out)
    spec = demo_scene()
    mesh = build_asset(spec).mesh()
    write_json(out / "reference.json", spec)
    video = out / "turntable.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 8, (320, 320))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot encode the demo AVI with MJPG")
    try:
        for azimuth in np.linspace(-90, 270, 24, endpoint=False):
            frame = np.asarray(render_mesh(mesh, float(azimuth), 25))
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    workspace = out / "job"
    prepare(
        video,
        "A blue ceramic mug, 10 cm tall, with an open cup and one curved side handle.",
        workspace,
        frame_limit=8,
        scale=Scale(size_m=0.1, axis="z"),
    )
    state = reconstruct(workspace, RunConfig(iterations=2), PhysicsConfig(mass_kg=0.25), DemoModel())
    state["demo"] = "Scripted known-answer demo; no GPT call or video inference was performed."
    write_json(workspace / "run.json", state)
    return {"workspace": str(workspace), **state}
